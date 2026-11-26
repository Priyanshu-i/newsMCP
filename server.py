import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import httpx
import pymupdf as fitz
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("upsc-pdf-annotator")

BASE_TMP_DIR = Path(os.getenv("ANNOTATED_FILES_DIR", "/tmp/upsc_mcp_storage"))
UPLOADS_DIR = BASE_TMP_DIR / "uploads"
ANNOTATED_DIR = BASE_TMP_DIR / "annotated"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", 24 * 60 * 60))  # 24 hours
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", 50 * 1024 * 1024))     # 50 MB


# ---------------------------------------------------------------------------
# Pydantic Schema for Structured Annotation Instructions
# ---------------------------------------------------------------------------
class AnnotationItem(BaseModel):
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number in the PDF.",
    )
    action: Literal["box", "highlight", "underline", "cross", "note"] = Field(
        ...,
        description=(
            "Visual annotation action: "
            "'box' (green border + top-left UPSC tag for must-read articles), "
            "'highlight' (transparent yellow highlight over key facts/data/schemes), "
            "'underline' (clean blue underline under key terms/definitions), "
            "'cross' (large red diagonal X across irrelevant sections to skip), "
            "'note' (interactive PDF sticky note explaining UPSC relevance)."
        ),
    )
    coordinates: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalized bounding box [ymin, xmin, ymax, xmax] on a 0 to 1000 scale.",
    )
    target_text: Optional[str] = Field(
        default=None,
        description=(
            "Exact snippet of text from the PDF to snap onto. Strongly recommended for "
            "'highlight' and 'underline' (exact phrase/number/term) and helpful for 'box' "
            "(article headline) so PyMuPDF aligns to exact text coordinates."
        ),
    )
    label: Optional[str] = Field(
        default=None,
        description="Short UPSC tag for 'box' (e.g., '[GS2 - Polity]', '[Prelims]', '[Editorial]').",
    )
    note_text: Optional[str] = Field(
        default=None,
        description="Detailed explanation for 'note' sticky notes (why it matters for UPSC & what to focus on).",
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, coords: list[float]) -> list[float]:
        ymin, xmin, ymax, xmax = coords
        if not all(0.0 <= v <= 1000.0 for v in coords):
            raise ValueError("All coordinates [ymin, xmin, ymax, xmax] must be between 0 and 1000.")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("Invalid bounding box: require ymin < ymax and xmin < xmax.")
        return coords


# ---------------------------------------------------------------------------
# File Storage & Automatic 24-Hour Cleanup
# ---------------------------------------------------------------------------
def cleanup_expired_files() -> int:
    """Delete uploaded and annotated PDFs older than FILE_TTL_SECONDS."""
    now = time.time()
    deleted = 0
    for folder in (UPLOADS_DIR, ANNOTATED_DIR):
        for file_path in folder.glob("*.pdf"):
            try:
                if now - file_path.stat().st_mtime > FILE_TTL_SECONDS:
                    file_path.unlink(missing_ok=True)
                    deleted += 1
            except OSError as exc:
                logger.warning("Failed to remove expired file %s: %s", file_path, exc)
    if deleted:
        logger.info("Cleaned up %d expired PDF file(s).", deleted)
    return deleted


async def periodic_cleanup_worker() -> None:
    """Background task running hourly to purge files older than 24 hours."""
    while True:
        try:
            cleanup_expired_files()
        except Exception as exc:
            logger.exception("Error in periodic cleanup worker: %s", exc)
        await asyncio.sleep(3600)


def resolve_public_base_url(ctx: Optional[Context] = None, request: Optional[Request] = None) -> str:
    explicit_url = os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    req = request
    if req is None and ctx is not None:
        try:
            req = ctx.request_context.request
        except Exception:
            req = None

    if req is not None:
        proto = req.headers.get("x-forwarded-proto", "https")
        host = req.headers.get("x-forwarded-host") or req.headers.get("host")
        if host:
            return f"{proto}://{host}".rstrip("/")

    return "http://localhost:8000"


# ---------------------------------------------------------------------------
# PDF Loading & Geometry Snapping Engine (PyMuPDF)
# ---------------------------------------------------------------------------
async def load_pdf_bytes(pdf_url: Optional[str], pdf_base64: Optional[str]) -> bytes:
    if bool(pdf_url) == bool(pdf_base64):
        raise ValueError("Provide exactly one of 'pdf_url' or 'pdf_base64'.")

    if pdf_base64:
        raw_b64 = pdf_base64.split(",", 1)[1] if "," in pdf_base64 else pdf_base64
        try:
            data = base64.b64decode(raw_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 PDF payload: {exc}") from exc
        if len(data) > MAX_PDF_BYTES:
            raise ValueError(f"Decoded PDF exceeds maximum allowed size ({MAX_PDF_BYTES} bytes).")
        return data

    assert pdf_url is not None

    if "/uploads/" in pdf_url:
        candidate_name = pdf_url.rsplit("/uploads/", 1)[-1].split("?", 1)[0]
        if "/" not in candidate_name and "\\" not in candidate_name and candidate_name.endswith(".pdf"):
            local_path = (UPLOADS_DIR / candidate_name).resolve()
            if local_path.parent == UPLOADS_DIR.resolve() and local_path.is_file():
                return local_path.read_bytes()

    async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as client:
        resp = await client.get(pdf_url)
        resp.raise_for_status()
        if len(resp.content) > MAX_PDF_BYTES:
            raise ValueError(f"Downloaded PDF exceeds maximum allowed size ({MAX_PDF_BYTES} bytes).")
        return resp.content


def normalized_to_pdf_rect(coords: list[float], page: fitz.Page) -> fitz.Rect:
    ymin, xmin, ymax, xmax = coords
    pw = page.rect.width
    ph = page.rect.height
    return fitz.Rect(
        (xmin / 1000.0) * pw,
        (ymin / 1000.0) * ph,
        (xmax / 1000.0) * pw,
        (ymax / 1000.0) * ph,
    )


def pdf_rect_to_normalized(rect: fitz.Rect, page: fitz.Page) -> list[int]:
    pw = page.rect.width or 1.0
    ph = page.rect.height or 1.0
    return [
        max(0, min(1000, round((rect.y0 / ph) * 1000))),
        max(0, min(1000, round((rect.x0 / pw) * 1000))),
        max(0, min(1000, round((rect.y1 / ph) * 1000))),
        max(0, min(1000, round((rect.x1 / pw) * 1000))),
    ]


def clamp_rect_to_page(rect: fitz.Rect, page: fitz.Page, margin: float = 6.0) -> fitz.Rect:
    """Keep bounding boxes strictly inside visible page boundaries."""
    x0 = max(margin, min(rect.x0, page.rect.width - margin - 10.0))
    y0 = max(margin, min(rect.y0, page.rect.height - margin - 10.0))
    x1 = max(x0 + 10.0, min(rect.x1, page.rect.width - margin))
    y1 = max(y0 + 10.0, min(rect.y1, page.rect.height - margin))
    return fitz.Rect(x0, y0, x1, y1)


def find_best_text_quads(
    page: fitz.Page,
    target_text: str,
    approx_rect: fitz.Rect,
) -> list[fitz.Quad]:
    """
    Search exact text on the PDF page and pick the occurrence closest to approx_rect.
    """
    cleaned = " ".join(target_text.split()).strip()
    if not cleaned:
        return []

    quads = page.search_for(cleaned, quads=True)
    if not quads and len(cleaned) > 35:
        # Fallback to first 35 chars if multi-line hyphenation broke exact match
        quads = page.search_for(cleaned[:35].rsplit(" ", 1)[0], quads=True)

    if not quads:
        return []

    # Filter quads whose center falls inside or very close to approx_rect
    expanded = fitz.Rect(
        approx_rect.x0 - 45,
        approx_rect.y0 - 45,
        approx_rect.x1 + 45,
        approx_rect.y1 + 45,
    )
    nearby = [q for q in quads if expanded.contains((q.rect.tl + q.rect.br) * 0.5)]
    return nearby if nearby else quads


def snap_section_rect(
    page: fitz.Page,
    approx_rect: fitz.Rect,
    headline_text: Optional[str] = None,
) -> fitz.Rect:
    """
    Snap a rough article/section box onto actual PDF text blocks so boxes align
    with newspaper columns and don't cut halfway through paragraphs or spill off-page.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [
        fitz.Rect(b[0], b[1], b[2], b[3])
        for b in blocks
        if b[6] == 0 and b[4].strip()
    ]

    if not text_blocks:
        return clamp_rect_to_page(approx_rect, page)

    # If headline_text is given, anchor top edge to the matched headline block
    top_anchor = approx_rect.y0
    if headline_text:
        h_quads = find_best_text_quads(page, headline_text, approx_rect)
        if h_quads:
            top_anchor = min(q.rect.y0 for q in h_quads)

    search_box = fitz.Rect(
        approx_rect.x0,
        min(approx_rect.y0, top_anchor),
        approx_rect.x1,
        approx_rect.y1,
    )

    # Keep text blocks whose horizontal & vertical centers fall inside search_box
    matched: list[fitz.Rect] = []
    for b_rect in text_blocks:
        cx = (b_rect.x0 + b_rect.x1) * 0.5
        cy = (b_rect.y0 + b_rect.y1) * 0.5
        if search_box.x0 <= cx <= search_box.x1 and search_box.y0 <= cy <= search_box.y1:
            matched.append(b_rect)

    if not matched:
        return clamp_rect_to_page(approx_rect, page)

    union = fitz.Rect(matched[0])
    for r in matched[1:]:
        union.include_rect(r)

    pad = 4.0
    snapped = fitz.Rect(union.x0 - pad, union.y0 - pad, union.x1 + pad, union.y1 + pad)
    return clamp_rect_to_page(snapped, page)


def snap_line_words(page: fitz.Page, approx_rect: fitz.Rect) -> list[fitz.Rect]:
    """
    When target_text isn't provided for highlight/underline, snap to actual words
    overlapping approx_rect and group them into clean per-line rectangles.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    selected = []
    for w in words:
        w_rect = fitz.Rect(w[0], w[1], w[2], w[3])
        cy = (w_rect.y0 + w_rect.y1) * 0.5
        cx = (w_rect.x0 + w_rect.x1) * 0.5
        if approx_rect.x0 <= cx <= approx_rect.x1 and approx_rect.y0 <= cy <= approx_rect.y1:
            selected.append((w[5], w[6], w_rect))

    if not selected:
        return []

    lines: dict[tuple[int, int], fitz.Rect] = {}
    for block_no, line_no, w_rect in selected:
        key = (block_no, line_no)
        if key not in lines:
            lines[key] = fitz.Rect(w_rect)
        else:
            lines[key].include_rect(w_rect)
    return list(lines.values())


# ---------------------------------------------------------------------------
# Visual Rendering Engine
# ---------------------------------------------------------------------------
def apply_annotation(page: fitz.Page, item: AnnotationItem) -> None:
    raw_rect = clamp_rect_to_page(normalized_to_pdf_rect(item.coordinates, page), page)

    if item.action == "box":
        rect = snap_section_rect(page, raw_rect, headline_text=item.target_text)
        green = (0.06, 0.52, 0.22)
        page.draw_rect(rect, color=green, width=1.8, overlay=True)

        tag_text = (item.label or "[UPSC Must-Read]").strip()
        if not tag_text.startswith("["):
            tag_text = f"[{tag_text}]"

        font_size = 8.0
        text_width = fitz.get_text_length(tag_text, fontname="hebo", fontsize=font_size)
        pad_x = 4.0
        badge_h = 12.5
        badge_w = min(text_width + (pad_x * 2), page.rect.width - rect.x0 - 4.0)

        badge_y0 = rect.y0 - badge_h if rect.y0 >= badge_h + 4.0 else rect.y0
        badge_rect = fitz.Rect(rect.x0, badge_y0, rect.x0 + badge_w, badge_y0 + badge_h)
        page.draw_rect(badge_rect, color=green, fill=green, overlay=True)
        page.insert_text(
            fitz.Point(badge_rect.x0 + pad_x, badge_rect.y1 - 3.0),
            tag_text,
            fontname="hebo",
            fontsize=font_size,
            color=(1.0, 1.0, 1.0),
            overlay=True,
        )

    elif item.action == "highlight":
        yellow = (1.0, 0.92, 0.15)
        quads = find_best_text_quads(page, item.target_text, raw_rect) if item.target_text else []
        if quads:
            annot = page.add_highlight_annot(quads)
            annot.set_colors(stroke=yellow)
            annot.set_opacity(0.45)
            annot.update()
        else:
            line_rects = snap_line_words(page, raw_rect) or [raw_rect]
            for lr in line_rects:
                annot = page.add_highlight_annot(lr)
                annot.set_colors(stroke=yellow)
                annot.set_opacity(0.45)
                annot.update()

    elif item.action == "underline":
        blue = (0.08, 0.36, 0.84)
        quads = find_best_text_quads(page, item.target_text, raw_rect) if item.target_text else []
        if quads:
            for q in quads:
                r = q.rect
                page.draw_line(
                    fitz.Point(r.x0, r.y1 + 0.8),
                    fitz.Point(r.x1, r.y1 + 0.8),
                    color=blue,
                    width=1.4,
                    overlay=True,
                )
        else:
            line_rects = snap_line_words(page, raw_rect)
            if line_rects:
                for lr in line_rects:
                    page.draw_line(
                        fitz.Point(lr.x0, lr.y1 + 0.8),
                        fitz.Point(lr.x1, lr.y1 + 0.8),
                        color=blue,
                        width=1.4,
                        overlay=True,
                    )
            else:
                page.draw_line(
                    fitz.Point(raw_rect.x0, raw_rect.y1),
                    fitz.Point(raw_rect.x1, raw_rect.y1),
                    color=blue,
                    width=1.4,
                    overlay=True,
                )

    elif item.action == "cross":
        rect = snap_section_rect(page, raw_rect, headline_text=item.target_text)
        red = (0.85, 0.12, 0.12)
        page.draw_rect(rect, color=red, width=0.8, stroke_opacity=0.45, overlay=True)
        page.draw_line(
            fitz.Point(rect.x0, rect.y0),
            fitz.Point(rect.x1, rect.y1),
            color=red,
            width=2.0,
            overlay=True,
        )
        page.draw_line(
            fitz.Point(rect.x0, rect.y1),
            fitz.Point(rect.x1, rect.y0),
            color=red,
            width=2.0,
            overlay=True,
        )

    elif item.action == "note":
        rect = snap_section_rect(page, raw_rect, headline_text=item.target_text)
        content = (item.note_text or item.label or "Important UPSC reference.").strip()
        if item.label and item.note_text:
            content = f"{item.label}\n\n{item.note_text}"

        note_point = fitz.Point(
            min(max(rect.x0 + 2.0, 6.0), page.rect.width - 24.0),
            min(max(rect.y0 + 2.0, 6.0), page.rect.height - 24.0),
        )
        annot = page.add_text_annot(note_point, content, icon="Note")
        annot.set_colors(stroke=(0.96, 0.68, 0.05))
        annot.update()


# ---------------------------------------------------------------------------
# FastMCP Server Definition (Stateless HTTP)
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="UPSC Newspaper PDF Annotator",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


@mcp.tool()
async def get_pdf_layout(
    pdf_url: Optional[str] = None,
    pdf_base64: Optional[str] = None,
    pages: Optional[list[int]] = None,
) -> dict:
    """
    Extract exact normalized [ymin, xmin, ymax, xmax] (0-1000) bounding boxes of
    text blocks and headlines per page so annotations align cleanly with newspaper columns.
    """
    pdf_bytes = await load_pdf_bytes(pdf_url=pdf_url, pdf_base64=pdf_base64)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = len(doc)
        target_pages = pages if pages else list(range(1, total_pages + 1))
        layout = []
        for pno in target_pages:
            if pno < 1 or pno > total_pages:
                continue
            page = doc[pno - 1]
            blocks_out = []
            for b in page.get_text("blocks"):
                if b[6] != 0:
                    continue
                txt = " ".join(b[4].split()).strip()
                if not txt:
                    continue
                coords = pdf_rect_to_normalized(fitz.Rect(b[0], b[1], b[2], b[3]), page)
                blocks_out.append({
                    "coordinates": coords,
                    "text_preview": txt[:140],
                })
            layout.append({"page_number": pno, "blocks": blocks_out})
        return {"total_pages": total_pages, "pages": layout}
    finally:
        doc.close()


@mcp.tool()
async def annotate_pdf_pages(
    annotations: list[AnnotationItem],
    pdf_url: Optional[str] = None,
    pdf_base64: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """
    Visually annotate a newspaper or study PDF for UPSC preparation and return
    an instant clickable HTTPS link to open the annotated PDF in Chrome.

    Tip for razor-sharp alignment:
    - Always provide 'target_text' (exact phrase/number for highlight/underline,
      or exact headline for box/cross) alongside 'coordinates' [ymin, xmin, ymax, xmax].
    """
    cleanup_expired_files()

    if not annotations:
        raise ValueError("The 'annotations' list cannot be empty.")

    pdf_bytes = await load_pdf_bytes(pdf_url=pdf_url, pdf_base64=pdf_base64)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to parse input as a valid PDF document: {exc}") from exc

    total_pages = len(doc)
    applied_count = 0
    skipped_items: list[str] = []

    try:
        for idx, item in enumerate(annotations):
            if item.page_number > total_pages:
                skipped_items.append(
                    f"Item #{idx + 1}: page_number {item.page_number} exceeds total pages ({total_pages})."
                )
                continue

            page = doc[item.page_number - 1]
            apply_annotation(page, item)
            applied_count += 1

        file_id = f"{uuid.uuid4().hex}.pdf"
        output_path = ANNOTATED_DIR / file_id
        doc.save(str(output_path), garbage=3, deflate=True)
    finally:
        doc.close()

    base_url = resolve_public_base_url(ctx=ctx)
    download_url = f"{base_url}/files/{file_id}"

    return {
        "status": "success",
        "annotated_pdf_url": download_url,
        "total_pages": total_pages,
        "annotations_applied": applied_count,
        "skipped": skipped_items,
        "expires_in_hours": round(FILE_TTL_SECONDS / 3600, 1),
    }


# ---------------------------------------------------------------------------
# Web Upload UI & HTTP File Serving Routes
# ---------------------------------------------------------------------------
UPLOAD_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UPSC Newspaper PDF Uploader — MCP Bridge</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 16px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 28px; max-width: 560px; width: 100%; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }
    h1 { font-size: 1.35rem; margin: 0 0 8px; }
    p { color: #94a3b8; font-size: 0.92rem; line-height: 1.5; margin: 0 0 20px; }
    .dropzone { border: 2px dashed #475569; border-radius: 10px; padding: 32px 16px; text-align: center; cursor: pointer; transition: 0.2s; background: #0f172a; }
    .dropzone:hover, .dropzone.drag { border-color: #22c55e; background: #052e16; }
    input[type=file] { display: none; }
    .result { display: none; margin-top: 20px; background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
    textarea { width: 100%; height: 140px; background: #1e293b; color: #e2e8f0; border: 1px solid #475569; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.82rem; box-sizing: border-box; resize: none; }
    button { margin-top: 10px; width: 100%; background: #16a34a; color: white; border: none; padding: 11px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; }
    button:hover { background: #15803d; }
    .status { margin-top: 12px; font-size: 0.88rem; color: #38bdf8; }
  </style>
</head>
<body>
  <div class="card">
    <h1>UPSC Newspaper PDF Annotator</h1>
    <p>1. Attach your newspaper PDF in Gemini chat.<br>2. Drop the same PDF below to generate a temporary link (auto-deletes in 24h) and paste the copied prompt into Gemini.</p>
    <div class="dropzone" id="dz" onclick="document.getElementById('fileInput').click()">
      <strong>Click or drag &amp; drop PDF here</strong><br><span style="font-size:0.82rem;color:#64748b;">Max 50 MB</span>
      <input type="file" id="fileInput" accept="application/pdf" />
    </div>
    <div class="status" id="status"></div>
    <div class="result" id="result">
      <div style="font-size:0.82rem;color:#4ade80;margin-bottom:8px;">Ready! Paste this into your Gemini chat:</div>
      <textarea id="promptBox" readonly></textarea>
      <button onclick="copyPrompt()">Copy Prompt for Gemini</button>
    </div>
  </div>
  <script>
    const dz = document.getElementById('dz');
    const fi = document.getElementById('fileInput');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    const promptBox = document.getElementById('promptBox');

    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
    dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });
    fi.addEventListener('change', () => { if (fi.files[0]) uploadFile(fi.files[0]); });

    async function uploadFile(file) {
      status.textContent = 'Uploading ' + file.name + '...';
      result.style.display = 'none';
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');
        status.textContent = 'Uploaded (' + data.pages + ' pages).';
        promptBox.value = `Annotate the attached newspaper PDF for UPSC CSE Prelims & Mains using pdf_url="${data.pdf_url}". First call get_pdf_layout to read exact column/block coordinates, then call annotate_pdf_pages (always passing exact target_text for every highlight, underline, and article headline) to add green box tags ([Prelims], [GS2 - Polity], [GS3 - Economy], [Editorial]) on must-read articles, yellow highlights on key facts/schemes/committees, blue underlines on key terms, red crosses on ads/politics/crime/sports, and sticky notes explaining UPSC relevance.`;
        result.style.display = 'block';
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      }
    }

    function copyPrompt() {
      promptBox.select();
      navigator.clipboard.writeText(promptBox.value);
      status.textContent = 'Copied to clipboard! Paste it into Gemini.';
    }
  </script>
</body>
</html>
"""


async def home_page(request: Request) -> HTMLResponse:
    return HTMLResponse(UPLOAD_PAGE_HTML)


async def upload_pdf(request: Request) -> JSONResponse:
    cleanup_expired_files()
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "No PDF file provided under form field 'file'."}, status_code=400)

    data = await upload.read()
    if not data:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)
    if len(data) > MAX_PDF_BYTES:
        return JSONResponse({"error": f"File exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit."}, status_code=413)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except Exception:
        return JSONResponse({"error": "Invalid PDF file."}, status_code=400)

    file_id = f"{uuid.uuid4().hex}.pdf"
    (UPLOADS_DIR / file_id).write_bytes(data)

    base_url = resolve_public_base_url(request=request)
    return JSONResponse({
        "status": "ok",
        "pdf_url": f"{base_url}/uploads/{file_id}",
        "pages": page_count,
        "expires_in_hours": round(FILE_TTL_SECONDS / 3600, 1),
    })


def _serve_pdf_from_dir(directory: Path, filename: str) -> FileResponse | JSONResponse:
    if "/" in filename or "\\" in filename or not filename.endswith(".pdf"):
        return JSONResponse({"error": "Invalid file name."}, status_code=400)

    file_path = (directory / filename).resolve()
    if file_path.parent != directory.resolve() or not file_path.is_file():
        return JSONResponse({"error": "PDF not found or expired."}, status_code=404)

    if time.time() - file_path.stat().st_mtime > FILE_TTL_SECONDS:
        file_path.unlink(missing_ok=True)
        return JSONResponse({"error": "PDF has expired (24h limit)."}, status_code=410)

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


async def serve_uploaded_pdf(request: Request) -> FileResponse | JSONResponse:
    return _serve_pdf_from_dir(UPLOADS_DIR, request.path_params["filename"])


async def serve_annotated_pdf(request: Request) -> FileResponse | JSONResponse:
    return _serve_pdf_from_dir(ANNOTATED_DIR, request.path_params["filename"])


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "upsc-pdf-annotator"})


mcp_http_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: Starlette):
    cleanup_task = asyncio.create_task(periodic_cleanup_worker())
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = Starlette(
    routes=[
        Route("/", home_page, methods=["GET"]),
        Route("/upload", upload_pdf, methods=["POST"]),
        Route("/health", health_check, methods=["GET"]),
        Route("/uploads/{filename:str}", serve_uploaded_pdf, methods=["GET"]),
        Route("/files/{filename:str}", serve_annotated_pdf, methods=["GET"]),
        Mount("/", app=mcp_http_app),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)