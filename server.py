import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pymupdf as fitz
import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("upsc-pdf-annotator")

FILES_DIR = Path(os.getenv("ANNOTATED_FILES_DIR", "/tmp/annotated_pdfs"))
FILES_DIR.mkdir(parents=True, exist_ok=True)

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
    """Delete annotated PDFs older than FILE_TTL_SECONDS."""
    now = time.time()
    deleted = 0
    for file_path in FILES_DIR.glob("*.pdf"):
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


def resolve_public_base_url(ctx: Optional[Context] = None) -> str:
    """
    Determine the public HTTPS base URL for clickable links.
    Priority:
      1. PUBLIC_BASE_URL env var (recommended for custom domains / Cloud Run)
      2. RENDER_EXTERNAL_URL env var (automatically injected by Render)
      3. Incoming HTTP request Host / X-Forwarded-Proto headers
    """
    explicit_url = os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    if ctx is not None:
        try:
            request = ctx.request_context.request
            if request is not None:
                proto = request.headers.get("x-forwarded-proto", "https")
                host = request.headers.get("x-forwarded-host") or request.headers.get("host")
                if host:
                    return f"{proto}://{host}".rstrip("/")
        except Exception:
            pass

    return "http://localhost:8000"


# ---------------------------------------------------------------------------
# PDF Loading & Visual Rendering Engine (PyMuPDF)
# ---------------------------------------------------------------------------
async def load_pdf_bytes(pdf_url: Optional[str], pdf_base64: Optional[str]) -> bytes:
    if bool(pdf_url) == bool(pdf_base64):
        raise ValueError("Provide exactly one of 'pdf_url' or 'pdf_base64'.")

    if pdf_base64:
        # Strip optional data URI prefix if passed (e.g., "data:application/pdf;base64,...")
        raw_b64 = pdf_base64.split(",", 1)[1] if "," in pdf_base64 else pdf_base64
        try:
            data = base64.b64decode(raw_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 PDF payload: {exc}") from exc
        if len(data) > MAX_PDF_BYTES:
            raise ValueError(f"Decoded PDF exceeds maximum allowed size ({MAX_PDF_BYTES} bytes).")
        return data

    assert pdf_url is not None
    async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as client:
        resp = await client.get(pdf_url)
        resp.raise_for_status()
        if len(resp.content) > MAX_PDF_BYTES:
            raise ValueError(f"Downloaded PDF exceeds maximum allowed size ({MAX_PDF_BYTES} bytes).")
        return resp.content


def normalized_to_pdf_rect(coords: list[float], page: fitz.Page) -> fitz.Rect:
    """Convert [ymin, xmin, ymax, xmax] (0..1000) to exact PyMuPDF page coordinates."""
    ymin, xmin, ymax, xmax = coords
    pw = page.rect.width
    ph = page.rect.height
    return fitz.Rect(
        (xmin / 1000.0) * pw,
        (ymin / 1000.0) * ph,
        (xmax / 1000.0) * pw,
        (ymax / 1000.0) * ph,
    )


def apply_annotation(page: fitz.Page, item: AnnotationItem) -> None:
    """Render a single UPSC visual annotation onto a PDF page."""
    rect = normalized_to_pdf_rect(item.coordinates, page)

    if item.action == "box":
        # Crisp emerald-green border around must-read UPSC article
        green = (0.06, 0.52, 0.22)
        page.draw_rect(rect, color=green, width=1.8, overlay=True)

        tag_text = (item.label or "[UPSC Must-Read]").strip()
        if not tag_text.startswith("["):
            tag_text = f"[{tag_text}]"

        font_size = 8.5
        text_width = fitz.get_text_length(tag_text, fontname="hebo", fontsize=font_size)
        pad_x = 4.0
        badge_h = 13.0
        badge_w = text_width + (pad_x * 2)

        # Position badge cleanly at top-left corner (inside top edge if near page top)
        badge_y0 = rect.y0 - badge_h if rect.y0 >= badge_h + 2 else rect.y0
        badge_rect = fitz.Rect(
            rect.x0,
            badge_y0,
            min(rect.x0 + badge_w, page.rect.width),
            badge_y0 + badge_h,
        )
        page.draw_rect(badge_rect, color=green, fill=green, overlay=True)
        page.insert_text(
            fitz.Point(badge_rect.x0 + pad_x, badge_rect.y1 - 3.2),
            tag_text,
            fontname="hebo",
            fontsize=font_size,
            color=(1.0, 1.0, 1.0),
            overlay=True,
        )

    elif item.action == "highlight":
        # True transparent yellow PDF highlight annotation
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1.0, 0.92, 0.15))
        annot.set_opacity(0.45)
        annot.update()

    elif item.action == "underline":
        # Clean blue underline along the bottom edge of the target bounding box
        blue = (0.08, 0.36, 0.84)
        p1 = fitz.Point(rect.x0, rect.y1)
        p2 = fitz.Point(rect.x1, rect.y1)
        page.draw_line(p1, p2, color=blue, width=1.6, overlay=True)

    elif item.action == "cross":
        # Large red diagonal "X" across irrelevant sections to skip
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
        # Interactive PDF sticky-note annotation at top-left of target region
        content = (item.note_text or item.label or "Important UPSC reference.").strip()
        if item.label and item.note_text:
            content = f"{item.label}\n\n{item.note_text}"

        note_point = fitz.Point(
            min(max(rect.x0, 4.0), page.rect.width - 20.0),
            min(max(rect.y0, 4.0), page.rect.height - 20.0),
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
)


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

    Args:
        annotations: List of page-by-page visual instructions with normalized
            [ymin, xmin, ymax, xmax] coordinates on a 0-1000 scale.
        pdf_url: Direct HTTP/HTTPS URL to the source PDF (provide either pdf_url or pdf_base64).
        pdf_base64: Base64-encoded PDF bytes (provide either pdf_url or pdf_base64).
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
        output_path = FILES_DIR / file_id
        doc.save(str(output_path), garbage=3, deflate=True)
    finally:
        doc.close()

    base_url = resolve_public_base_url(ctx)
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
# HTTP Routes (/files/<unique_id>.pdf, /health, and /mcp)
# ---------------------------------------------------------------------------
async def serve_annotated_pdf(request: Request) -> FileResponse | JSONResponse:
    filename = request.path_params["filename"]

    # Prevent path traversal; only allow flat <uuid>.pdf filenames
    if "/" in filename or "\\" in filename or not filename.endswith(".pdf"):
        return JSONResponse({"error": "Invalid file name."}, status_code=400)

    file_path = (FILES_DIR / filename).resolve()
    if file_path.parent != FILES_DIR.resolve() or not file_path.is_file():
        return JSONResponse({"error": "Annotated PDF not found or expired."}, status_code=404)

    # Check 24-hour expiration on access
    if time.time() - file_path.stat().st_mtime > FILE_TTL_SECONDS:
        file_path.unlink(missing_ok=True)
        return JSONResponse({"error": "Annotated PDF has expired (24h limit)."}, status_code=410)

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "upsc-pdf-annotator"})


# Build FastMCP's Streamable HTTP ASGI app (exposes /mcp)
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
        Route("/health", health_check, methods=["GET"]),
        Route("/files/{filename:str}", serve_annotated_pdf, methods=["GET"]),
        Mount("/", app=mcp_http_app),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)