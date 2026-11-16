"""Downsample attachment images before they get embedded in a prompt sent to
an Image GenAI model (REQUEST.md [1] step [4] / [3c]): "Prompt gồm text và
append ảnh sẽ có dung lượng khá lớn do ảnh. Do đó cần phải có bước xử lý ảnh
giảm mẫu, giảm phân giải về nhỏ hơn trước khi append vào prompt".

Resized copies are cached under `output/<request_name>/_resized_refs/` and
reused across models/versions of the SAME run (REQUEST.md [3d]: build once,
reuse), keyed by the ORIGINAL file's name + mtime so an edited source asset
is picked up again automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.core.output_manager import resized_refs_dir
from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("image_prep")


@dataclass
class PreparedImage:
    label: str
    source_path: Path
    resized_path: Path
    mime_type: str


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)


def _prepare_one(request_name: str, label: str, source: Path) -> PreparedImage | None:
    settings = get_settings()
    out_dir = resized_refs_dir(request_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        mtime_tag = int(source.stat().st_mtime)
    except OSError:
        mtime_tag = 0

    try:
        with Image.open(source) as img:
            img.load()
            keep_alpha = _has_alpha(img)
            ext = "png" if keep_alpha else "jpg"
            cache_name = f"{source.stem}__{mtime_tag}.{ext}"
            dest = out_dir / cache_name

            if dest.exists():
                logger.info("Reusing cached downsampled image for %s -> %s", source.name, dest.name)
                return PreparedImage(label=label, source_path=source, resized_path=dest, mime_type=f"image/{'png' if keep_alpha else 'jpeg'}")

            w, h = img.size
            max_dim = settings.attach_image_max_dim
            scale = min(1.0, max_dim / max(w, h)) if max(w, h) > max_dim else 1.0
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

            if keep_alpha:
                img = img.convert("RGBA")
                img.save(dest, format="PNG", optimize=True)
            else:
                img = img.convert("RGB")
                img.save(dest, format="JPEG", quality=settings.attach_image_jpeg_quality, optimize=True)

            logger.info(
                "Downsampled %s (%dx%d) -> %s (%dx%d, %.1f KB)",
                source.name,
                w,
                h,
                dest.name,
                img.size[0],
                img.size[1],
                dest.stat().st_size / 1024,
            )
            return PreparedImage(label=label, source_path=source, resized_path=dest, mime_type=f"image/{'png' if keep_alpha else 'jpeg'}")
    except Exception:  # noqa: BLE001 -- never let 1 bad asset kill the whole run
        logger.exception("Failed to downsample image %s -- skipping this attachment", source)
        return None


def prepare_attachments(request_name: str, labeled_paths: list[tuple[str, Path]]) -> list[PreparedImage]:
    """`labeled_paths`: e.g. `[("theme", Path(...)), ("context", Path(...)), ...]`.
    Order is preserved; entries that fail to process are dropped (logged, not fatal)."""
    prepared: list[PreparedImage] = []
    for label, path in labeled_paths:
        item = _prepare_one(request_name, label, path)
        if item is not None:
            prepared.append(item)
    return prepared
