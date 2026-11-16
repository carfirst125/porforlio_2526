#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
round_output_corners.py
========================

Quet toan bo anh trong thu muc `output/`, tim cac anh co do phan giai
CHINH XAC 1024x440, be tron 4 goc voi ban kinh 16px, va luu ket qua vao
thu muc `output_process/` (giu nguyen cau truc thu muc con nhu trong
`output/`).

AN TOAN:
    - Script chi MO anh trong `output/` de DOC (read-only). Khong bao gio
      ghi/de len bat ky file nao trong `output/`.
    - Tat ca file da xu ly duoc luu vao `output_process/`, mot thu muc
      hoan toan tach biet (tao moi neu chua co). Co them 1 lop kiem tra
      an toan (guard) truoc khi ghi de dam bao duong dan dich khong the
      nao nam trong thu muc `output/` goc.
    - Anh khong dung 1024x440 se bi BO QUA (khong copy, khong sua) --
      dung theo yeu cau chi xu ly anh dung kich thuoc nay.

VE DUNG LUONG & CHAT LUONG:
    - Anh sau khi be tron goc can channel alpha (trong suot) o 4 goc,
      nen luon duoc luu duoi dang PNG (RGBA) truoc tien, voi nen PNG
      toi da (optimize=True, compress_level=9).
    - Neu file PNG van > 500KB (mac dinh, chinh duoc qua --max-size-kb),
      script se tu dong fallback sang WEBP (van giu duoc alpha, nen tot
      hon PNG rat nhieu voi anh chup/photo) va giam dan quality (90 -> 30)
      cho den khi dat duoi nguong dung luong. Trong truong hop nay, DUOI
      FILE cua anh output se la .webp thay vi .png/.jpg goc -- vi JPG
      khong ho tro alpha nen KHONG the dung de luu anh da be goc.
    - Neu ca o quality=30 (WEBP) van chua dat duoi nguong, script van
      giu lai file co dung luong nho nhat co the va se canh bao ro trong
      log cuoi cung, KHONG lam hong / bo qua anh do.

CACH DUNG:
    python round_output_corners.py
    python round_output_corners.py --radius 16 --max-size-kb 500
    python round_output_corners.py --input /path/to/output --output /path/to/output_process
    python round_output_corners.py --dry-run   # chi liet ke, khong ghi file

Yeu cau: pip install Pillow
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    print(
        "[LOI] Khong tim thay thu vien Pillow. Hay cai dat truoc khi chay script:\n"
        "    pip install Pillow\n",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------
# Cau hinh mac dinh
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # version_2h_me090926/

DEFAULT_INPUT_DIR = PROJECT_DIR / "output"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output_process"

# Mac dinh ban kinh bo goc theo do phan giai
SIZE_RADIUS_MAP = {
    (1024, 440): 16,
    (720, 1280): 60,
}
DEFAULT_MAX_SIZE_KB = 500

# Cac duoi file anh se duoc quet (khong phan biet hoa/thuong)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Cac muc quality se thu lan luot khi fallback sang WEBP de giam dung luong
WEBP_QUALITY_STEPS = (90, 85, 80, 75, 70, 60, 50, 40, 30)

SUPERSAMPLE_FACTOR = 4  # de mask goc tron duoc muot (anti-alias) khi resize xuong


@dataclass
class RunStats:
    scanned: int = 0
    matched_size: int = 0
    processed_ok: int = 0
    skipped_wrong_size: int = 0
    skipped_unreadable: int = 0
    over_size_limit: int = 0
    fallback_webp_used: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Xu ly mask be tron goc
# --------------------------------------------------------------------------

def build_rounded_mask(size: tuple[int, int], radius_px: int) -> "Image.Image":
    """
    Tao mask "L" (grayscale, 0-255) co hinh chu nhat bo tron 4 goc, dung
    de gan lam alpha channel. Ve o kich thuoc lon hon (supersample) roi
    resize xuong bang LANCZOS de vien goc tron duoc muot, khong bi rang cua.
    """
    w, h = size
    scale = SUPERSAMPLE_FACTOR
    big_w, big_h = w * scale, h * scale
    r = max(0, radius_px * scale)

    mask = Image.new("L", (big_w, big_h), 0)
    draw = ImageDraw.Draw(mask)

    # Phan than chinh (rectangle ngang + rectangle doc), phu het tru 4 goc
    draw.rectangle([r, 0, big_w - r, big_h], fill=255)
    draw.rectangle([0, r, big_w, big_h - r], fill=255)

    # 4 hinh tron o 4 goc de tao duong bo tron
    draw.ellipse([0, 0, 2 * r, 2 * r], fill=255)                                   # top-left
    draw.ellipse([big_w - 2 * r, 0, big_w, 2 * r], fill=255)                       # top-right
    draw.ellipse([0, big_h - 2 * r, 2 * r, big_h], fill=255)                       # bottom-left
    draw.ellipse([big_w - 2 * r, big_h - 2 * r, big_w, big_h], fill=255)           # bottom-right

    mask = mask.resize((w, h), Image.LANCZOS)
    return mask


def apply_rounded_corners(img: "Image.Image", radius_px: int) -> "Image.Image":
    """Tra ve ban sao RGBA cua `img` voi 4 goc duoc be tron (trong suot)."""
    rgba = img.convert("RGBA")
    mask = build_rounded_mask(rgba.size, radius_px)

    # Neu anh goc da co alpha (hiem, vi day la banner quang cao), giu phan
    # trong suot goc bang cach lay min(mask, alpha_goc) -- thuc te hau het
    # anh dau vao la anh dac (alpha=255 toan bo) nen phep nhan nay khong
    # anh huong gi den phan chinh giua anh.
    orig_alpha = rgba.getchannel("A")
    from PIL import ImageChops
    combined_alpha = ImageChops.darker(mask, orig_alpha)

    rgba.putalpha(combined_alpha)
    return rgba


# --------------------------------------------------------------------------
# Luu anh voi rang buoc dung luong
# --------------------------------------------------------------------------

def save_under_size_limit(
    img: "Image.Image", out_path_no_ext: Path, max_size_bytes: int
) -> tuple[Path, int, bool]:
    """
    Thu luu `img` (RGBA) sao cho dung luong <= max_size_bytes.
    1) Thu PNG toi uu nen toi da truoc.
    2) Neu van vuot nguong, fallback sang WEBP, giam dan quality.

    Tra ve (duong_dan_thuc_te_da_luu, dung_luong_bytes, da_dat_nguong_hay_chua).
    """
    png_path = out_path_no_ext.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    img.save(png_path, format="PNG", optimize=True, compress_level=9)
    size_bytes = png_path.stat().st_size
    if size_bytes <= max_size_bytes:
        return png_path, size_bytes, True

    # --- Fallback: WEBP (van ho tro alpha, nen tot hon PNG voi anh photo) ---
    webp_path = out_path_no_ext.with_suffix(".webp")
    best_path = png_path
    best_size = size_bytes

    for quality in WEBP_QUALITY_STEPS:
        img.save(webp_path, format="WEBP", quality=quality, method=6)
        webp_size = webp_path.stat().st_size
        if webp_size < best_size:
            best_size = webp_size
            best_path = webp_path
        if webp_size <= max_size_bytes:
            # Da dat nguong -- xoa ban PNG qua khong dat de tranh luu 2 ban
            if png_path.exists() and best_path != png_path:
                png_path.unlink()
            return webp_path, webp_size, True

    # Khong quality nao dat duoi nguong -- giu lai ban nho nhat tung dat duoc
    if best_path == png_path and webp_path.exists():
        webp_path.unlink()
    elif best_path == webp_path and png_path.exists():
        png_path.unlink()

    return best_path, best_size, False


# --------------------------------------------------------------------------
# Duyet thu muc va xu ly
# --------------------------------------------------------------------------

def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def process_all(
    input_dir: Path,
    output_dir: Path,
    radius_override: int | None = None,
    filter_size: tuple[int, int] | None = None,
    max_size_kb: int = DEFAULT_MAX_SIZE_KB,
    dry_run: bool = False,
) -> RunStats:
    stats = RunStats()
    max_size_bytes = max_size_kb * 1024

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    for src_path in iter_image_files(input_dir):
        stats.scanned += 1
        rel_path = src_path.relative_to(input_dir)

        try:
            with Image.open(src_path) as im:
                im.load()
                width, height = im.size
                current_size = (width, height)

                if filter_size and current_size != filter_size:
                    stats.skipped_wrong_size += 1
                    continue

                if current_size not in SIZE_RADIUS_MAP and not filter_size and radius_override is None:
                    stats.skipped_wrong_size += 1
                    continue

                # Xac dinh ban kinh bo goc
                if radius_override is not None:
                    actual_radius = radius_override
                else:
                    actual_radius = SIZE_RADIUS_MAP.get(current_size, 16)

                stats.matched_size += 1
                print(f"[MATCH] {rel_path}  ({width}x{height}, radius={actual_radius}px)")

                if dry_run:
                    continue

                rounded = apply_rounded_corners(im, actual_radius)

                out_no_ext = output_dir / rel_path.with_suffix("")

                # Lop kiem tra an toan: dam bao duong dan dich KHONG BAO GIO
                # nam trong thu muc input goc (output/).
                resolved_out_parent = out_no_ext.parent.resolve()
                if input_dir == resolved_out_parent or input_dir in resolved_out_parent.parents:
                    raise RuntimeError(
                        f"Duong dan dich '{resolved_out_parent}' nam trong thu muc "
                        f"input goc '{input_dir}' -- da huy de bao ve anh goc."
                    )

                saved_path, size_bytes, reached_target = save_under_size_limit(
                    rounded, out_no_ext, max_size_bytes
                )

                stats.processed_ok += 1
                if saved_path.suffix.lower() == ".webp":
                    stats.fallback_webp_used += 1

                size_kb = size_bytes / 1024
                status = "OK" if reached_target else "VUOT NGUONG"
                print(
                    f"        -> {saved_path.relative_to(output_dir)}  "
                    f"({size_kb:.1f} KB, {status})"
                )
                if not reached_target:
                    stats.over_size_limit += 1

        except Exception as exc:  # noqa: BLE001 - log va tiep tuc, khong dung ca script
            stats.skipped_unreadable += 1
            msg = f"{rel_path}: {exc}"
            stats.errors.append(msg)
            print(f"[LOI] {msg}", file=sys.stderr)
            continue

    return stats


def print_summary(stats: RunStats, output_dir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("TONG KET")
    print("=" * 60)
    print(f"Tong so anh da quet          : {stats.scanned}")
    print(f"So anh khop kich thuoc       : {stats.matched_size}")
    if not dry_run:
        print(f"Da xu ly & luu thanh cong    : {stats.processed_ok}")
        print(f"  - trong do dung WEBP fallback: {stats.fallback_webp_used}")
        print(f"  - van vuot nguong dung luong : {stats.over_size_limit}")
    print(f"Bo qua (khong dung size)     : {stats.skipped_wrong_size}")
    print(f"Loi / khong doc duoc         : {stats.skipped_unreadable}")
    if stats.errors:
        print("\nChi tiet loi:")
        for e in stats.errors:
            print(f"  - {e}")
    if not dry_run:
        print(f"\nKet qua da xu ly duoc luu tai: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quet toan bo anh trong output/, be tron 4 goc (16px cho Native 1024x440, "
            "60px cho Popup 720x1280), luu vao output_process/ (khong dong den output/)."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Thu muc anh nguon (mac dinh: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Thu muc luu anh da xu ly (mac dinh: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--radius", type=int, default=None,
        help="Ghi de ban kinh be goc cho moi size (don vi px). Mac dinh: tu dong (16px cho 1024x440, 60px cho 720x1280).",
    )
    parser.add_argument(
        "--size", type=str, default=None,
        help="Chi xu ly anh co kich thuoc cu the (vi du: '1024x440' hoac '720x1280').",
    )
    parser.add_argument(
        "--max-size-kb", type=int, default=DEFAULT_MAX_SIZE_KB,
        help=f"Dung luong toi da moi anh, don vi KB (mac dinh: {DEFAULT_MAX_SIZE_KB})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi liet ke anh phu hop, khong ghi file nao ca.",
    )
    args = parser.parse_args()

    filter_size = None
    if args.size:
        parts = args.size.lower().replace("x", ",").split(",")
        filter_size = (int(parts[0].strip()), int(parts[1].strip()))

    if not args.input.exists():
        print(f"[LOI] Khong tim thay thu muc input: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Input : {args.input.resolve()}")
    print(f"Output: {args.output.resolve()}")
    radius_info = f"{args.radius}px (override)" if args.radius is not None else "Auto (1024x440->16px, 720x1280->60px)"
    size_info = f"{filter_size[0]}x{filter_size[1]}" if filter_size else "All supported (1024x440, 720x1280)"
    print(f"Radius: {radius_info} | Target Size: {size_info} | Max size: {args.max_size_kb}KB | Dry-run: {args.dry_run}\n")

    stats = process_all(
        input_dir=args.input,
        output_dir=args.output,
        radius_override=args.radius,
        filter_size=filter_size,
        max_size_kb=args.max_size_kb,
        dry_run=args.dry_run,
    )
    print_summary(stats, args.output, args.dry_run)


if __name__ == "__main__":
    main()

