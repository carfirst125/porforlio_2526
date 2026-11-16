"""Thin REST client — webapp Streamlit KHÔNG gọi thẳng vào app/core/*, mà
luôn đi qua FastAPI backend qua HTTP (REQUEST.md [6])."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests  # noqa: E402

from app.settings import get_settings  # noqa: E402

# UPDATE 2026-08-15: user report "chạy Generate xong CÓ ảnh trên đĩa nhưng
# streamlit không hiển thị gì" -- root cause: webapp trước đây tự
# `st.image(output_path_tra_ve_tu_API)`, ngầm giả định Streamlit process đọc
# thẳng được filesystem của backend. Giờ luôn build URL qua `/files` (backend
# serve lại đúng file đó qua HTTP, xem `app/api/routes.py::get_file`) rồi
# `st.image(url)` -- webapp không còn cần filesystem access vào backend nữa.
GENERATE_TIMEOUT_SECONDS = 1800  # sinh ảnh (đặc biệt mode=all, nhiều model/version) có thể mất nhiều phút


def _base_url() -> str:
    return get_settings().backend_base_url.rstrip("/")


def image_url(path: str) -> str:
    """Build URL phục vụ 1 ảnh qua `GET /files` -- dùng làm nguồn cho
    `st.image()` thay vì mở thẳng path cục bộ (xem comment ở đầu file)."""
    return f"{_base_url()}/files?path={quote(path)}"


def _get(path: str, **kwargs) -> dict:
    try:
        resp = requests.get(f"{_base_url()}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"GET {path} failed: {exc}"}


def _post(path: str, json_body: dict, timeout: int = 600) -> dict:
    try:
        resp = requests.post(f"{_base_url()}{path}", json=json_body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"POST {path} failed: {exc}"}


def health() -> dict:
    return _get("/health")


def list_models() -> dict:
    return _get("/models")


def list_requests() -> dict:
    return _get("/requests")


def get_request(request_name: str) -> dict:
    return _get(f"/requests/{request_name}")


def list_personas(request_name: str | None = None) -> dict:
    params = {"request_name": request_name} if request_name else None
    return _get("/personas", params=params, timeout=10)


def list_outputs(request_name: str, since_ts: float) -> dict:
    """Quét ảnh MỚI xuất hiện trong `output/<request_name>/` kể từ `since_ts`
    (epoch giây, xem `time.time()`/`datetime.timestamp()`) -- dùng để POLL
    trong lúc chờ `/generate` (blocking, có thể tới 30 phút) chạy xong thay
    vì chỉ đợi response cuối cùng, xem `app/api/routes.py::list_outputs` +
    `webapp/views/generate.py`."""
    return _get(f"/outputs/{request_name}", params={"since_ts": since_ts}, timeout=10)


def get_latest_outputs(request_name: str) -> dict:
    """Lấy các ảnh thuộc lượt chạy (Run) GẦN NHẤT hiện có trên đĩa cho
    request này -- khác `list_outputs()` ở chỗ KHÔNG phụ thuộc session hiện
    tại (không cần `since_ts`), nên vẫn hiện đúng kết quả sau khi reload
    trang / mở session Streamlit mới, xem `app/api/routes.py::latest_outputs`."""
    return _get(f"/outputs/{request_name}/latest", timeout=10)


def get_outputs_at_timestamp(request_name: str, ts: str) -> dict:
    """Lấy đúng các ảnh có hậu tố timestamp khớp `ts` (do user tự gõ vào ô
    text ở Generate page) -- ghi đè lên rule tự động (poll theo since_ts /
    gom cụm "lượt gần nhất"), dùng khi user muốn xem lại 1 lượt chạy CỤ THỂ,
    xem `app/api/routes.py::outputs_at_timestamp`."""
    return _get(f"/outputs/{request_name}/at-timestamp", params={"ts": ts}, timeout=10)


def generate(
    request_name: str,
    mode: str,
    custom_models: list[str] | None = None,
    num_version_override: int | None = None,
    generate_person_override: bool | None = None,
    persona_override: str | None = None,
) -> dict:
    return _post(
        "/generate",
        {
            "request_name": request_name,
            "mode": mode,
            "custom_models": custom_models or [],
            "num_version_override": num_version_override,
            "generate_person_override": generate_person_override,
            "persona_override": persona_override,
        },
        timeout=GENERATE_TIMEOUT_SECONDS,
    )


def clone(request_name: str, source_image_path: str, select_model: str) -> dict:
    return _post(
        "/clone",
        {"request_name": request_name, "source_image_path": source_image_path, "select_model": select_model},
        timeout=GENERATE_TIMEOUT_SECONDS,
    )


def release(request_name: str, selected_output_paths: list[str]) -> dict:
    return _post("/release", {"request_name": request_name, "selected_output_paths": selected_output_paths})


def get_released(request_name: str) -> dict:
    return _get(f"/release/{request_name}")


def get_clone_status(request_name: str, since_ts: float = 0.0, source_stem: str = "") -> dict:
    """Quét file clone (`*_clone_*.png`) trong `output/<request_name>/release/`
    -- `since_ts` lọc theo thời điểm bấm "Run clone" (dùng khi POLL 1 lượt
    clone đang chạy), `source_stem` lọc theo đúng 1 ảnh nguồn (dùng khi hiện
    lại các file clone ĐÃ CÓ SẴN của ảnh đang chọn), xem
    `app/api/routes.py::clone_status`."""
    return _get(
        f"/release/{request_name}/clone-status",
        params={"since_ts": since_ts, "source_stem": source_stem},
        timeout=10,
    )


def clone_from_release(request_name: str, released_image_path: str, display_sizes: list[str], image_model: str) -> dict:
    return _post(
        "/clone-from-release",
        {
            "request_name": request_name,
            "released_image_path": released_image_path,
            "display_sizes": display_sizes,
            "image_model": image_model,
        },
        timeout=GENERATE_TIMEOUT_SECONDS,
    )
