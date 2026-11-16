"""Release view -- loaded by the router (`webapp/Home.py`) via `st.navigation()`.

UPDATE 2026-08-16 (theo yêu cầu user):
1. Khu vực nội dung bên phải: bỏ logo OCB khỏi tiêu đề trang, "Release" nằm
   sát biên đầu dòng -- giống hệt cách đã làm ở `webapp/views/generate.py`
   (`page_title_block(..., show_logo=False)`, không icon trong title).
2. Thêm bảng chọn Display size (multi-select dạng bảng, dùng
   `st.dataframe(..., on_select="rerun", selection_mode="multi-row")`) thay
   cho multiselect tự do + ô nhập size thủ công trước đây. 2 bảng do user
   cung cấp, chọn bằng 1 checkbox "Display size resolution: High" (mặc định
   TẮT -- hiện bảng "Low-cost Test Size" rẻ để test):
   - TẮT (mặc định): `_LOW_COST_SIZES` -- cột "Low-cost Test Size" (vd
     512x268) dùng luôn làm size clone.
   - BẬT: `_HIGH_RES_SIZES` -- có CẢ "Test Canvas Size" lẫn "Production
     Size" cho mỗi ID. Vì user không chỉ định rõ dùng cột nào khi bảng High
     đang hiện, thêm 1 radio "Clone at" (mặc định "Production size (final)",
     đúng tinh thần "Thực tế thì chạy với resolution High" user nói) để user
     tự chọn ngay lúc chạy demo thay vì hard-code -- linh hoạt hơn mà không
     cần hỏi lại.
3. "Expected clone files": suy ra TRƯỚC đường dẫn file clone sẽ nằm ở đâu,
   hoàn toàn từ tên ảnh nguồn + size đã chọn + model -- đúng convention thật
   `output_manager.clone_release_path()`
   ("<source_stem>_clone_<display_size>_<image_model>.png", nằm PHẲNG ngay
   trong `output/<request_name>/release/`, KHÔNG phải thư mục con riêng).
4. `POST /clone-from-release` giờ chạy trong `threading.Thread` nền (daemon)
   giống hệt Generate page (`generate.py`'s `_run_generate_job`), script
   chính poll endpoint mới `GET /release/{request_name}/clone-status`
   (`app/api/routes.py::clone_status` + `api_client.get_clone_status`) mỗi
   ~2s bằng `time.sleep()` + `st.rerun()` trong lúc chạy, và luôn hiện lại
   các file clone ĐÃ CÓ SẴN trên đĩa của ảnh đang chọn (lọc theo
   `source_stem`) khi không đang chạy -- không phụ thuộc session, cùng
   nguyên tắc đã áp dụng cho Generate page (`generate.py` UPDATE 2 & 3).
5. Ảnh clone hiển thị nhóm theo display_size -> image_model (`_render_clone_files`),
   đúng kiểu layout đã làm ở Generate page, và luôn là phần CUỐI của trang."""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils import api_client  # noqa: E402
from webapp.utils.theme import badge, card, image_stretched, page_title_block  # noqa: E402

page_title_block("Release", "Released images + clone into other sizes", show_logo=False)

# ---------------------------------------------------------------------------
# UPDATE 2026-09-14 (theo user báo lỗi ở Generate page -- cùng lớp lỗi có thể
# xảy ra ở đây): thêm health-check backend giống `generate.py` (xem comment
# dài ở đầu file đó) -- nếu quên chạy `uvicorn app.main:app --reload` song
# song với `streamlit run webapp/Home.py`, mọi `api_client.*` phía dưới sẽ
# fail với `ConnectionError`, dễ bị hiểu lầm thành lỗi request/data thay vì
# "backend chưa chạy". Chặn NGAY ĐẦU trang bằng 1 thông báo rõ ràng.
# ---------------------------------------------------------------------------
_health = api_client.health()
if not _health.get("ok"):
    st.error(
        f"⚠️ Could not connect to the backend API -- {_health.get('error')}\n\n"
        f"This page needs the FastAPI backend running alongside Streamlit. Start it first, then reload this page:\n\n"
        f"```\nuvicorn app.main:app --reload --reload-dir app\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Display size tables (do user cung cấp 2026-08-16) -- ID/Aspect Ratio giống
# nhau ở cả 2 bảng, chỉ khác cột size. "size"/"test_canvas"/"production" đều
# ở dạng "W,H" (khớp định dạng `OutputSpec.size` request.json dùng xuyên
# suốt app, xem `app/models/schemas.py`).
# ---------------------------------------------------------------------------
_LOW_COST_SIZES = [
    {"id": "LANDSCAPE", "ratio": "1.91:1", "size": "512,268"},
    {"id": "SQUARE", "ratio": "1:1", "size": "512,512"},
    {"id": "PORTRAIT", "ratio": "4:5", "size": "512,640"},
    {"id": "FULLSCREEN", "ratio": "9:16", "size": "512,910"},
    {"id": "APP_BANNER", "ratio": "2.5:1", "size": "512,205"},
    {"id": "APP_OVERLAY", "ratio": "9:16", "size": "512,910"},
]

_HIGH_RES_SIZES = [
    {"id": "LANDSCAPE", "ratio": "1.91:1", "test_canvas": "768,402", "production": "1200,628"},
    {"id": "SQUARE", "ratio": "1:1", "test_canvas": "768,768", "production": "1200,1200"},
    {"id": "PORTRAIT", "ratio": "4:5", "test_canvas": "768,960", "production": "1080,1350"},
    {"id": "FULLSCREEN", "ratio": "9:16", "test_canvas": "576,1024", "production": "1080,1920"},
    {"id": "APP_BANNER", "ratio": "2.5:1", "test_canvas": "640,256", "production": "750,300"},
    {"id": "APP_OVERLAY", "ratio": "9:16", "test_canvas": "576,1024", "production": "750,1334"},
]


def _to_wxh(size_str: str) -> str:
    """"W,H" -> "WxH" -- đúng convention filename thật của
    `output_manager.clone_release_path()` (`display_size = f"{width}x{height}"`)."""
    w, h = size_str.split(",")
    return f"{w.strip()}x{h.strip()}"


def _parse_custom_sizes(raw: str) -> tuple[list[str], list[str]]:
    """UPDATE 2026-08-17 (user request): parse the free-text "Custom display
    size(s)" box -- a comma-separated list of "WxH" tokens (e.g. "1024x1024,
    1536x1024") -- into the app's internal "W,H" size-string convention
    (same format `OutputSpec.size`/`_LOW_COST_SIZES`/`_HIGH_RES_SIZES` use
    everywhere else, so a custom size flows through `display_sizes` ->
    `effective_sizes` -> `POST /clone-from-release` exactly like a
    table-ticked one, no special-casing needed downstream).

    Returns (valid_sizes, invalid_tokens) instead of silently dropping bad
    input -- the caller shows `invalid_tokens` back to the user as a warning
    so a typo (e.g. "1024-1024" or "abcx100") doesn't just vanish."""
    valid: list[str] = []
    invalid: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            invalid.append(token)
            continue
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            invalid.append(token)
            continue
        if width <= 0 or height <= 0:
            invalid.append(token)
            continue
        valid.append(f"{width},{height}")
    return valid, invalid


def _render_clone_files(files: list[dict]) -> None:
    """Vẽ danh sách file clone (mỗi item có display_size/image_model/filename/
    path) theo block display_size -> image_model, cùng kiểu layout đã dùng ở
    Generate page (`webapp/views/generate.py::_render_grouped_images`)."""
    if not files:
        return

    sizes_order: list[str] = []
    by_size: dict[str, list[dict]] = {}
    for f in files:
        size = f.get("display_size") or "-"
        if size not in by_size:
            by_size[size] = []
            sizes_order.append(size)
        by_size[size].append(f)

    for size in sizes_order:
        st.markdown(f"##### {size}")
        models_order: list[str] = []
        by_model: dict[str, list[dict]] = {}
        for f in by_size[size]:
            model = f.get("image_model") or "-"
            if model not in by_model:
                by_model[model] = []
                models_order.append(model)
            by_model[model].append(f)

        for model in models_order:
            st.markdown(f"**{model}**")
            cols = st.columns(3)
            for i, f in enumerate(by_model[model]):
                with cols[i % 3]:
                    with card():
                        st.image(api_client.image_url(f["path"]), caption=f.get("filename", ""))


def _run_clone_job(job: dict, request_name: str, select_image: str, display_sizes: list[str], clone_model: str) -> None:
    """Chạy trong thread nền (daemon) -- `POST /clone-from-release` có thể
    block tới 30 phút (nhiều size). Ghi kết quả vào `job` (dict lưu trong
    `st.session_state`) để luồng chính đọc được mà không phải chờ, cùng
    pattern với `generate.py::_run_generate_job`."""
    resp = api_client.clone_from_release(request_name, select_image, display_sizes, clone_model)
    if "results" in resp:
        job["results"] = resp.get("results", [])
        job["error"] = None
    else:
        job["results"] = []
        job["error"] = resp.get("error", "Unknown error calling /clone-from-release")
    job["done"] = True


with st.sidebar:
    st.markdown("### 📄 Request")
    req_list = api_client.list_requests().get("requests", [])
    # UPDATE 2026-08-17 (user report): "khi tôi chọn bên phải, như select
    # model thì lại thấy bên sidebar request.json bị nhảy sang file khác."
    # Root cause: the widget had NO `key=`, so its value was re-derived from
    # `index=default_idx` on EVERY script rerun -- and `default_idx` was
    # computed fresh each time from `st.session_state["last_request_name"]`,
    # a value this page only ever READ, never WROTE. On some Streamlit
    # versions a re-passed `index=` argument re-asserts the widget's value
    # on every rerun (unlike a `key=`'d widget, whose value only changes via
    # direct user interaction) -- so ANY unrelated widget interaction
    # elsewhere on the page (e.g. picking "Image model") that triggers a
    # full-script rerun could silently snap "Request" back to whatever stale
    # value `last_request_name` still held, fighting the user's own in-page
    # choice. `generate.py`'s own "Input file request_name.json" selectbox
    # never has this problem because it has no `index=` recompute at all.
    #
    # Fix: give the widget an explicit `key=` so Streamlit itself owns and
    # persists the value across EVERY rerun (regardless of Streamlit
    # version-specific `index=` semantics) until the user picks something
    # else. `last_request_name` is used ONLY ONCE, to SEED the initial
    # value the very first time this page is opened in a session (nice
    # carry-over from the Generate page) -- never re-applied afterwards.
    if req_list:
        if "release_request_name" not in st.session_state:
            seed = st.session_state.get("last_request_name")
            st.session_state["release_request_name"] = seed if seed in req_list else req_list[0]
        request_name = st.selectbox("Request", req_list, key="release_request_name")
    else:
        # Cùng lớp lỗi (fallback path, hiếm khi vào -- chỉ khi input/ không
        # có file request*.json nào): `value=` được nạp lại MỚI mỗi lần
        # render từ `last_request_name`, không có `key=` -- cùng nguyên
        # nhân bug #48 nếu user gõ tay 1 tên khác rồi tương tác widget khác.
        # Seed-once + key, cùng pattern với nhánh `if req_list:` ở trên.
        if "release_request_name_manual" not in st.session_state:
            st.session_state["release_request_name_manual"] = st.session_state.get("last_request_name", "sample_request")
        request_name = st.text_input("Request name", key="release_request_name_manual")

    # Đồng bộ lại last_request_name để các trang khác (Generate) vẫn thấy
    # đúng lựa chọn gần nhất -- KHÔNG dùng ngược giá trị này để ép lại
    # selectbox mỗi lần render nữa (đó chính là nguồn gây bug ở trên).
    st.session_state["last_request_name"] = request_name

if not request_name:
    st.stop()

released_resp = api_client.get_released(request_name)
released_paths = [p for p in released_resp.get("released", []) if "_clone" not in Path(p).stem]

with card():
    st.markdown(f"#### 📦 Released images -- request: `{request_name}`")

    if not released_paths:
        st.info("No images released yet. Go to the Generate page, run it, then check 'Select for release'.")
        st.stop()

    cols = st.columns(3)
    for i, p in enumerate(released_paths):
        with cols[i % 3]:
            st.image(api_client.image_url(p), caption=Path(p).name)

st.divider()

# ---------------------------------------------------------------------------
# [8e] Clone a released image into other sizes
# ---------------------------------------------------------------------------
with card():
    st.markdown("#### 🧬 Clone a released image into other sizes")

    preview_col, form_col = st.columns([1, 2])
    with preview_col:
        # UPDATE 2026-08-17 (bug #50, user report): "chỗ Choose a released
        # image khi tôi chọn thì bên sidebar chỗ request bị load lại tên
        # request ban đầu." This selectbox had NO `key=` -- same latent-bug
        # shape flagged in bug #48 ("a selectbox with no key is a latent
        # sidebar-jump bug"). Its OPTIONS list (`released_paths`) also
        # changes with `request_name`, so a single shared `key=` across every
        # request would crash (`StreamlitAPIException`) the moment the
        # currently-selected path isn't in a different request's list --
        # scope the key PER REQUEST (same pattern as `sel_state_key` above)
        # so each request keeps its own remembered selection safely.
        select_image = st.selectbox(
            "Choose a released image",
            released_paths,
            format_func=lambda p: Path(p).name,
            key=f"release_select_image::{request_name}",
        )
        image_stretched(api_client.image_url(select_image), caption="Source image to clone")

    with form_col:
        # UPDATE 2026-08-17 (bug #54, user report): "khi tôi chọn một image
        # model bên phải thì bên sidebar bên trái giá trị request.json lại
        # bị reset." Rà soát TOÀN TRANG (theo yêu cầu user "xem xét tổng
        # thể"): đây là contributor thứ 3 của ĐÚNG cùng 1 lớp lỗi đã sửa ở
        # bug #48 (sidebar "Request") và bug #50 ("Choose a released
        # image") -- widget KHÔNG có `key=` là 1 lỗi tiềm ẩn trong app này
        # (đã 2 lần được xác nhận là nguyên nhân thật), và "Image model"
        # bên dưới đây chưa từng được đụng tới ở cả 2 lần sửa trước, dù nó
        # NẰM CÙNG TRANG và mọi tương tác của nó cũng kích hoạt full-script
        # rerun giống hệt các widget đã sửa. Thêm `key=` cho nó VÀ 2 widget
        # keyless còn lại trên trang (`high_res` checkbox, `clone_at`
        # radio) trong cùng 1 lần sửa này để đóng hoàn toàn lớp lỗi này
        # trên trang Release -- không để lại thêm 1 "contributor thứ 4"
        # nào lộ ra ở lần báo lỗi kế tiếp.
        models_info = api_client.list_models()
        available_models = models_info.get("available_providers", [])
        clone_model = st.selectbox("Image model", available_models, key="release_clone_model")

        high_res = st.checkbox(
            "Display size resolution: High",
            value=False,
            key="release_high_res",
            help=(
                "Off (default) -- low-cost small sizes for cheap test runs. "
                "On -- choose a higher-res test canvas or the full production size below."
            ),
        )

        if high_res:
            size_table = _HIGH_RES_SIZES
            clone_at = st.radio(
                "Clone at",
                ["Production size (final)", "Test canvas size (cheaper)"],
                horizontal=True,
                key="release_clone_at",
            )
            size_key = "production" if clone_at.startswith("Production") else "test_canvas"
            editor_key = "display_size_table_high"
            extra_cols = {
                "Test Canvas Size": "test_canvas",
                "Production Size": "production",
            }
        else:
            size_table = _LOW_COST_SIZES
            size_key = "size"
            editor_key = "display_size_table_low"
            extra_cols = {"Low-cost Test Size": "size"}

        # UPDATE 2026-08-16 (3): user report "tick chọn xong dấu tick tự biến
        # mất, bấm Run clone thì mất hết lựa chọn phía trên" -- `st.data_editor`
        # được build lại HOÀN TOÀN MỚI mỗi lần script rerun (kể cả rerun tự
        # động của vòng poll bên dưới trong lúc clone đang chạy), và không
        # nên tin widget tự khôi phục đúng ô đã tick qua NHIỀU lần rerun như
        # vậy chỉ dựa vào `key`. Giờ tự lưu tường minh tập ID đã tick vào
        # `st.session_state` (không hề bị xoá bởi bất kỳ hành động nào khác
        # trên trang, kể cả bấm "Run clone") -- bảng được SEED sẵn "Select"
        # từ tập này mỗi lần render, và mỗi lần user tick/bỏ tick thì đồng bộ
        # lại tập này ngay lập tức. Nhờ vậy lựa chọn không còn "biến mất" qua
        # rerun nữa. Key riêng cho 2 bảng Low/High nên đổi qua lại không lẫn
        # lộn giữa 2 tập lựa chọn.
        sel_state_key = f"release_selected_ids::{editor_key}"
        selected_ids: set = st.session_state.setdefault(sel_state_key, set())

        table_rows = [
            {
                "Select": row["id"] in selected_ids,
                "ID": row["id"],
                "Aspect Ratio": row["ratio"],
                **{label: _to_wxh(row[src_key]) for label, src_key in extra_cols.items()},
            }
            for row in size_table
        ]

        st.caption("Tick the checkbox in the \"Select\" column for every display size you want to clone.")
        other_cols = [c for c in table_rows[0].keys() if c != "Select"]
        edited_rows = st.data_editor(
            table_rows,
            use_container_width=True,
            hide_index=True,
            key=editor_key,
            disabled=other_cols,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select", help="Tick to include this display size", default=False
                ),
            },
        )

        # Đồng bộ lại tập ID đã tick TỪ NGAY kết quả edit vừa rồi -- nguồn sự
        # thật duy nhất cho "đã chọn size nào" từ đây trở đi là
        # `st.session_state[sel_state_key]`, không phải biến `edited_rows`
        # (biến này chỉ tồn tại trong lần rerun hiện tại).
        st.session_state[sel_state_key] = {row["ID"] for row in edited_rows if row.get("Select")}
        selected_ids = st.session_state[sel_state_key]
        selected_rows = [row for row in size_table if row["id"] in selected_ids]
        table_sizes = [row[size_key] for row in selected_rows]

        # UPDATE 2026-08-17 (user request): "Custom display size" free-text
        # box -- user types a comma-separated list of WxH values (e.g.
        # "1024x1024, 1536x1024") and they get added to the size list
        # alongside whatever's ticked in the table above, ALL cloned together
        # in one "Run clone" click. `st.text_input`'s own `key=` already
        # persists the typed value across every rerun by itself (unlike the
        # `st.data_editor` above, which needed the explicit session_state
        # workaround from bug #35) -- no extra bookkeeping needed here.
        custom_raw = st.text_input(
            "Custom display size(s)",
            key="release_custom_sizes_text",
            placeholder="e.g. 1024x1024, 1536x1024",
            help="Comma-separated list of extra sizes as WIDTHxHEIGHT. Added to the sizes ticked above (not a replacement for them).",
        )
        custom_sizes, invalid_tokens = _parse_custom_sizes(custom_raw)
        if invalid_tokens:
            st.warning(
                "Bỏ qua giá trị không hợp lệ (cần đúng dạng WxH, ví dụ 1024x1024): "
                + ", ".join(invalid_tokens)
            )

        # Gộp size đã tick trong bảng + size custom thành 1 danh sách, khử
        # trùng (cùng 1 "W,H" xuất hiện ở cả bảng lẫn ô custom chỉ clone 1
        # lần), giữ nguyên thứ tự thêm vào.
        display_sizes: list[str] = []
        for s in table_sizes + custom_sizes:
            if s not in display_sizes:
                display_sizes.append(s)

        # UPDATE 2026-08-16 (3): "ô Selected sizes" riêng, luôn đọc từ
        # session_state/text_input ở trên -- sống sót qua MỌI rerun kể cả sau
        # khi bấm "Run clone" (khác badge cũ vốn chỉ tồn tại trong cùng 1 lần
        # chạy script với thao tác tick). UPDATE 2026-08-17: size custom hiện
        # badge tone "gold" (khác "green" của size tick từ bảng) để phân biệt
        # trực quan nguồn gốc từng size trong danh sách gộp.
        st.markdown("**Selected sizes:**")
        if display_sizes:
            table_size_set = {row[size_key] for row in selected_rows}
            badges_html = []
            for s in display_sizes:
                if s in table_size_set:
                    row = next(r for r in selected_rows if r[size_key] == s)
                    badges_html.append(badge(f"{row['id']} {_to_wxh(s)}", tone="green"))
                else:
                    badges_html.append(badge(f"CUSTOM {_to_wxh(s)}", tone="gold"))
            st.markdown(" ".join(badges_html), unsafe_allow_html=True)
        else:
            st.caption("None ticked/entered yet -- Run clone will fall back to output.clone in the request's .json file.")

    source_stem = Path(select_image).stem if select_image else ""

    # UPDATE 2026-08-16 (bug #35): REQUEST.md [4] "mode=clone" -- nếu user
    # không tick size nào ở bảng trên, KHÔNG còn chặn "Run clone" nữa mà
    # fallback dùng output.clone của chính request_name.json (giống hệt CLI
    # `--mode=clone`, xem `release_manager.clone_from_release()`). Resolve
    # ngay ở đây (không chỉ ở backend) để "Expected clone files" + nút Run
    # hiển thị ĐÚNG size sẽ thực sự chạy trước khi user bấm.
    # UPDATE 2026-08-17: `display_sizes` now already includes BOTH the
    # table-ticked sizes AND any valid custom size(s) typed in the box above.
    if display_sizes:
        effective_sizes, sizes_source = display_sizes, "the sizes ticked/entered above"
    else:
        req_resp = api_client.get_request(request_name)
        effective_sizes = ((req_resp.get("request") or {}).get("output") or {}).get("clone", []) if req_resp.get("ok") else []
        sizes_source = f"output.clone in {request_name}.json (no size ticked above)"

    with card():
        st.markdown("#### 📂 Expected clone files")
        st.caption(f"Known in advance from the naming convention -- using {sizes_source}.")
        if not clone_model:
            st.info("Choose an image model above.")
        elif not effective_sizes:
            st.warning(
                "No display size ticked above, AND request_name.json has no output.clone sizes either -- "
                "tick at least 1 row above or add sizes to output.clone."
            )
        else:
            for size_str in effective_sizes:
                filename = f"{source_stem}_clone_{_to_wxh(size_str)}_{clone_model}.png"
                st.code(f"output/{request_name}/release/{filename}", language=None)

    run_clone_clicked = st.button(
        "Run clone",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.get("clone_run_in_progress", False),
    )

if run_clone_clicked:
    if not clone_model:
        st.error("Choose a model.")
    elif not effective_sizes:
        st.error(
            "No display size ticked above, AND request_name.json has no output.clone sizes either -- "
            "tick at least 1 row above or add sizes to output.clone."
        )
    else:
        run_started_dt = datetime.now()
        st.session_state["clone_run_started_ts"] = run_started_dt.timestamp()
        st.session_state["clone_run_in_progress"] = True
        st.session_state["clone_request_name"] = request_name
        st.session_state["clone_source_stem"] = source_stem
        st.session_state["clone_effective_sizes"] = effective_sizes
        st.session_state["last_clone_error"] = None
        st.session_state["last_clone_results"] = []

        job: dict = {"done": False, "results": None, "error": None}
        st.session_state["_clone_job"] = job
        threading.Thread(
            target=_run_clone_job,
            args=(job, request_name, select_image, effective_sizes, clone_model),
            daemon=True,
        ).start()

st.markdown("#### 🖼️ Clone output")

# UPDATE 2026-08-17 (bug #50, user report): "chỗ Choose a released image
# khi tôi chọn thì bên sidebar chỗ request bị load lại tên request ban
# đầu." Root cause (2nd contributor, on top of the `select_image` key fix
# above): while a clone job is running, this block used a TOP-LEVEL
# `time.sleep(2)` + `st.rerun()` poll loop -- the exact same anti-pattern
# already root-caused and fixed for the Generate page in bug #43 (see that
# page's own long comment above `_render_output_body`). Every ~2s the
# ENTIRE script re-executes, including the sidebar's `Request` selectbox --
# if the user interacts with anything on the page (incl. "Choose a
# released image") while a background clone job happens to be running, the
# continuous auto-rerun can race with that interaction and visually snap a
# widget back to a stale value. Fix: same `st.fragment(run_every=...)`
# pattern as Generate -- ONLY this Clone-output section auto-reruns while a
# clone job is active; the sidebar and everything else on the page is
# completely outside the fragment and can no longer be touched by the poll.
clone_is_active = (
    st.session_state.get("clone_request_name") == request_name
    and st.session_state.get("clone_run_in_progress", False)
)

_use_fragment = hasattr(st, "fragment")  # phòng trường hợp streamlit cũ chưa có API này


def _render_clone_output_body() -> None:
    # Tính lại NGAY TRONG fragment (không dùng closure biến ngoài) -- lý do
    # giống hệt `generate.py::_render_output_body`: fragment có thể tự chạy
    # lại nhiều lần độc lập với script top-level, cần đọc session_state MỚI
    # NHẤT mỗi lần.
    still_active = (
        st.session_state.get("clone_request_name") == request_name
        and st.session_state.get("clone_run_in_progress", False)
    )

    if still_active:
        job = st.session_state.get("_clone_job") or {}
        since_ts = st.session_state.get("clone_run_started_ts") or 0
        # Dùng size list ĐÃ CHỐT lúc bấm Run (`clone_effective_sizes`), không
        # phải `display_sizes` của lần render hiện tại -- 2 giá trị này có thể
        # khác nhau nếu user đổi lựa chọn trên bảng trong lúc job nền đang chạy.
        expected_total = len(st.session_state.get("clone_effective_sizes") or [])

        status_resp = api_client.get_clone_status(request_name, since_ts=since_ts)
        found_files = status_resp.get("files", []) if status_resp.get("ok") else []

        status_label = f"Cloning `{Path(select_image).name}` -- {len(found_files)}/{expected_total} file(s) written so far..."
        with st.status(status_label, expanded=True) as status:
            if found_files:
                _render_clone_files(found_files)
            else:
                st.caption("No clone files written yet.")

            if job.get("done"):
                st.session_state["clone_run_in_progress"] = False
                st.session_state["last_clone_results"] = job.get("results") or []
                st.session_state["last_clone_error"] = job.get("error")
                status.update(
                    label="Clone failed" if job.get("error") else "Clone complete",
                    state="error" if job.get("error") else "complete",
                )

        if job.get("done"):
            # Job vừa xong -- rerun TOÀN APP (scope mặc định) để sidebar +
            # fragment này được định nghĩa lại với run_every=None, dừng poll.
            st.rerun()
        elif not _use_fragment:
            # Fallback cho streamlit cũ không có st.fragment.
            time.sleep(2)
            st.rerun()
    else:
        same_source_session = st.session_state.get("clone_source_stem") == source_stem

        if same_source_session and st.session_state.get("last_clone_error"):
            st.error(f"Could not complete the clone: {st.session_state['last_clone_error']}")

        if same_source_session and st.session_state.get("last_clone_results"):
            failed = [r for r in st.session_state["last_clone_results"] if not r.get("ok")]
            for r in failed:
                st.warning(f"**{r.get('display_size')}** failed: {r.get('error')}")

        clone_status_resp = api_client.get_clone_status(request_name, source_stem=source_stem)
        clone_files = clone_status_resp.get("files", []) if clone_status_resp.get("ok") else []

        if clone_files:
            _render_clone_files(clone_files)
        else:
            st.info("No clone files yet for this image. Pick display sizes above and click 'Run clone'.")


if _use_fragment:
    st.fragment(_render_clone_output_body, run_every=2 if clone_is_active else None)()
else:
    _render_clone_output_body()
