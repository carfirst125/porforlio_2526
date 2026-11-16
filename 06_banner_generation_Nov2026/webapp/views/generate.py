"""Generate view -- loaded by the router (`webapp/Home.py`) via `st.navigation()`.

UPDATE 2026-08-15 (theo yêu cầu user): bỏ toàn bộ phần Clone (checkbox +
source path + model) khỏi sidebar trang này -- clone giờ chỉ còn 1 chỗ duy
nhất là trang Release (`webapp/views/release.py`), tránh trùng lặp chức
năng. Cũng rút gọn sidebar (bớt 1 divider + cả khối clone) để nút Run luôn
hiện mà không cần cuộn, và toàn bộ text UI dịch sang tiếng Anh.

UPDATE 2026-08-16 (theo yêu cầu user):
1. Khu vực nội dung bên phải (main content, khác với sidebar bên trái): bỏ
   logo OCB khỏi tiêu đề trang (`page_title_block(..., show_logo=False)`),
   đưa "Generate banner" lên đầu dòng, bỏ icon paint (🎨) khỏi title.
2. Sau khi chọn request_name.json và bấm Run: trước đây `POST /generate`
   chạy BLOCKING ngay trong request chính của Streamlit -- script đứng im,
   không hiện gì cho tới khi backend trả về HOÀN TOÀN (có thể tới 30 phút
   với mode=all). User muốn: (a) biết TRƯỚC những thư mục ảnh sẽ nằm ở đâu,
   suy ra được ngay từ quy ước đặt tên thư mục
   (`output/<request_name>/<display_size>/<image_model>/...`, xem
   `app/core/output_manager.py`) -- không cần chờ; (b) lưu lại mốc thời gian
   lúc bấm Run (định dạng `YYYY-mm-dd-HHMM`, khớp hậu tố tên file ảnh thật);
   (c) trong lúc chờ, LIÊN TỤC quét các thư mục output đó xem ảnh đã xuất
   hiện chưa, và hiển thị ngay khi có (không đợi hết cả batch).

   Cách làm: `POST /generate` giờ chạy trong 1 `threading.Thread` nền (daemon)
   thay vì chặn thẳng script chính. Script chính poll endpoint mới
   `GET /outputs/{request_name}?since_ts=...` (xem `app/api/routes.py` +
   `webapp/utils/api_client.py::list_outputs`) mỗi ~2 giây bằng vòng lặp
   `time.sleep()` + `st.rerun()` -- endpoint backend tự quét
   `output/<request_name>/` tìm ảnh .png có mtime >= thời điểm bấm Run (chính
   là mốc "gần thời gian lúc nhấn Run" user yêu cầu dùng để nhận diện đúng
   ảnh của phiên chạy này, vì mỗi request chỉ chạy 1 phiên tại 1 thời điểm).
   Khi thread nền báo `done`, dừng poll và chuyển qua hiển thị kết quả cuối
   cùng (bao gồm cả các model bị lỗi, đã có sẵn logic ở dưới).

UPDATE 2026-08-16 (2) (theo yêu cầu user): mục Output khi KHÔNG đang chạy
(is_active=False) trước đây chỉ đọc `st.session_state["last_results"]` --
mất trắng khi reload trang/mở session Streamlit mới, dù ảnh vẫn còn nguyên
trên đĩa. Giờ nguồn hiển thị chính là `GET /outputs/{request_name}/latest`
(`app/api/routes.py::latest_outputs` + `api_client.get_latest_outputs`) --
LUÔN quét trực tiếp `output/<request_name>/` và trả về đúng cụm ảnh của lượt
chạy GẦN NHẤT (gom cụm theo mtime, xem docstring `latest_outputs`), độc lập
với session. `last_results`/`last_generate_error` trong session (nếu có, cho
CHÍNH request đang chọn) vẫn được dùng để báo model nào lỗi -- model lỗi
không ghi file ra đĩa nên không thể lấy lại từ `latest_outputs`.

UPDATE 2026-08-16 (3) (theo yêu cầu user):
1. Mọi nơi hiển thị ảnh trong mục Output (đang chạy / lượt gần nhất / tra
   theo timestamp thủ công) giờ NHÓM theo cấu trúc:
   display_size (in đậm, vd "750x300") -> bên trong là image_model (in đậm,
   vd "gpt-image-2") -> ảnh của đúng model đó bên dưới. Xem
   `_render_grouped_images()`.
2. Thêm ô nhập text cho user gõ CHÍNH XÁC hậu tố timestamp của tên file ảnh
   (`YYYY-mm-dd-HHMM`, chấp nhận cả dạng có gạch giữa giờ/phút
   "YYYY-mm-dd-HH-MM" -- xem `_normalize_run_timestamp()`) để "gọi lại" đúng
   ảnh của 1 lượt chạy CỤ THỂ, ghi đè lên rule tự động (poll theo since_ts
   trong lúc chạy, hoặc gom cụm "lượt gần nhất" khi không chạy) -- dùng khi
   rule tự động chọn nhầm hoặc user muốn xem lại 1 lượt chạy cũ. Nguồn dữ
   liệu: `GET /outputs/{request_name}/at-timestamp?ts=...`
   (`app/api/routes.py::outputs_at_timestamp` +
   `api_client.get_outputs_at_timestamp`).

UPDATE 2026-08-16 (bug #40, theo user báo lỗi): ô nhập ở trên trước đây chỉ
khớp bằng `startswith` trên hậu tố timestamp đã parse của tên file -- nghĩa
là chuỗi gõ vào PHẢI đứng đầu phần timestamp, không dùng được để tìm theo
tên model/creative_id/size. User làm rõ: chỉ cần chuỗi đó CÓ MẶT ở bất kỳ
đâu trong tên file ảnh là đủ (không cần đứng đầu, không giới hạn ở phần
timestamp). `outputs_at_timestamp` giờ khớp bằng `in` (substring, không phân
biệt hoa/thường) trên TOÀN BỘ tên file -- vẫn khớp y hệt như cũ với timestamp
(đầy đủ hay 1 phần), giờ cũng khớp được với tên model, creative_id, display
size, hay bất kỳ phần nào khác của tên file."""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils import api_client  # noqa: E402
from webapp.utils.theme import badge, card, page_title_block  # noqa: E402

page_title_block(
    "Generate banner",
    "Choose a request, run mode, and generate the banner",
    show_logo=False,
)

# ---------------------------------------------------------------------------
# UPDATE 2026-09-14 (theo user báo lỗi #2): trước đây trang này KHÔNG check
# backend FastAPI còn sống trước khi làm gì khác -- nếu user chỉ chạy
# `streamlit run webapp/Home.py` mà QUÊN chạy `uvicorn app.main:app --reload`
# song song (xem README.md "Quickest path to running it" -- luôn cần CẢ 2
# tiến trình), mọi lệnh gọi `api_client.*` phía dưới đều fail với
# `requests.exceptions.ConnectionError` (bị `_get`/`_post` nuốt gọn thành
# `{"ok": False, "error": "...Connection refused..."}`), rồi lỗi đó lại bị
# hiểu lầm thành "request.json bị thiếu" (như lần báo lỗi trước) vì thông báo
# cuối cùng hiện ra là `Could not load request '<default>': GET ... failed:
# ...actively refused...` -- che mất nguyên nhân thật. `webapp/views/home.py`
# đã có sẵn health-check này (dòng "Backend FastAPI" ở trang Home) nhưng
# Generate/Release lại không có, nên user vẫn có thể mở thẳng Generate và gặp
# lỗi khó hiểu. Thêm check tương tự NGAY ĐẦU trang, dừng lại bằng 1 thông báo
# RÕ RÀNG duy nhất (kèm đúng lệnh cần chạy) trước khi kịp render bất cứ phần
# nào khác phụ thuộc API.
# ---------------------------------------------------------------------------
_health = api_client.health()
if not _health.get("ok"):
    st.error(
        f"⚠️ Could not connect to the backend API -- {_health.get('error')}\n\n"
        f"This page needs the FastAPI backend running alongside Streamlit. Start it first, then reload this page:\n\n"
        f"```\nuvicorn app.main:app --reload --reload-dir app\n```"
    )
    st.stop()


def _expected_display_size(req: dict) -> str:
    """Suy ra display_size dạng "WxH" (đúng convention thư mục output, xem
    `app/core/output_manager.display_size_slug`) -- để biết TRƯỚC ảnh sẽ nằm
    ở thư mục nào, không cần gọi API/chờ chạy xong.

    UPDATE 2026-08-20 (theo yêu cầu user -- REVERT lại UPDATE 2026-08-19
    round 3 bên dưới): request.json thật hoá ra đã đổi lại đúng theo
    template gốc trong REQUEST.md (nested `generation.canvas`/
    `generation.safe_area` + `output.size` là OBJECT `{width, height}` =
    kích thước output CUỐI CÙNG, không còn field `design_options.
    protect_area` dạng string nữa -- xem `app/models/schemas.py` module
    docstring và `4-project-evaluation-results.md` 2026-08-20). File
    `input/MKT_DEMO_TET.json` dùng làm căn cứ cho bản 2026-08-19 round 3 bên
    dưới hoá ra là 1 file cũ user quên xoá, không phải schema thật.
    `req` ở đây là dict thô parse thẳng từ request.json (không qua pydantic),
    nên tự đọc `output.size.width`/`.height` thủ công, không import ngược
    app.* vào webapp.

    (UPDATE 2026-08-19 round 3, ĐÃ REVERT: từng ưu tiên đọc
    `design_options.protect_area` dạng string "W,H" vì tưởng đó là kích
    thước PIXEL THẬT, còn `output.size` chỉ là canvas generate nội bộ --
    ngược lại hoàn toàn so với schema thật.)"""
    size = (req.get("output") or {}).get("size") or {}
    try:
        return f"{int(size.get('width'))}x{int(size.get('height'))}"
    except Exception:  # noqa: BLE001
        return "-"


def _normalize_run_timestamp(raw: str) -> str:
    """Chuẩn hoá timestamp user gõ tay về đúng dạng hậu tố tên file ảnh thật
    (`YYYY-mm-dd-HHMM`, xem `output_manager.TIMESTAMP_FORMAT`). User được
    yêu cầu gõ "yyyy-mm-dd-hh-mm" (có gạch giữa giờ/phút) -- 5 phần cách nhau
    bởi "-" -- trong khi tên file thật chỉ có 4 phần (giờ+phút dính liền,
    không gạch). Nếu phát hiện đúng 5 phần với 2 phần cuối ngắn (giờ/phút)
    thì ghép lại; ngược lại giữ nguyên (đã đúng dạng, hoặc user chỉ gõ 1
    phần rút gọn như "2026-08-16" -- vẫn dùng được làm tiền tố khớp cả
    ngày, xem `outputs_at_timestamp`)."""
    raw = raw.strip()
    if not raw:
        return ""
    parts = raw.split("-")
    if len(parts) == 5 and len(parts[3]) <= 2 and len(parts[4]) <= 2:
        yyyy, mm, dd, hh, mi = parts
        return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}-{hh.zfill(2)}{mi.zfill(2)}"
    return raw


_SEGMENT_SUFFIXES = ("_MASS_AFF", "_AFF", "_MASS")


def _split_segment(name: str) -> tuple[str, str] | None:
    """Tách 1 request_name thành (base, segment) nếu tên khớp đúng 1 trong 3
    hậu tố phân khúc DB (kiểm tra `_MASS_AFF` trước `_AFF`/`_MASS` để không bị
    khớp nhầm phần con của hậu tố dài hơn). Trả None nếu không khớp -- dùng
    cho batch-run (`_group_by_product`) lẫn hiển thị trạng thái batch."""
    for suffix in _SEGMENT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return None


def _group_by_product(req_list: list[str]) -> dict[str, dict[str, str]]:
    """Gom các request_name theo "sản phẩm" (phần trước hậu tố phân khúc) --
    dùng cho batch-run nhiều phân khúc MASS/MASS_AFF/AFF của 1 sản phẩm.
    Không hard-code danh sách sản phẩm (xem comment về `DEFAULT_REQUEST_NAME`
    đã lỗi thời nhiều lần) -- suy trực tiếp từ `req_list` thật đang có trong
    input/. Chỉ giữ lại sản phẩm có >= 2 phân khúc; sản phẩm chỉ có 1 file
    (hoặc không khớp hậu tố nào) vẫn chạy được như cũ qua ô chọn file đơn."""
    groups: dict[str, dict[str, str]] = {}
    for name in req_list:
        split = _split_segment(name)
        if split is None:
            continue
        base, segment = split
        groups.setdefault(base, {})[segment] = name
    return {base: segs for base, segs in groups.items() if len(segs) >= 2}


def _render_grouped_images(images: list[dict], key_prefix: str, allow_release: bool, request_name: str) -> None:
    """Vẽ danh sách ảnh (mỗi item có display_size/image_model/filename/path)
    thành từng block: 1 block cho mỗi display_size (tiêu đề in đậm) -> bên
    trong đó, 1 block con cho mỗi image_model (tên model in đậm) -> ảnh của
    đúng model đó xếp bên dưới -- theo yêu cầu user 2026-08-16 (3)."""
    if not images:
        return

    sizes_order: list[str] = []
    by_size: dict[str, list[dict]] = {}
    for img in images:
        size = img.get("display_size") or "-"
        if size not in by_size:
            by_size[size] = []
            sizes_order.append(size)
        by_size[size].append(img)

    to_release: list[str] = []
    for size in sizes_order:
        st.markdown(f"##### {size}")

        models_order: list[str] = []
        by_model: dict[str, list[dict]] = {}
        for img in by_size[size]:
            model = img.get("image_model") or "-"
            if model not in by_model:
                by_model[model] = []
                models_order.append(model)
            by_model[model].append(img)

        for model in models_order:
            st.markdown(f"**{model}**")
            model_images = by_model[model]
            cols = st.columns(3)
            for i, img in enumerate(model_images):
                with cols[i % 3]:
                    with card():
                        st.image(api_client.image_url(img["path"]), caption=img.get("filename", ""))
                        if allow_release:
                            checked = st.checkbox(
                                "Select for release",
                                key=f"release_{key_prefix}_{size}_{model}_{i}_{img['path']}",
                            )
                            if checked:
                                to_release.append(img["path"])

    if allow_release and to_release:
        if st.button(
            f"Release {len(to_release)} selected image(s)",
            type="primary",
            key=f"release_btn_{key_prefix}",
        ):
            rel_resp = api_client.release(request_name, to_release)
            released = rel_resp.get("released_paths", [])
            st.success(f"Released {len(released)} image(s) to output/{request_name}/release/. See the Release page.")


def _render_batch_segments(exclude_name: str | None = None) -> None:
    """UPDATE 2026-09-23 (2, theo user báo lỗi -- "chạy 3 phân khúc mà chỉ
    thấy 1 ảnh"): hiển thị ảnh của TỪNG phân khúc trong batch đang chạy/mới
    chạy xong, không chỉ phân khúc "đang active" (`request_name` hiện tại).
    Mỗi phân khúc ghi ảnh vào `output/<tên request riêng>/` (folder khác
    nhau vì mỗi phân khúc là 1 request_name riêng), nên gọi lại
    `list_outputs` cho TỪNG tên trong `batch_status`, lọc theo
    `batch_started_ts` (mốc bắt đầu CẢ BATCH, không bị ghi đè khi chuyển
    sang phân khúc kế tiếp, khác `run_started_ts` -- chỉ để loại các ảnh cũ
    có sẵn trước khi bấm Run, không liên quan gì tới batch này)."""
    batch_status = st.session_state.get("batch_status")
    if not batch_status:
        return
    since_ts = st.session_state.get("batch_started_ts") or 0
    batch_errors = st.session_state.get("batch_errors") or {}
    for name, stat in batch_status.items():
        if name == exclude_name:
            continue
        split = _split_segment(name)
        label = split[1] if split else name
        resp = api_client.list_outputs(name, since_ts)
        images = resp.get("images", []) if resp.get("ok") else []
        icon = "✅" if stat == "done" else "❌" if stat == "error" else "⏹️"
        st.markdown(f"##### {label} {icon}")
        if stat == "error" and name in batch_errors:
            st.warning(f"Could not complete the run: {batch_errors[name]}")
        if images:
            _render_grouped_images(images, key_prefix=f"batch_{name}", allow_release=True, request_name=name)
        elif stat != "error":
            st.caption("Chưa chạy." if stat == "pending" else "No image files found for this segment.")


def _run_generate_job(
    job: dict,
    request_name: str,
    mode: str,
    custom_models: list[str],
    num_version_override: int | None = None,
    generate_person_override: bool | None = None,
    persona_override: str | None = None,
) -> None:
    """Chạy trong thread nền (daemon) -- `POST /generate` có thể block tới 30
    phút (mode=all, nhiều model/version). Ghi kết quả vào `job` (dict lưu
    trong `st.session_state`) để luồng chính (script Streamlit, tự rerun mỗi
    ~2s khi đang poll) đọc được mà không phải tự chờ thread này chạy xong."""
    resp = api_client.generate(
        request_name,
        mode,
        custom_models,
        num_version_override=num_version_override,
        generate_person_override=generate_person_override,
        persona_override=persona_override,
    )
    if "results" in resp:
        job["results"] = resp.get("results", [])
        job["error"] = None
    else:
        job["results"] = []
        job["error"] = resp.get("error", "Unknown error calling /generate")
    job["done"] = True


_NONE_PERSONA = "__none__"


def _persona_selectbox(key: str, request_name: str, default_persona: str | None, help_text: str) -> str:
    """Optional 1-of-8 RB card-spend persona. Empty string = no persona this run."""
    resp = api_client.list_personas(request_name)
    catalog = resp.get("personas") or []
    options = [_NONE_PERSONA]
    labels = {_NONE_PERSONA: "Không dùng (mặc định)"}
    suggested_ids: set[str] = set()
    for item in catalog:
        pid = item.get("id")
        if not pid:
            continue
        options.append(pid)
        mark = " · gợi ý" if item.get("suggested") else ""
        labels[pid] = f"{item.get('name_vi', pid)} — {item.get('name_en', pid)}{mark}"
        if item.get("suggested"):
            suggested_ids.add(pid)
    default = default_persona if default_persona in options else _NONE_PERSONA
    picked = st.selectbox(
        "Chân dung chi tiêu thẻ",
        options,
        index=options.index(default),
        format_func=lambda value: labels.get(value, value),
        key=key,
        help=help_text,
    )
    if picked != _NONE_PERSONA and catalog and picked not in suggested_ids:
        st.caption("Chân dung này không nằm trong gợi ý cho sản phẩm hiện tại — vẫn chạy được.")
    return "" if picked == _NONE_PERSONA else str(picked)


def _persona_display_name(persona_id: str, request_name: str) -> str:
    if not persona_id:
        return "Không dùng"
    resp = api_client.list_personas(request_name)
    for item in resp.get("personas") or []:
        if item.get("id") == persona_id:
            return str(item.get("name_vi") or persona_id)
    return persona_id


# ---------------------------------------------------------------------------
# [8a] Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Run configuration")

    # UPDATE 2026-09-23 (theo yêu cầu user): "chạy 1 sản phẩm (3 usecase là 3
    # phân khúc) không thay vì bấm từng cái" -- toggle này cho chạy TUẦN TỰ
    # nhiều phân khúc (MASS/MASS_AFF/AFF) của cùng 1 sản phẩm với 1 lần bấm
    # Run, dùng chung num_version/generate_person cho cả batch. Xem override
    # `request_name`/`num_version`/`generate_person` ngay sau `with
    # st.sidebar:` này, và phần tự động chuyển sang phân khúc kế tiếp trong
    # `_render_output_body()`.
    batch_mode = st.toggle(
        "🗂️ Batch: chạy nhiều phân khúc của 1 sản phẩm",
        value=False,
        key="batch_mode",
        help=(
            "Chọn 1 sản phẩm (vd AUTOBILL), tick các phân khúc muốn chạy "
            "(MASS/MASS_AFF/AFF) -- chạy tuần tự, dùng chung Versions per "
            "model + Có người trong ảnh + chân dung chi tiêu thẻ cho tất cả phân khúc đã chọn."
        ),
    )

    # UPDATE 2026-08-18 (theo yêu cầu user): mặc định request.json khi CHƯA
    # chọn gì (lần đầu mở trang / chưa tương tác với widget) là 1 tên cố định
    # thay vì luôn rơi vào item đầu tiên theo alphabet của `req_list`.
    #
    # UPDATE 2026-08-20 rồi UPDATE 2026-09-14 (theo user báo lỗi lần 1): tên
    # cố định này ("MKT_DEMO_TET" -> "DB_NOTI_AUTOPAY_NOEL") lần lượt bị xoá
    # khỏi input/ theo thời gian -- input/ giờ có ~70 request thật khác (vd
    # các file DB_*_1024x440_*.json / MKT_SLIDE*.json / NATIVE_FLEXI*.json).
    # Đổi sang 1 file ĐANG THẬT CÓ trong input/ hiện tại; nhưng đây vẫn chỉ là
    # 1 "gợi ý" best-effort (có thể lại bị xoá trong tương lai) -- xem widget
    # bên dưới để hiểu vì sao không còn crash được nữa dù hằng số này lại trở
    # nên lỗi thời lần nữa.
    #
    # UPDATE 2026-09-14 (theo user báo lỗi lần 2 -- "vẫn lỗi", "thêm dropbox
    # hỗ trợ nhập text + shortlist"): lỗi lần 2 hoá ra KHÔNG liên quan gì đến
    # hằng số trên nữa -- nguyên nhân thật là backend FastAPI chưa chạy (xem
    # health-check mới thêm ở đầu file, ngay sau `page_title_block`), lỗi đó
    # đã được chặn RIÊNG ở trên rồi nên tới đây chắc chắn API đang sống. Nhân
    # lúc sửa, đổi luôn 2 nhánh `st.selectbox` (khi có `req_list`) /
    # `st.text_input` (fallback khi không có) thành 1 WIDGET DUY NHẤT dùng
    # `st.selectbox(..., accept_new_options=True)`.
    #
    # UPDATE 2026-09-14 (theo user báo lỗi lần 3 -- "TypeError: ... unexpected
    # keyword argument 'accept_new_options'"): kwarg này chỉ có ở 1 bản
    # Streamlit khá mới (đã bump `requirements.txt>=1.42`), nhưng môi trường
    # thật của user (`D:\python_env\.venv_image`) đang chạy 1 bản CŨ HƠN chưa
    # có kwarg này -- và không chắc user sẽ (hoặc có thể) `pip install
    # --upgrade streamlit` ngay. Để KHÔNG còn phụ thuộc vào việc user phải
    # nâng cấp đúng phiên bản Streamlit, bỏ hẳn `accept_new_options` và tự
    # code lại hành vi tương đương bằng 2 widget CƠ BẢN (chạy được trên mọi
    # bản Streamlit tương đối gần đây, giống cách làm trước UPDATE lần 2):
    #   1. `st.selectbox` bình thường trên `req_list` + 1 option đặc biệt
    #      "✏️ Nhập tên khác..." được thêm vào CUỐI danh sách. Việc gõ vào ô
    #      để LỌC/rút ngắn danh sách xuống các mục khớp vẫn là hành vi có sẵn
    #      của `st.selectbox` -- không mất tính năng "shortlist" user đã yêu
    #      cầu ở lần 2.
    #   2. Nếu user chọn đúng option đặc biệt đó (hoặc `req_list` rỗng, tự
    #      động coi như đã "chọn" nhánh nhập tay), hiện thêm 1 `st.text_input`
    #      ngay dưới để gõ tên request_name mới hoàn toàn -- giá trị cuối
    #      cùng dùng cho `request_name` lấy từ ô này.
    _NEW_NAME_OPTION = "✏️ Nhập tên khác (chưa có trong danh sách)..."
    DEFAULT_REQUEST_NAME = "DB_AUTOBILL_1024x440_MASS"

    req_list = api_client.list_requests().get("requests", [])
    if req_list:
        select_options = [*req_list, _NEW_NAME_OPTION]
        default_index = req_list.index(DEFAULT_REQUEST_NAME) if DEFAULT_REQUEST_NAME in req_list else 0
        picked = st.selectbox(
            "Input file request_name.json",
            select_options,
            index=default_index,
            help=(
                "Click to browse, type to filter the list, or pick "
                f"'{_NEW_NAME_OPTION}' to type a brand-new request_name."
            ),
        )
    else:
        st.caption("No files found in input/ via the API -- type a name manually below.")
        picked = _NEW_NAME_OPTION

    if picked == _NEW_NAME_OPTION:
        request_name = st.text_input(
            "New request_name (not in the list above yet)",
            placeholder="e.g. DB_AUTOBILL_1024x440_MASS",
        )
    else:
        request_name = picked

    batch_request_names: list[str] = []
    batch_num_version_widget = 1
    batch_generate_person_widget = False
    batch_persona_widget = ""
    if batch_mode:
        st.divider()
        groups = _group_by_product(req_list)
        if not groups:
            st.info("Không tìm thấy sản phẩm nào có nhiều phân khúc (MASS/MASS_AFF/AFF).")
        else:
            picked_product = st.selectbox("Sản phẩm", sorted(groups), key="batch_product")
            product_segments = groups[picked_product]
            selected_segments: list[str] = []
            for seg in ("MASS", "MASS_AFF", "AFF"):
                if seg in product_segments:
                    checked = st.checkbox(seg, value=True, key=f"batch_seg_{picked_product}_{seg}")
                    if checked:
                        selected_segments.append(seg)
            batch_request_names = [product_segments[seg] for seg in selected_segments]

            if not batch_request_names:
                st.warning("Chọn ít nhất 1 phân khúc để chạy batch.")
            else:
                _first_req_resp = api_client.get_request(batch_request_names[0])
                _first_req = _first_req_resp.get("request", {}) if _first_req_resp.get("ok") else {}
                batch_num_version_widget = st.number_input(
                    "Versions per model (batch)",
                    min_value=1,
                    max_value=10,
                    value=int(_first_req.get("num_version", 1)),
                    step=1,
                    key="batch_num_version_widget",
                    help="Dùng chung cho tất cả phân khúc đã tick trong batch này.",
                )
                batch_generate_person_widget = st.checkbox(
                    "Có người trong ảnh (batch)",
                    value=bool(_first_req.get("design_options", {}).get("generate_person", False)),
                    key="batch_generate_person_widget",
                    help="Dùng chung cho tất cả phân khúc đã tick trong batch này.",
                )
                batch_persona_widget = _persona_selectbox(
                    "batch_persona_widget",
                    batch_request_names[0],
                    (_first_req.get("audience") or {}).get("persona"),
                    "1 trong 8 chân dung chi tiêu thẻ (RB 3/2026). Dùng chung cho cả batch. "
                    "Không tạo format_type mới — MASS/AFF vẫn giữ layout. Để trống = hành vi cũ.",
                )
        st.divider()

    models_info = api_client.list_models()
    available_models = models_info.get("available_providers", [])
    model_usage = models_info.get("model_usage", {})
    model_usage_error = models_info.get("model_usage_error")

    mode = st.selectbox("Run mode", ["default", "quality", "eco", "custom", "all"])

    custom_models: list[str] = []
    if mode == "custom":
        # UPDATE 2026-08-16 (theo báo lỗi user): "chọn select box, tôi vừa
        # chọn là streamlit lại chạy, chưa chọn hết đã cập nhật" -- đúng hành
        # vi mặc định của Streamlit là RERUN TOÀN SCRIPT sau MỖI lần click
        # vào 1 option của `st.multiselect`, nên khi cần chọn NHIỀU model
        # cùng lúc, mỗi cú click lại kích hoạt 1 lần rerun + cập nhật ngay
        # "Models to run"/"Expected output folders" phía dưới trước khi user
        # chọn xong hết. Khắc phục CHUẨN theo khuyến nghị của Streamlit:
        # `st.form` -- các widget BÊN TRONG 1 form KHÔNG kích hoạt rerun khi
        # tương tác, giá trị chỉ được "chốt" (và script mới chạy lại đúng 1
        # lần) khi bấm nút `st.form_submit_button` của chính form đó. Giá trị
        # đã áp dụng được lưu vào `st.session_state["custom_models_selected"]`
        # để giữ nguyên qua các lần rerun khác (đổi Run mode, bấm Run, v.v. --
        # cùng nguyên tắc "không tự ý đổi giá trị" đã áp dụng ở bug #43).
        with st.form("custom_models_form", border=False):
            _picked = st.multiselect(
                "Choose models (mode=custom)",
                available_models,
                default=st.session_state.get("custom_models_selected", []),
            )
            _apply_clicked = st.form_submit_button("✅ Apply model selection")
        if _apply_clicked:
            st.session_state["custom_models_selected"] = _picked
        custom_models = st.session_state.get("custom_models_selected", [])
        if not custom_models:
            st.caption("Select model(s) above, then click 'Apply model selection'.")

    st.divider()
    run_clicked = st.button(
        "Run",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.get("run_in_progress", False) or (batch_mode and not batch_request_names),
    )

# UPDATE 2026-09-23: batch-run override -- khi 1 batch đang chạy (hoặc mới
# bật toggle, chưa bấm Run), `request_name` "đang active" của toàn trang bị
# đè lại thành phân khúc tương ứng trong `batch_queue`/`batch_request_names`.
# Đè SỚM ở đây (trước mọi phần dùng `request_name` bên dưới: design brief,
# "Models to run & Output", expected output folders, `is_active`) nên toàn
# bộ thân trang không cần biết gì về batch -- vẫn chỉ thấy "1 request_name"
# như bình thường, chỉ là giá trị này tự đổi qua từng phân khúc.
if st.session_state.get("batch_queue"):
    _batch_idx = min(st.session_state.get("batch_idx", 0), len(st.session_state["batch_queue"]) - 1)
    request_name = st.session_state["batch_queue"][_batch_idx]
elif batch_mode and batch_request_names:
    request_name = batch_request_names[0]

if not request_name:
    st.stop()

if model_usage_error:
    # UPDATE 2026-08-16 (bug #42): trước đây model_usage.json hỏng (vd JSON
    # không hợp lệ do sửa tay) chỉ khiến MỌI mode âm thầm hiện "Could not
    # determine any models" -- không rõ vì sao. Giờ backend trả kèm lỗi parse
    # cụ thể (xem `app/api/routes.py::models`), hiện NỔI BẬT ở đây để không
    # bị nhầm là "chỉ mode X mới bị lỗi" hay bug ở phần khác của trang.
    st.error(
        f"⚠️ Could not load `config/model_usage.json` -- every run mode below will show 0 models until this is "
        f"fixed:\n\n```\n{model_usage_error}\n```\n\nCommon cause: JSON does not support `//` comments -- check "
        f"for any stray comment lines or trailing commas in the file."
    )

# ---------------------------------------------------------------------------
# Load request detail
# ---------------------------------------------------------------------------
req_resp = api_client.get_request(request_name)
if not req_resp.get("ok"):
    st.error(f"Could not load request '{request_name}': {req_resp.get('error')}")
    st.stop()

req = req_resp["request"]
expected_display_size = _expected_display_size(req)

# ---------------------------------------------------------------------------
# [8b] Main screen: design brief, bucket asset gallery, text, model list, output size
# ---------------------------------------------------------------------------
with card():
    st.markdown("#### 🗒️ Design brief")
    design_brief = (req.get("design_brief") or "").strip()
    if design_brief:
        st.write(design_brief)
    else:
        st.caption("(this request.json has no design_brief)")

with card():
    st.markdown("#### 📁 Asset buckets")
    st.caption("Attached assets and reference")

    ASSET_LABELS = {
        "theme": "🖼️ Theme (Background)",
        "context": "🧑 Context (Character)",
        "product": "📦 Product",
        "reference": "🎯 Reference (Style guide)",
    }

    resolved_assets = req_resp.get("resolved_assets", [])
    if not resolved_assets:
        st.caption("(this request.json does not specify any assets)")
    else:
        # UPDATE 2026-08-17 (theo yêu cầu user): 1 bucket (theme/context/
        # product) giờ có thể có NHIỀU ảnh -- mỗi ảnh là 1 entry riêng trong
        # `resolved_assets`, cùng chung `label` gốc nhưng khác `index`/
        # `count` (xem `app/api/routes.py::get_request`). Khi `count` > 1,
        # thêm hậu tố " #i/n" vào title để phân biệt các ảnh trong cùng 1
        # bucket -- khi `count` == 1 (trường hợp cũ, phổ biến nhất) title
        # giữ nguyên y hệt như trước.
        asset_cols = st.columns(len(resolved_assets))
        for col, asset in zip(asset_cols, resolved_assets):
            with col:
                base_title = ASSET_LABELS.get(asset["label"], asset["label"])
                count = asset.get("count", 1)
                title = base_title if count <= 1 else f"{base_title} #{asset.get('index', 1)}/{count}"
                st.markdown(f"**{title}**")
                if asset.get("kind") == "text":
                    # 2026-09-04 (user request): this bucket entry is a
                    # free-text description instead of an attached image --
                    # no file was resolved (`path` is empty), just show the
                    # description itself instead of attempting st.image().
                    st.info(f"📝 {asset['value']}")
                else:
                    st.caption(f"`{asset['value']}`")
                    if asset.get("ok"):
                        st.image(api_client.image_url(asset["path"]), width=180)
                    else:
                        st.error(f"File not found -- {asset.get('error')}")

with card():
    st.markdown("#### 📝 Text content")
    text = req.get("text", {})
    tcols = st.columns(3)
    tcols[0].metric("Headline", text.get("headline") or "-")
    tcols[1].metric("Message", text.get("message") or "-")
    tcols[2].metric("CTA", text.get("cta_label") or "-")

with card():
    st.markdown("#### 🧠 Models to run & Output")
    if mode == "custom":
        resolved_models = custom_models
    elif mode == "all":
        resolved_models = []
        for grp in model_usage.get("all", []):
            for m in model_usage.get(grp, []):
                if m not in resolved_models:
                    resolved_models.append(m)
    else:
        resolved_models = model_usage.get(mode, [])

    if resolved_models:
        st.markdown(" ".join(badge(m, tone="green") for m in resolved_models), unsafe_allow_html=True)
    else:
        st.warning("Could not determine any models -- check `model_usage.json`.")

    mcols = st.columns(3)
    mcols[0].metric("Output display size", expected_display_size)
    if batch_mode:
        # UPDATE 2026-09-23: đang ở batch mode -- 2 widget num_version/
        # generate_person đã hiện ở sidebar (dùng chung cho cả batch), ở đây
        # chỉ hiện lại giá trị (không phải widget mới) để tránh trùng lặp
        # control + tránh 2 key Streamlit khác nhau vô tình lệch giá trị.
        # Khi 1 batch ĐANG CHẠY, dùng đúng giá trị đã "chốt" lúc bấm Run
        # (`batch_num_version`/`batch_generate_person`) -- không đọc lại theo
        # widget sidebar (user có thể đã đổi số cho lượt batch KẾ TIẾP trong
        # lúc batch hiện tại vẫn còn đang chạy phân khúc trước).
        if st.session_state.get("batch_queue"):
            num_version = st.session_state.get("batch_num_version", int(batch_num_version_widget))
            generate_person = st.session_state.get("batch_generate_person", bool(batch_generate_person_widget))
            persona = st.session_state.get("batch_persona", batch_persona_widget)
        else:
            num_version = int(batch_num_version_widget)
            generate_person = bool(batch_generate_person_widget)
            persona = batch_persona_widget
        mcols[1].metric("Versions per model", num_version)
        mcols[2].metric("Có người trong ảnh", "Có" if generate_person else "Không")
        st.caption(f"Chân dung chi tiêu thẻ: {_persona_display_name(str(persona or ''), request_name)}")
    else:
        num_version = mcols[1].number_input(
            "Versions per model",
            min_value=1,
            max_value=10,
            value=int(req.get("num_version", 1)),
            step=1,
            key=f"num_version_{request_name}",
            help="Ghi đè số lượng ảnh sinh ra mỗi model cho lượt Run này -- không sửa vào file input JSON.",
        )
        generate_person = mcols[2].checkbox(
            "Có người trong ảnh",
            value=bool(req.get("design_options", {}).get("generate_person", False)),
            key=f"generate_person_{request_name}",
            help="Ghi đè design_options.generate_person cho lượt Run này -- không sửa vào file input JSON.",
        )
        persona = _persona_selectbox(
            f"persona_{request_name}",
            request_name,
            (req.get("audience") or {}).get("persona"),
            "1 trong 8 chân dung chi tiêu thẻ (RB 3/2026). Ghi đè audience.persona cho lượt này -- không sửa JSON. "
            "MASS/AFF vẫn giữ layout. Không dùng = hành vi cũ.",
        )

with card():
    st.markdown("#### 📂 Expected output folders")
    st.caption(
        "Known in advance from the output naming convention -- images will be written here as each model finishes."
    )
    if resolved_models:
        for m in resolved_models:
            st.code(f"output/{request_name}/{expected_display_size}/{m}/", language=None)
    else:
        st.caption("(no models resolved yet)")

st.divider()

# ---------------------------------------------------------------------------
# [8c] Run in a background thread + poll output folders while it runs
# ---------------------------------------------------------------------------
if run_clicked and not st.session_state.get("run_in_progress", False):
    # UPDATE 2026-09-23: nếu batch_mode đang bật (và có >= 1 phân khúc được
    # tick), Run khởi động cả 1 HÀNG ĐỢI phân khúc thay vì 1 request đơn --
    # chỉ phân khúc ĐẦU TIÊN được chạy ngay ở đây; các phân khúc kế tiếp được
    # tự động khởi động trong nhánh "done" của `_render_output_body()` bên
    # dưới, mỗi khi 1 phân khúc chạy xong (xem `batch_queue`/`batch_idx`).
    if batch_mode and batch_request_names:
        st.session_state["batch_queue"] = list(batch_request_names)
        st.session_state["batch_idx"] = 0
        st.session_state["batch_num_version"] = int(num_version)
        st.session_state["batch_generate_person"] = bool(generate_person)
        st.session_state["batch_persona"] = str(persona or "")
        st.session_state["batch_status"] = {n: "pending" for n in batch_request_names}
        st.session_state["batch_errors"] = {}
        # UPDATE 2026-09-23 (2, theo user báo lỗi: "chạy 3 phân khúc mà chỉ
        # thấy 1 ảnh"): mốc thời gian CỦA CẢ BATCH (không đổi khi chuyển
        # sang phân khúc kế tiếp, khác `run_started_ts` vốn bị ghi đè mỗi
        # phân khúc) -- dùng để lọc đúng ảnh sinh ra TRONG batch này khi
        # gom hiển thị cả 3 phân khúc cùng lúc, xem `_render_batch_segments`
        # trong `_render_output_body()`.
        st.session_state["batch_started_ts"] = datetime.now().timestamp()
        request_name = batch_request_names[0]
    else:
        st.session_state["batch_queue"] = None
        st.session_state["batch_status"] = None
        st.session_state["batch_errors"] = None
        st.session_state["batch_started_ts"] = None

    run_started_dt = datetime.now()
    st.session_state["run_started_ts"] = run_started_dt.timestamp()
    st.session_state["run_timestamp_str"] = run_started_dt.strftime("%Y-%m-%d-%H%M")
    st.session_state["last_request_name"] = request_name
    st.session_state["run_in_progress"] = True
    st.session_state["last_generate_error"] = None
    st.session_state["last_results"] = []

    job: dict = {"done": False, "results": None, "error": None}
    st.session_state["_generate_job"] = job
    threading.Thread(
        target=_run_generate_job,
        args=(job, request_name, mode, custom_models, int(num_version), bool(generate_person), str(persona or "")),
        daemon=True,
    ).start()

_batch_status = st.session_state.get("batch_status")
if _batch_status:
    _done_icons = {"done": "✅", "error": "❌"}
    _parts = []
    for _name, _stat in _batch_status.items():
        _split = _split_segment(_name)
        _label = _split[1] if _split else _name
        if _stat in _done_icons:
            _icon = _done_icons[_stat]
        elif _name == request_name and st.session_state.get("run_in_progress"):
            _icon = "⏳"
        else:
            _icon = "⏹️"
        _parts.append(f"{_label} {_icon}")
    st.caption("Batch: " + "   ".join(_parts))

st.markdown("#### 🖼️ Output")

# ---------------------------------------------------------------------------
# UPDATE 2026-08-16 (bug #43, theo user báo lỗi): "Input file"/"Run mode" ở
# sidebar bị NHẢY về giá trị khác mỗi khi bấm nút khác (vd Run). Root cause:
# trước đây vòng lặp poll dùng `time.sleep(2)` + `st.rerun()` ở CẤP TOP-LEVEL
# của script -- mỗi lần poll là 1 lần CHẠY LẠI TOÀN BỘ script từ đầu, kể cả
# phần sidebar. Đây là anti-pattern đã biết của Streamlit: rerun toàn trang
# liên tục mỗi ~2s có thể "đụng độ" (race) với tương tác của user trên các
# widget khác (vd đổi selectbox) đang diễn ra CÙNG LÚC -- 1 lần rerun tự động
# có thể chạy dựa trên state CŨ, ghi đè lên lựa chọn user vừa mới đổi, tạo
# cảm giác "nhảy giá trị". Khắc phục CHUẨN theo khuyến nghị của Streamlit:
# dùng `st.fragment(run_every=...)` để CHỈ phần Output tự động chạy lại định
# kỳ, sidebar (Input file/Run mode/nút Run) nằm NGOÀI fragment nên không bao
# giờ bị đụng vào bởi vòng poll tự động nữa.
# `run_every` được set NGAY TRƯỚC khi định nghĩa fragment, dựa trên
# `is_active` tính từ session_state tại thời điểm script TOP-LEVEL chạy lại
# gần nhất (mỗi lần user tương tác với sidebar) -- khi job chạy xong, bên
# trong fragment gọi `st.rerun()` (mặc định rerun TOÀN APP, không phải
# `scope="fragment"`) đúng 1 lần để buộc script top-level chạy lại, từ đó
# `is_active` được tính lại thành False và fragment được định nghĩa lại với
# `run_every=None` -- dừng hẳn vòng poll, không tốn tài nguyên chạy nền khi
# đã xong.
# ---------------------------------------------------------------------------
is_active = (
    st.session_state.get("last_request_name") == request_name
    and st.session_state.get("run_in_progress", False)
)

_use_fragment = hasattr(st, "fragment")  # phòng trường hợp streamlit cũ chưa có API này


def _render_output_body() -> None:
    manual_ts_raw = st.text_input(
        "Look up a specific run instead (optional)",
        value=st.session_state.get("manual_run_ts", ""),
        placeholder="e.g. 2026-08-16-0000, gpt-image-2, or any part of the filename",
        help=(
            "Enter any text to match against image filenames -- a timestamp (yyyy-mm-dd-hh-mm), part of a "
            "model name, the creative_id, a display size, or any other substring. It just needs to appear "
            "ANYWHERE in the filename, not only at the start. Pulls every matching image, overriding the "
            "automatic 'in progress' / 'most recent run' selection below."
        ),
    )
    st.session_state["manual_run_ts"] = manual_ts_raw
    manual_ts = _normalize_run_timestamp(manual_ts_raw)

    # Tính lại `is_active` NGAY TRONG fragment (không dùng closure của biến
    # ngoài) -- vì fragment có thể tự chạy lại nhiều lần (mỗi ~2s) ĐỘC LẬP
    # với script top-level, cần đọc session_state MỚI NHẤT mỗi lần, không
    # phải giá trị đã "đóng băng" từ lần script top-level chạy lần trước.
    still_active = (
        st.session_state.get("last_request_name") == request_name
        and st.session_state.get("run_in_progress", False)
    )

    if manual_ts:
        # User muốn xem lại 1 lượt chạy CỤ THỂ -- ưu tiên tuyệt đối, bỏ qua cả
        # live-poll (still_active) lẫn rule "lượt gần nhất" tự động bên dưới.
        at_resp = api_client.get_outputs_at_timestamp(request_name, manual_ts)
        at_images = at_resp.get("images", []) if at_resp.get("ok") else []
        if at_images:
            st.caption(f"Showing images with filenames containing \"{manual_ts}\"")
            _render_grouped_images(at_images, key_prefix="manual", allow_release=True, request_name=request_name)
        else:
            st.warning(f"No images found with a filename containing \"{manual_ts}\".")
    elif still_active:
        if batch_mode:
            # Các phân khúc ĐÃ CHẠY XONG trong batch này -- hiện luôn ở trên,
            # trước phân khúc đang chạy (bên dưới) -- xem docstring
            # `_render_batch_segments`.
            _render_batch_segments(exclude_name=request_name)

        job = st.session_state.get("_generate_job") or {}
        run_ts_str = st.session_state.get("run_timestamp_str", "-")
        since_ts = st.session_state.get("run_started_ts") or 0
        expected_total = len(resolved_models) * num_version

        out_resp = api_client.list_outputs(request_name, since_ts)
        found_images = out_resp.get("images", []) if out_resp.get("ok") else []

        status_label = f"Running (started {run_ts_str}) -- {len(found_images)}/{expected_total} image(s) written so far..."
        with st.status(status_label, expanded=True) as status:
            if found_images:
                _render_grouped_images(found_images, key_prefix="live", allow_release=False, request_name=request_name)
            else:
                st.caption("No image files written yet -- waiting for the first model to finish.")

            if job.get("done"):
                st.session_state["run_in_progress"] = False
                st.session_state["last_results"] = job.get("results") or []
                st.session_state["last_generate_error"] = job.get("error")
                status.update(
                    label="Generate failed" if job.get("error") else "Generate complete",
                    state="error" if job.get("error") else "complete",
                )

                # UPDATE 2026-09-23: batch-run -- phân khúc này vừa xong, tự
                # động khởi động phân khúc KẾ TIẾP trong `batch_queue` (nếu
                # còn) trước khi rerun ở dưới. Rerun đó sẽ khiến override
                # request_name ở đầu trang đọc `batch_idx` MỚI, tự chuyển
                # sang phân khúc kế tiếp -- không cần đụng gì thêm vào cơ chế
                # fragment/poll hiện có.
                batch_queue = st.session_state.get("batch_queue")
                if batch_queue:
                    st.session_state.setdefault("batch_status", {})[request_name] = (
                        "error" if job.get("error") else "done"
                    )
                    if job.get("error"):
                        st.session_state.setdefault("batch_errors", {})[request_name] = job.get("error")
                    next_idx = st.session_state.get("batch_idx", 0) + 1
                    if next_idx < len(batch_queue):
                        st.session_state["batch_idx"] = next_idx
                        next_name = batch_queue[next_idx]
                        next_started = datetime.now()
                        next_job: dict = {"done": False, "results": None, "error": None}
                        st.session_state["_generate_job"] = next_job
                        st.session_state["run_started_ts"] = next_started.timestamp()
                        st.session_state["run_timestamp_str"] = next_started.strftime("%Y-%m-%d-%H%M")
                        st.session_state["last_request_name"] = next_name
                        st.session_state["run_in_progress"] = True
                        st.session_state["last_generate_error"] = None
                        st.session_state["last_results"] = []
                        threading.Thread(
                            target=_run_generate_job,
                            args=(
                                next_job,
                                next_name,
                                mode,
                                custom_models,
                                st.session_state["batch_num_version"],
                                st.session_state["batch_generate_person"],
                                st.session_state.get("batch_persona", ""),
                            ),
                            daemon=True,
                        ).start()
                    else:
                        st.session_state["batch_queue"] = None  # cả batch đã xong

        if job.get("done"):
            # Job vừa xong -- rerun TOÀN APP (scope mặc định) để sidebar +
            # fragment này được định nghĩa lại với run_every=None, dừng poll.
            st.rerun()
        elif not _use_fragment:
            # Fallback cho streamlit cũ không có st.fragment -- quay lại cách
            # cũ (kém hơn: rerun toàn script mỗi 2s, có thể vẫn gặp hiện
            # tượng "nhảy giá trị" ở sidebar khi tương tác đúng lúc poll).
            time.sleep(2)
            st.rerun()
    elif batch_mode and st.session_state.get("batch_status"):
        # Batch đã chạy xong (hoặc không có phân khúc nào đang active nữa) --
        # hiện đủ ảnh của TẤT CẢ phân khúc trong batch, không chỉ 1 phân khúc
        # (`request_name` lúc này chỉ còn là 1 giá trị "đại diện" do override
        # ở đầu trang, không còn ý nghĩa để chọn lượt chạy gần nhất riêng lẻ).
        _render_batch_segments()
    else:
        # Không có override thủ công, không đang chạy -- hiện lượt chạy GẦN
        # NHẤT đang có trên đĩa, luôn đúng kể cả sau khi reload trang / session
        # mới (xem `GET /outputs/{request_name}/latest`, gom cụm theo mtime).
        # Vẫn giữ báo lỗi/model lỗi từ `last_results` của session này (nếu
        # có) vì model bị lỗi KHÔNG ghi file ra đĩa nên không thể lấy lại từ đó.
        same_request_session = st.session_state.get("last_request_name") == request_name

        if same_request_session and st.session_state.get("last_generate_error"):
            st.error(f"Could not complete the run: {st.session_state['last_generate_error']}")

        if same_request_session and st.session_state.get("last_results"):
            failed = [r for r in st.session_state["last_results"] if not r.get("ok")]
            for r in failed:
                st.warning(f"**{r.get('image_model')}** failed: {r.get('error')}")

        latest_resp = api_client.get_latest_outputs(request_name)
        latest_images = latest_resp.get("images", []) if latest_resp.get("ok") else []

        if latest_images:
            run_ts = latest_resp.get("run_timestamp") or "-"
            st.caption(f"Most recent run found in output/{request_name}/ -- around {run_ts}")
            _render_grouped_images(latest_images, key_prefix="latest", allow_release=True, request_name=request_name)
        elif not (same_request_session and st.session_state.get("last_generate_error")):
            st.info("No output images found yet for this request. Click 'Run' in the sidebar.")


if _use_fragment:
    st.fragment(_render_output_body, run_every=2 if is_active else None)()
else:
    _render_output_body()
