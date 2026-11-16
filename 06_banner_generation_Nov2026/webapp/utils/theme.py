"""OCB branding cho webapp: màu sắc tham khảo `lib/theme/` (xem OCB_* hex
constants bên dưới + `.streamlit/config.toml`) + logo tại
`lib/ocb_asset/logo.png` (REQUEST.md [8e], `logo_path()` cũng thử
`lib/my/ocb_asset/logo.png` -- vị trí thật của file này trong `lib/` hiện tại).

UPDATE 2026-08-15 (redesign — user report "màu rất tệ, không đọc được gì"):
Bản trước tự chèn CSS `section[data-testid="stSidebar"] * { color: white
!important; }` — selector `*` này ép TOÀN BỘ phần tử con trong sidebar
(kể cả input/label/value BÊN TRONG mỗi widget) sang chữ trắng, trong khi nền
"chrome" mặc định của widget (selectbox/text_input/multiselect...) vẫn là
màu sáng -> chữ trắng trên nền sáng = không đọc được, đúng như user report.

Fix: màu sắc giờ khai báo ở `.streamlit/config.toml` (theme NATIVE của
Streamlit) — Streamlit tự tính contrast đúng cho MỌI widget, kể cả trong
sidebar, dựa trên `secondaryBackgroundColor` + `textColor`. File này CHỈ còn
chèn CSS phạm vi hẹp (chọn đúng class/tag cụ thể: heading, badge, divider),
KHÔNG BAO GIỜ dùng selector `*` hay động vào nội bộ widget nữa.

UPDATE 2026-08-15 (2) (theo yêu cầu user): trước đây `apply_ocb_theme()` vừa
gọi `st.set_page_config()` vừa tự vẽ logo/title vào sidebar -- và vì mỗi
trang Streamlit "classic" (`webapp/pages/*.py`) tự động có 1 khối nav link
do Streamlit tự chèn Ở ĐẦU sidebar (không thể đổi thứ tự bằng code), nên
logo/title luôn bị nav link đè lên trên, không thể "đưa lên trên cùng" như
user muốn. Giải pháp: chuyển sang `st.navigation(..., position="hidden")`
(webapp/Home.py đóng vai trò router) để tắt hẳn nav widget tự động, rồi tự
vẽ link bằng `st.page_link()` theo đúng thứ tự mình muốn. Theo đó hàm này
tách làm 2:
  - `inject_base_css()`: gọi 1 lần duy nhất ở router (Home.py), TRƯỚC
    `st.navigation()` -- chỉ lo `st.set_page_config()` + CSS trang trí.
  - `render_sidebar_nav(pages)`: gọi 1 lần duy nhất ở router, SAU
    `st.navigation()` nhưng TRƯỚC `pg.run()` -- vẽ logo + title lên đầu
    sidebar, rồi mới tới các `st.page_link()`.
Nhờ vậy logo/title luôn ở trên cùng, và không còn cần gọi lại theme mỗi khi
đổi trang (mỗi *view* trong `webapp/views/` chỉ cần gọi `page_title_block()`
cho phần header ở khu vực nội dung chính)."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from app.settings import get_settings  # noqa: E402

# Cùng bộ màu OCB dùng xuyên suốt webapp (khớp với .streamlit/config.toml).
OCB_GREEN = "#00853E"
OCB_DARK_GREEN = "#004823"
OCB_GOLD = "#D4A017"
OCB_LIGHT_GREEN_BG = "#EAF5EE"
OCB_TEXT = "#17241C"


def logo_path() -> Path | None:
    """`Settings` has no `asset_library_path` attribute (2026-08-20 bug fix
    -- the real field is `asset_library_dir`, `AttributeError:
    'Settings' object has no attribute 'asset_library_path'` when opening
    any page). Also tries BOTH `lib/ocb_asset/logo.png` (the literal path
    named in REQUEST.md [8e]) and `lib/my/ocb_asset/logo.png` (where the
    logo actually lives in this project's real `lib/` tree right now),
    same "trust the real file layout over the literal doc text" lesson as
    `app/models/schemas.py`'s request.json schema correction."""
    base = get_settings().asset_library_dir
    for candidate in (base / "ocb_asset" / "logo.png", base / "my" / "ocb_asset" / "logo.png"):
        if candidate.exists():
            return candidate
    return None


def inject_base_css() -> None:
    """Gọi 1 lần duy nhất, ở đầu router (`webapp/Home.py`), TRƯỚC
    `st.navigation()`. Set page config + CSS trang trí phạm vi hẹp (heading,
    badge, divider, bo góc nút) -- không đè màu chữ/nền của bất kỳ widget
    input nào, màu sắc chính lấy từ `.streamlit/config.toml` (theme native)."""
    st.set_page_config(page_title="OCB — DCO", page_icon="🟢", layout="wide")

    st.markdown(
        f"""
        <style>
        /* Heading dùng màu xanh đậm OCB cho đồng bộ thương hiệu -- chỉ nhắm
           đúng thẻ h1/h2/h3, KHÔNG ảnh hưởng chữ trong widget. */
        h1, h2, h3 {{
            color: {OCB_DARK_GREEN};
        }}
        /* Badge nhỏ dùng để đánh dấu trạng thái (vd "Đang dùng", tên model) --
           class riêng, không đụng tới widget nào khác. */
        .ocb-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .ocb-badge-gold {{ background-color: {OCB_GOLD}; color: {OCB_TEXT}; }}
        .ocb-badge-green {{ background-color: {OCB_GREEN}; color: #FFFFFF; }}
        /* Accent bar mảnh dưới tiêu đề mỗi trang, chỉ decor. */
        .ocb-accent-bar {{
            height: 4px;
            width: 64px;
            background: linear-gradient(90deg, {OCB_GREEN}, {OCB_GOLD});
            border-radius: 2px;
            margin: 2px 0 14px 0;
        }}
        /* Bo góc + đậm chữ cho button -- màu nền/chữ đã do primaryColor
           trong config.toml quyết định, ở đây chỉ chỉnh hình dạng. */
        div.stButton > button {{
            border-radius: 8px;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav(pages: list) -> None:
    """Gọi 1 lần duy nhất ở router (`webapp/Home.py`), SAU `st.navigation()`
    nhưng TRƯỚC `pg.run()`. Vẽ logo + title "DCO — Dynamic Creative
    Optimization" lên ĐẦU sidebar, rồi mới tới link điều hướng từng trang
    (`st.page_link`) -- thứ tự này chỉ có thể ép được vì router dùng
    `st.navigation(pages, position="hidden")` để tắt nav widget tự động của
    Streamlit (nav widget tự động luôn nằm trên cùng, không đổi thứ tự
    được)."""
    logo = logo_path()
    with st.sidebar:
        if logo:
            st.image(str(logo), width=140)
        else:
            st.markdown("### 🟢 OCB")
        st.markdown("**DCO — Dynamic Creative Optimization**")
        st.caption("OCB · Image GenAI demo")
        st.divider()
        for p in pages:
            st.page_link(p)
        st.divider()


def page_title_block(title: str, subtitle: str = "", show_logo: bool = True) -> None:
    """Header đồng bộ cho mỗi trang: logo (nếu có) + title + accent bar +
    subtitle, rồi 1 divider -- dùng ở đầu mỗi view trong `webapp/views/`.

    UPDATE 2026-08-16 (theo yêu cầu user): trang Generate không muốn logo OCB
    ở khu vực nội dung bên phải nữa -- title cần nằm ngay đầu dòng. Thêm
    tham số `show_logo` (mặc định True, KHÔNG đổi hành vi của Home/Release)
    để `webapp/views/generate.py` tắt riêng logo cho trang này mà không ảnh
    hưởng các trang khác hay logo trên sidebar (`render_sidebar_nav`, vẫn giữ
    nguyên -- yêu cầu chỉ nói tới "nội dung bên phải", không phải sidebar)."""
    logo = logo_path() if show_logo else None
    if logo:
        cols = st.columns([1, 8])
        with cols[0]:
            st.image(str(logo), width=64)
        with cols[1]:
            st.title(title)
    else:
        st.title(title)

    st.markdown('<div class="ocb-accent-bar"></div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def badge(text: str, tone: str = "gold") -> str:
    """Trả về HTML 1 badge nhỏ (dùng với `st.markdown(..., unsafe_allow_html=True)`).
    `tone`: "gold" (mặc định, dùng cho trạng thái/nhãn thường) hoặc "green"
    (dùng cho trạng thái tích cực/đang chọn)."""
    css_class = "ocb-badge-green" if tone == "green" else "ocb-badge-gold"
    return f'<span class="ocb-badge {css_class}">{text}</span>'


def image_stretched(source: str, caption: str | None = None) -> None:
    """`st.image()` co giãn hết chiều ngang container, nhưng KHÔNG giả định
    trước tên tham số nào đang dùng được trên bản streamlit đang chạy.

    UPDATE 2026-08-16 (bug #34): user report lỗi thật trên máy
    (`C:\\...\\04_pyenv\\langchain_bot_env`):
    `TypeError: ImageMixin.image() got an unexpected keyword argument
    'use_container_width'` -- streamlit đã đổi tên tham số này của RIÊNG
    `st.image()` qua nhiều lần (`use_column_width` (cũ) -> `use_container_width`
    -> `width="stretch"` (mới, `use_container_width` bị deprecate sau
    2025-12-31)) và bản cài trên máy user không khớp cái tên mà lần trước tôi
    giả định (1.61.1, theo ghi chú phiên trước -- hoá ra sai, hoặc user đang
    chạy Streamlit từ 1 venv khác/cũ hơn cho trang này). Vì không có cách nào
    từ sandbox này biết CHÍNH XÁC bản streamlit thật trên máy user, hàm này
    tự dò: thử lần lượt từng cách gọi, dùng cách đầu tiên KHÔNG lỗi
    `TypeError` (báo hiệu tham số không tồn tại) -- image_manager, không phải
    người dùng, phải chịu trách nhiệm tương thích ngược/xuôi này."""
    kwargs = {"caption": caption} if caption else {}
    for extra in ({"use_container_width": True}, {"use_column_width": True}, {"width": "stretch"}, {}):
        try:
            st.image(source, **kwargs, **extra)
            return
        except TypeError:
            continue


@contextmanager
def card(border: bool = True):
    """Container dạng "card" nhất quán cho mọi trang -- bọc 1 khối nội dung
    (asset gallery, kết quả sinh ảnh, form...) trong khung bo góc có viền,
    giúp phân tách các phần rõ ràng hơn so với bản cũ (chỉ dùng
    `st.subheader` rời rạc, không có khung/nhóm trực quan)."""
    with st.container(border=border) as c:
        yield c
