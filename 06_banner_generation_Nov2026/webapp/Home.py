"""Streamlit entrypoint / router.

Run:
    streamlit run webapp/Home.py

Requires the FastAPI backend running at BACKEND_BASE_URL (default
http://localhost:8000, see `.env`):
    uvicorn app.main:app --reload

UPDATE 2026-08-15 (theo yêu cầu user): file này trước đây tự vẽ nội dung
trang Home. Giờ chuyển thành ROUTER duy nhất của cả app, dùng
`st.navigation(pages, position="hidden")` để tự quản lý danh sách trang +
TẮT nav widget tự động của Streamlit (nav widget tự động luôn nằm trên
cùng sidebar, không thể chèn logo/title lên trước nó được). Nhờ vậy
`render_sidebar_nav()` có thể vẽ logo + "DCO" lên ĐẦU sidebar, rồi mới tới
link điều hướng từng trang -- đúng yêu cầu "logo ocb với DCO đưa lên trên
cùng". Nội dung từng trang giờ nằm ở `webapp/views/{home,generate,release}.py`
(thay cho `webapp/pages/*.py` cũ, không còn dùng auto-discovery nữa)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils.theme import inject_base_css, render_sidebar_nav  # noqa: E402

inject_base_css()

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/generate.py", title="Generate", icon="🎨"),
    st.Page("views/release.py", title="Release", icon="🚀"),
]
pg = st.navigation(pages, position="hidden")
render_sidebar_nav(pages)
pg.run()
