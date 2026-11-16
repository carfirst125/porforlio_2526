from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils import api_client  # noqa: E402
from webapp.utils.theme import apply_ocb_theme, badge, card, page_title_block  # noqa: E402

apply_ocb_theme("Release")
page_title_block("🚀 Release", "Ảnh đã chọn release + clone sang các kích thước khác")

with st.sidebar:
    st.markdown("### 📄 Request")
    req_list = api_client.list_requests().get("requests", [])
    default_idx = 0
    if req_list and st.session_state.get("last_request_name") in req_list:
        default_idx = req_list.index(st.session_state["last_request_name"])
    request_name = (
        st.selectbox("Request", req_list, index=default_idx)
        if req_list
        else st.text_input("Request name", value=st.session_state.get("last_request_name", "sample_request"))
    )

if not request_name:
    st.stop()

released_resp = api_client.get_released(request_name)
released_paths = [p for p in released_resp.get("released", []) if "_clone" not in Path(p).stem]

with card():
    st.markdown(f"#### 📦 Ảnh đã release — request: `{request_name}`")

    if not released_paths:
        st.info("Chưa có ảnh nào được release. Sang trang Generate, chạy rồi tick 'Chọn release'.")
        st.stop()

    cols = st.columns(3)
    for i, p in enumerate(released_paths):
        with cols[i % 3]:
            st.image(p, caption=Path(p).name)

st.divider()

# ---------------------------------------------------------------------------
# [8e] Clone ảnh đã release sang kích thước khác
# ---------------------------------------------------------------------------
with card():
    st.markdown("#### 🧬 Clone ảnh đã release sang kích thước khác")

    preview_col, form_col = st.columns([1, 2])
    with preview_col:
        select_image = st.selectbox("Chọn 1 ảnh trong release", released_paths, format_func=lambda p: Path(p).name)
        st.image(select_image, caption="Ảnh nguồn để clone", use_container_width=True)

    with form_col:
        models_info = api_client.list_models()
        available_models = models_info.get("available_providers", [])

        display_sizes = st.multiselect(
            "Kích thước output (multiselect)", ["300,250", "728,90", "1024,768", "1536,1200", "320,480"]
        )
        extra_size = st.text_input('Hoặc nhập thêm size khác (vd "1080,1080")')
        if extra_size:
            display_sizes.append(extra_size)

        clone_model = st.selectbox("Image model", available_models)

        if display_sizes:
            st.markdown(" ".join(badge(s) for s in display_sizes), unsafe_allow_html=True)

        run_clone_clicked = st.button("🧬 Chạy clone", use_container_width=True, type="primary")

if run_clone_clicked:
    if not display_sizes or not clone_model:
        st.error("Cần chọn ít nhất 1 kích thước và 1 model.")
    else:
        with st.status(f"⏳ Đang clone `{Path(select_image).name}` sang {len(display_sizes)} kích thước...", expanded=True) as status:
            resp = api_client.clone_from_release(request_name, select_image, display_sizes, clone_model)
            results = resp.get("results", [])
            st.session_state["last_clone_results"] = results
            status.update(label="Clone hoàn tất", state="complete")

clone_results = st.session_state.get("last_clone_results", [])
if clone_results:
    with card():
        st.markdown("#### Kết quả clone")
        ccols = st.columns(3)
        for i, r in enumerate(clone_results):
            with ccols[i % 3]:
                if r.get("ok"):
                    st.image(r["output_path"], caption=f"{r['image_model']} · {r['display_size']}")
                else:
                    st.error(f"**{r.get('display_size')}** lỗi:\n\n{r.get('error')}")
