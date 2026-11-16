from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils import api_client  # noqa: E402
from webapp.utils.theme import apply_ocb_theme, badge, card, page_title_block  # noqa: E402

apply_ocb_theme("Generate")
page_title_block("🎨 Generate banner", "Chọn request, chế độ chạy, và sinh banner")

# ---------------------------------------------------------------------------
# [8a] Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình chạy")

    req_list = api_client.list_requests().get("requests", [])
    if req_list:
        request_name = st.selectbox("Input file request_name.json", req_list)
    else:
        request_name = st.text_input("Input file request_name.json (không kèm .json)", value="sample_request")
        st.caption("Không tìm thấy file nào trong input/ qua API — nhập tên thủ công.")

    models_info = api_client.list_models()
    available_models = models_info.get("available_providers", [])
    model_usage = models_info.get("model_usage", {})

    mode = st.selectbox("Chế độ chạy (mode)", ["default", "quality", "eco", "custom", "all"])

    custom_models: list[str] = []
    if mode == "custom":
        custom_models = st.multiselect("Chọn model (mode=custom)", available_models)

    st.divider()
    is_clone = st.checkbox("Clone thay vì generate mới (--mode=clone)")
    clone_source_path = ""
    clone_model = ""
    if is_clone:
        clone_source_path = st.text_input("Đường dẫn ảnh gốc để clone (path-to-select-image)")
        clone_model = st.selectbox("Model dùng để clone", available_models)

    st.divider()
    run_clicked = st.button("🚀 Chạy", use_container_width=True, type="primary")

if not request_name:
    st.stop()

# ---------------------------------------------------------------------------
# Load request detail
# ---------------------------------------------------------------------------
req_resp = api_client.get_request(request_name)
if not req_resp.get("ok"):
    st.error(f"Không load được request '{request_name}': {req_resp.get('error')}")
    st.stop()

req = req_resp["request"]

# ---------------------------------------------------------------------------
# [8b] Main screen: design brief, bucket asset gallery, text, model list, output size
# ---------------------------------------------------------------------------
with card():
    st.markdown("#### 🗒️ Design brief")
    design_brief = (req.get("design_brief") or "").strip()
    if design_brief:
        st.write(design_brief)
    else:
        st.caption("(request.json này không có design_brief)")

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
        st.caption("(request.json này không chỉ định asset nào)")
    else:
        asset_cols = st.columns(len(resolved_assets))
        for col, asset in zip(asset_cols, resolved_assets):
            with col:
                st.markdown(f"**{ASSET_LABELS.get(asset['label'], asset['label'])}**")
                if asset.get("kind") == "text":
                    # 2026-09-04 (user request): free-text description bucket
                    # entry -- no image was attached/resolved, show the text
                    # itself instead of calling st.image() on an empty path.
                    st.info(f"📝 {asset['value']}")
                else:
                    st.caption(f"`{asset['value']}`")
                    if asset.get("ok"):
                        st.image(asset["path"], width=180)
                    else:
                        st.error(f"Không tìm thấy file — {asset.get('error')}")

with card():
    st.markdown("#### 📝 Text nội dung")
    text = req.get("text", {})
    tcols = st.columns(3)
    tcols[0].metric("Headline", text.get("headline") or "-")
    tcols[1].metric("Message", text.get("message") or "-")
    tcols[2].metric("CTA", text.get("cta_label") or "-")

with card():
    st.markdown("#### 🧠 Model sẽ chạy & Output")
    if is_clone:
        st.markdown(f"**Clone mode** → model: {badge(clone_model or '(chưa chọn)', tone='green')}", unsafe_allow_html=True)
        st.write(f"Output sizes (`output.clone`): {req['output'].get('clone', [])}")
    else:
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
            st.warning("Chưa xác định được model nào — kiểm tra `model_usage.json`.")

        mcols = st.columns(2)
        mcols[0].metric("Output display size", req["output"].get("size"))
        mcols[1].metric("Số version / model", req.get("num_version", 1))

st.divider()

# ---------------------------------------------------------------------------
# [8c] Run + show results of THIS session only
# ---------------------------------------------------------------------------
if run_clicked:
    with st.status("⏳ Đang chạy — model đang sinh ảnh, có thể mất vài chục giây...", expanded=True) as status:
        if is_clone:
            if not clone_source_path or not clone_model:
                status.update(label="Thiếu thông tin clone", state="error")
                st.error("Cần nhập đường dẫn ảnh gốc và chọn model để clone.")
            else:
                st.write(f"Đang clone `{clone_source_path}` bằng model `{clone_model}`...")
                resp = api_client.clone(request_name, clone_source_path, clone_model)
                st.session_state["last_results"] = resp.get("results", [])
                status.update(label="Clone hoàn tất", state="complete")
        else:
            st.write(f"Đang generate cho request `{request_name}`, mode `{mode}`...")
            resp = api_client.generate(request_name, mode, custom_models)
            st.session_state["last_results"] = resp.get("results", [])
            status.update(label="Generate hoàn tất", state="complete")
    st.session_state["last_request_name"] = request_name

results = st.session_state.get("last_results", [])
if st.session_state.get("last_request_name") == request_name and results:
    st.markdown("#### 🖼️ Kết quả phiên chạy hiện tại")
    to_release: list[str] = []
    result_cols = st.columns(3)
    for i, r in enumerate(results):
        with result_cols[i % 3]:
            with card():
                if r.get("ok"):
                    st.image(r["output_path"], caption=f"{r['image_model']} · v{r['version']} · {r['display_size']}")
                    checked = st.checkbox("Chọn release", key=f"release_{i}_{r['output_path']}")
                    if checked:
                        to_release.append(r["output_path"])
                else:
                    st.error(f"**{r.get('image_model')}** lỗi:\n\n{r.get('error')}")

    if to_release and st.button(f"📦 Release {len(to_release)} ảnh đã chọn", type="primary"):
        rel_resp = api_client.release(request_name, to_release)
        released = rel_resp.get("released_paths", [])
        st.success(f"Đã release {len(released)} ảnh vào output/{request_name}/release/. Xem ở trang Release.")
elif st.session_state.get("last_request_name") == request_name:
    st.info("Chưa có kết quả nào trong phiên chạy này. Bấm '🚀 Chạy' ở sidebar.")
