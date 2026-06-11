"""
Streamlit frontend — VIB AI Chatbot V3
Intent-based: GREETING | PERSONAL_UNRELATED | PRODUCT_INFO_QA | PRODUCT_CONSULT | CUSTOMER_FEEDBACK
"""
import uuid
import httpx
import streamlit as st

st.set_page_config(
    page_title="VIB AI Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

INTENT_BADGE = {
    "GREETING_FAREWELL":   ("👋", "Chào hỏi",         "#4f8ef7"),
    "PERSONAL_UNRELATED":  ("💬", "Chủ đề cá nhân",   "#f5a623"),
    "PRODUCT_INFO_QA":     ("🔍", "Tra cứu thông tin", "#34a853"),
    "PRODUCT_CONSULT":     ("🎯", "Tư vấn sản phẩm",  "#c0397a"),
    "CUSTOMER_FEEDBACK":   ("📝", "Phản hồi KH",       "#7c5cbf"),
}

DOMAIN_BADGE = {
    "credit_card": "💳 Thẻ tín dụng",
    "insurance":   "🛡️ Bảo hiểm",
    "loan":        "🏠 Vay vốn",
    "savings":     "💰 Tiết kiệm",
    "general":     "🏦 Chung",
}

# ── Session state ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "UID0000"


# ── API helpers ──────────────────────────────────────────────────────────────
def check_api() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/admin/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def get_user_history(user_id: str) -> list:
    try:
        r = httpx.get(f"{API_BASE}/chat/history/{user_id}", timeout=10)
        return r.json().get("entries", [])
    except Exception:
        return []


def send_message(message: str, session_id: str, user_id: str = "UID0000") -> dict:
    try:
        r = httpx.post(
            f"{API_BASE}/chat/",
            json={"message": message, "session_id": session_id, "user_id": user_id},
            timeout=httpx.Timeout(connect=10.0, read=6000.0, write=10.0, pool=10.0),
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        return {"answer": "⏱️ Model xử lý quá lâu (>10 phút). Vui lòng thử lại hoặc kiểm tra Ollama.", "sources": []}
    except Exception as e:
        return {"answer": f"❌ Lỗi kết nối API: {e}", "sources": []}


def load_data(force_reload: bool = False) -> dict:
    try:
        r = httpx.post(
            f"{API_BASE}/admin/load?force_reload={str(force_reload).lower()}",
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_api_stats() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/admin/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 VIB AI Chatbot")
    st.caption("V2 — Intent-based LangGraph + DeepSeek-R1 + bge-m3")
    st.divider()

    # ── UserID ───────────────────────────────────────────────────────────────
    st.subheader("👤 Thông tin người dùng")
    user_id_input = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        placeholder="UID0000",
        help="Nhập UserID để lưu lịch sử chat. Mặc định: UID0000",
    )
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input or "UID0000"
        st.session_state.messages = []   # reset chat khi đổi user
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Đang dùng: **{st.session_state.user_id}**")
    st.divider()

    # Connection status
    health = check_api()
    if health:
        st.success("✅ API đang hoạt động")
    else:
        st.error(
            "❌ Không kết nối được API\n\n"
            "Chạy lệnh trong thư mục **version_2/**:\n"
            "```\npython -m uvicorn src.api.main:app --reload\n```"
        )

    if st.button("🔌 Kiểm tra lại kết nối"):
        st.rerun()

    st.divider()

    # Data management
    st.subheader("📦 Dữ liệu")
    stats = get_api_stats()
    vs = stats.get("vectorstore", {})
    hist = stats.get("history", {})
    if vs.get("ready"):
        st.success(f"✅ ChromaDB: {vs.get('count', 0):,} chunks")
    else:
        st.warning("⚠️ ChromaDB chưa load dữ liệu")
    if hist:
        st.info(
            f"📋 Cache: {hist.get('total_entries', 0)} Q&A "
            f"/ {hist.get('total_users', 0)} users "
            f"(indexed: {hist.get('chroma_indexed', 0)})"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Load data", use_container_width=True):
            with st.spinner("Đang load..."):
                result = load_data(force_reload=False)
                if result.get("status") in ("loaded", "already_loaded"):
                    st.success(f"✅ {result.get('message', 'OK')}")
                else:
                    st.error(f"❌ {result.get('message', 'Lỗi')}")
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            with st.spinner("Đang reload..."):
                result = load_data(force_reload=True)
                if result.get("status") == "loaded":
                    st.success(f"✅ {result.get('message', 'OK')}")
                else:
                    st.error(f"❌ {result.get('message', 'Lỗi')}")

    st.divider()

    # Session management
    st.subheader("🗂️ Session")
    st.code(f"ID: {st.session_state.session_id[:8]}...", language=None)

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Intent legend
    st.subheader("🧭 Intent")
    for intent, (icon, label, color) in INTENT_BADGE.items():
        st.markdown(
            f'<span style="color:{color}; font-size:12px">{icon} <b>{label}</b></span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick prompts
    st.subheader("💡 Thử ngay")
    quick_prompts = [
        ("👋", "Xin chào VIB"),
        ("💳", "Tôi muốn mở thẻ tín dụng"),
        ("🏠", "Tôi muốn vay mua nhà"),
        ("🛡️", "Tư vấn bảo hiểm cho tôi"),
        ("💰", "Tôi muốn gửi tiết kiệm"),
        ("🔍", "Lãi suất thẻ VIB Super Card?"),
        ("🔍", "Điều kiện mở thẻ Classic là gì?"),
    ]
    for icon, prompt in quick_prompts:
        if st.button(f"{icon} {prompt}", use_container_width=True, key=f"q_{prompt}"):
            st.session_state["quick_input"] = prompt
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Tư vấn sản phẩm ngân hàng VIB")
st.caption("Hỏi về thẻ tín dụng, bảo hiểm, vay vốn, tiết kiệm và các dịch vụ ngân hàng VIB")

# Display history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Xin chào! Tôi là trợ lý AI của VIB.\n\n"
            "Tôi có thể:\n"
            "- 🔍 **Tra cứu** thông tin sản phẩm/dịch vụ\n"
            "- 💳 **Tư vấn** thẻ tín dụng phù hợp\n"
            "- 🛡️ **Tư vấn** bảo hiểm theo nhu cầu\n"
            "- 🏠 **Tư vấn** vay vốn (nhà, xe, tiêu dùng)\n"
            "- 💰 **Tư vấn** tiết kiệm tối ưu\n\n"
            "Nhập câu hỏi hoặc chọn gợi ý ở bên trái!"
        )

    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🏦"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Show intent/domain/cache badges
            intent = msg.get("intent")
            domain = msg.get("advisor_domain")
            fc = msg.get("from_cache", False)
            cs = msg.get("cache_similarity")
            badges = []
            if fc:
                badges.append(
                    f'<span style="background:#ff990020; color:#cc7700; padding:2px 8px; '
                    f'border-radius:10px; font-size:11px">⚡ Cache ({cs:.2f})</span>'
                )
            if intent and intent in INTENT_BADGE:
                icon, label, color = INTENT_BADGE[intent]
                badges.append(
                    f'<span style="background:{color}20; color:{color}; padding:2px 8px; '
                    f'border-radius:10px; font-size:11px">{icon} {label}</span>'
                )
            if domain and domain in DOMAIN_BADGE:
                badges.append(
                    f'<span style="background:#88888820; color:#666; padding:2px 8px; '
                    f'border-radius:10px; font-size:11px">{DOMAIN_BADGE[domain]}</span>'
                )
            if badges:
                st.markdown(" ".join(badges), unsafe_allow_html=True)

            # Sources
            if msg.get("sources"):
                with st.expander("📄 Nguồn tài liệu"):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")

            # Advisor progress
            if msg.get("collected_info"):
                with st.expander("📋 Thông tin đã thu thập"):
                    for k, v in msg["collected_info"].items():
                        st.caption(f"• {k}: {v}")


# ── Input ─────────────────────────────────────────────────────────────────────
default_input = st.session_state.pop("quick_input", "")
user_input = st.chat_input("Nhập câu hỏi của bạn...", key="chat_input") or default_input

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant", avatar="🏦"):
        with st.spinner("Đang xử lý..."):
            response = send_message(user_input, st.session_state.session_id, st.session_state.user_id)

        answer = response.get("answer", "Xin lỗi, có lỗi xảy ra.")
        intent = response.get("intent")
        domain = response.get("advisor_domain")
        sources = response.get("sources", [])
        collected_info = response.get("collected_info")
        from_cache = response.get("from_cache", False)
        cache_sim = response.get("cache_similarity")

        st.markdown(answer)

        # Badge row
        badges = []
        if from_cache:
            badges.append(
                f'<span style="background:#ff990020; color:#cc7700; padding:2px 8px; '
                f'border-radius:10px; font-size:11px">⚡ Cache ({cache_sim:.2f})</span>'
            )
        if intent and intent in INTENT_BADGE:
            icon, label, color = INTENT_BADGE[intent]
            badges.append(
                f'<span style="background:{color}20; color:{color}; padding:2px 8px; '
                f'border-radius:10px; font-size:11px">{icon} {label}</span>'
            )
        if domain and domain in DOMAIN_BADGE:
            badges.append(
                f'<span style="background:#88888820; color:#666; padding:2px 8px; '
                f'border-radius:10px; font-size:11px">{DOMAIN_BADGE[domain]}</span>'
            )
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)

        if sources:
            with st.expander("📄 Nguồn tài liệu"):
                for src in sources:
                    st.caption(f"• {src}")

        if collected_info:
            with st.expander("📋 Thông tin đã thu thập"):
                for k, v in collected_info.items():
                    st.caption(f"• {k}: {v}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": intent,
        "advisor_domain": domain,
        "sources": sources,
        "collected_info": collected_info,
        "from_cache": from_cache,
        "cache_similarity": cache_sim,
    })
