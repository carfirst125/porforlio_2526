##########################################
# chatbot_api
# Show chat result of current local chatbot API
##########################################

import streamlit as st
import datetime
import time
import requests
import hashlib


# =========================================================
# Utils
# =========================================================

def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# =========================================================
# Chatbot API Caller
# =========================================================

def agentic_chatbot(**kwargs):
    """
    Call chatbot API and return normalized response
    """

    question = kwargs.get("question", "")
    userid   = kwargs.get("userid", "user_001")
    url      = kwargs.get("url")

    payload = {
        "userid": userid,
        "question": question
    }

    try:
        response = requests.post(
            url=url,
            json=payload,
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Connection error: {e}")
        return {}

    if response.status_code != 200:
        st.error(f"❌ HTTP {response.status_code}: {response.text}")
        return {}

    try:
        datax = response.json()
    except ValueError:
        st.error("❌ Invalid JSON response from server")
        return {}

    return {
        "data": {
            "content": datax.get("answer", "")
            # "token_usage": datax.get("token_usage", {})
        }
    }


# =========================================================
# Main App
# =========================================================

def run():

    # =====================================================
    # Page title
    # =====================================================
    st.title("📊 GenAI Chatbot API")
    st.write(
        """
        **Description**
        - This web app is used for testing chatbot APIs
        - Just provide your chatbot endpoint and start chatting
        """
    )
    st.markdown("##")

    # =====================================================
    # Style
    # =====================================================
    st.markdown("---")
    st.write("**Chatbot Log Information**")
    st.markdown(
        """
        <style>
        div[data-baseweb="input"] {
            background-color: #D3D3D3 !important;
            border-radius: 4px;
            padding: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # Settings
    # =====================================================
    chat_endpoint = st.text_input(
        "Chatbot Endpoint:",
        placeholder="http://localhost:8001/chat"
    )

    TODAY = datetime.date.today().strftime("%Y%m%d")
    layout = st.columns([1, 5, 5, 5], vertical_alignment="top")

    with layout[1]:
        chat_channel = st.text_input("Chat Channel:", value="streamlit_webapp")

    with layout[2]:
        chat_convid = st.text_input("Conversation ID:", value=TODAY)
        conv_id = md5_hash(chat_convid)
        st.caption(f"Conversation ID Hash: `{conv_id}`")

    with layout[3]:
        userid = st.text_input("UserID:", value="YOUR_USERID")

    # =====================================================
    # Conversation
    # =====================================================
    st.markdown("---")
    st.subheader("💬 Let's chat with the bot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    # Conversation history
    st.subheader("Conversation:")
    for idx, msg in enumerate(st.session_state.messages):
        if idx % 2 == 0:
            st.write(f"**You:** {msg}")
        else:
            st.write(f"**Bot:** {msg}", unsafe_allow_html=True)

    # =====================================================
    # Input
    # =====================================================
    user_input = st.text_input(
        "You:",
        key=f"input_{st.session_state.input_key}"
    )

    if user_input:

        if not chat_endpoint:
            st.warning("⚠️ Please provide chatbot endpoint URL")
            return

        start_time = time.time()

        kwargs = {
            "url": chat_endpoint,
            "question": user_input,
            "userid": userid,
            "conversation_id": conv_id,
            "channel": chat_channel,
        }

        response = agentic_chatbot(**kwargs)

        try:
            answer = response["data"]["content"]
        except Exception:
            st.error("❌ Unexpected response format")
            st.write(response)
            return

        elapsed_time = round(time.time() - start_time, 2)
        answer_mess = (
            f"{answer}"
            f"<br><small><em>⏱ elapsed time: {elapsed_time}s</em></small>"
        )

        # Save conversation
        st.session_state.messages.append(user_input)
        st.session_state.messages.append(answer_mess)

        # Reset input
        st.session_state.input_key += 1
        st.rerun()
