import streamlit as st  
import importlib
import warnings
warnings.filterwarnings('ignore')
 
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Data Science Board", page_icon=":bar_chart:", layout="wide")

# Tạo danh sách các page
pages = {
    "DRAG Agentic Chatbot": "agentic_chatbot_api",
}

# Chọn page từ sidebar
selected_page = st.sidebar.radio("Select Page:", list(pages.keys()))

# Import page tương ứng, chỉ chạy page được chọn
page_module = importlib.import_module(f"pages.{pages[selected_page]}")
page_module.run()