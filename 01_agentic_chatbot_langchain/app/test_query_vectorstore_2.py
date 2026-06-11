# app/tests/test_query_vectorstore_refactored.py

import os
from dotenv import load_dotenv
from app.config.env_loader import logger
from app.utils.vectorstore_faiss import VectorstoreFaiss  # <-- class ở file 1

load_dotenv()

# -------------------------------------------------------------------------
# CẤU HÌNH
# -------------------------------------------------------------------------
VECTORSTORE_PATH = "app/vectorstore/docs_index_cosine"


# -------------------------------------------------------------------------
# TRUY VẤN VECTORSTORE BẰNG LANGCHAIN CHUẨN
# -------------------------------------------------------------------------
def query_vectorstore(user_query: str, top_k: int = 5):
    """
    Thực hiện truy vấn top-k từ FAISS vectorstore đã build sẵn.
    - Tự động load vectorstore (nếu chưa load)
    - Tự động embedding query
    - Trả về top-k nội dung & metadata
    """
    logger.info(f"🔍 Đang load vectorstore từ: {VECTORSTORE_PATH}")

    # 1️⃣ Khởi tạo đối tượng vectorstore (tự load khi cần)
    vs = VectorstoreFaiss(VECTORSTORE_PATH)

    # 2️⃣ Gọi corpus_query (đã có sẵn trong class)
    results = vs.corpus_query(user_query, k=top_k)

    # 3️⃣ In kết quả
    print(f"\n🔹 Kết quả Top-{top_k} cho truy vấn: {user_query}\n")
    for i, (content, score) in enumerate(results, start=1):
        snippet = (content[:500] + "...") if len(content) > 500 else content
        print(f"--- Document {i} ---")
        print(f"Score: {score:.4f}")
        print(f"Content: {snippet}\n")

    return results


# -------------------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        query = input("Nhập câu hỏi của bạn: ").strip()
        if not query:
            print("⚠️ Câu hỏi không được để trống.")
        else:
            query_vectorstore(query, top_k=5)
    except Exception as e:
        logger.error(f"❌ Lỗi khi chạy truy vấn: {e}")
        raise
