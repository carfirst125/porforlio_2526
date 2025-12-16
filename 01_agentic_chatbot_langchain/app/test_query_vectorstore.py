# test_query_vectorstore_fixed.py
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

# Path import — điều chỉnh nếu bạn để file ở chỗ khác
from app.utils.custom_embedding import CustomEmbedding
from app.config.env_loader import logger

load_dotenv()

# -----------------------------------------------------------------------------
# CẤU HÌNH
# -----------------------------------------------------------------------------
VECTORSTORE_PATH = "app/vectorstore/docs_index_cosine"
INDEX_FILE = os.path.join(VECTORSTORE_PATH, "index.faiss")
PICKLE_FILE = os.path.join(VECTORSTORE_PATH, "index.pkl")


# -----------------------------------------------------------------------------
# HÀM HỖ TRỢ TRÍCH XUẤT DOCS TỪ NHIỀU CẤU TRÚC PICKLE
# -----------------------------------------------------------------------------
def _extract_from_docstore_obj(docstore):
    """
    Trích danh sách Document từ một docstore object (LangChain docstore).
    Trả về dict mapping id -> Document-like object (có attributes .page_content, .metadata).
    """
    # LangChain docstore thường có attribute _dict
    if hasattr(docstore, "_dict"):
        return docstore._dict  # dict: id -> Document
    # Đôi khi docstore chính là dict
    if isinstance(docstore, dict):
        return docstore
    # Nếu docstore có method get_items / items
    if hasattr(docstore, "items"):
        try:
            return dict(docstore.items())
        except Exception:
            pass
    raise ValueError("Không thể trích thông tin từ docstore object.")


def _normalize_text_and_metadata_list(texts_list, metadatas_list):
    """
    Đảm bảo texts_list và metadatas_list tồn tại và cùng length.
    Nếu metadatas_list là None, tạo list empty dicts.
    """
    if texts_list is None:
        texts_list = []
    if metadatas_list is None:
        metadatas_list = [{} for _ in texts_list]
    # if lengths differ, pad metadatas
    if len(metadatas_list) < len(texts_list):
        metadatas_list = list(metadatas_list) + [{}] * (len(texts_list) - len(metadatas_list))
    return texts_list, metadatas_list


# -----------------------------------------------------------------------------
# LOAD VECTORSTORE (INDEX + METADATA) - ROBUST
# -----------------------------------------------------------------------------
def load_vectorstore(index_path: str, pickle_path: str):
    """
    Load FAISS index và metadata từ index.pkl với nhiều format khác nhau.
    Trả về:
        - index (faiss.Index)
        - texts: list of strings aligned with index positions (index position -> texts[pos])
        - metadatas: list of metadata dicts aligned với texts
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Không tìm thấy file FAISS index tại: {index_path}")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata index.pkl tại: {pickle_path}")

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load pickle content
    with open(pickle_path, "rb") as f:
        store_data = pickle.load(f)

    texts = None
    metadatas = None
    index_to_docstore_id = None

    # Case A: store_data is dict and uses LangChain save_local structure
    if isinstance(store_data, dict):
        # Common keys: 'docstore', 'index_to_docstore_id', 'texts', 'metadatas'
        if "texts" in store_data and isinstance(store_data["texts"], list):
            # custom manual save format
            texts = store_data["texts"]
            metadatas = store_data.get("metadatas", [{}] * len(texts))
            logger.info("Load index.pkl: found manual dict with 'texts' key.")
        elif "docstore" in store_data:
            docstore = store_data["docstore"]
            try:
                doc_dict = _extract_from_docstore_obj(docstore)
            except Exception as e:
                raise ValueError(f"Cấu trúc docstore không hợp lệ trong index.pkl: {e}")

            index_to_docstore_id = store_data.get("index_to_docstore_id", None)
            if index_to_docstore_id:
                # Build texts/metadatas aligned with index_to_docstore_id
                texts = []
                metadatas = []
                for did in index_to_docstore_id:
                    if did in doc_dict:
                        doc = doc_dict[did]
                    else:
                        # Some docstore keys may be bytes/int, try str
                        doc = doc_dict.get(str(did)) or doc_dict.get(int(did))
                    if doc is None:
                        texts.append("")
                        metadatas.append({})
                    else:
                        # Document object may be a LangChain Document or simple dict
                        if hasattr(doc, "page_content"):
                            texts.append(doc.page_content)
                            metadatas.append(getattr(doc, "metadata", {}))
                        elif isinstance(doc, dict) and "page_content" in doc:
                            texts.append(doc.get("page_content", ""))
                            metadatas.append(doc.get("metadata", {}))
                        else:
                            # fallback: stringify
                            texts.append(str(doc))
                            metadatas.append({})
                logger.info("Load index.pkl: extracted from dict docstore + index_to_docstore_id.")
            else:
                # No index mapping provided; iterate doc_dict values in insertion order
                texts = []
                metadatas = []
                for doc in doc_dict.values():
                    if hasattr(doc, "page_content"):
                        texts.append(doc.page_content)
                        metadatas.append(getattr(doc, "metadata", {}))
                    elif isinstance(doc, dict) and "page_content" in doc:
                        texts.append(doc.get("page_content", ""))
                        metadatas.append(doc.get("metadata", {}))
                    else:
                        texts.append(str(doc))
                        metadatas.append({})
                logger.info("Load index.pkl: extracted docstore dict values (no index mapping).")

    # Case B: store_data is object that wraps docstore (e.g., pickled FAISS wrapper)
    elif hasattr(store_data, "docstore"):
        try:
            docstore = store_data.docstore
            doc_dict = _extract_from_docstore_obj(docstore)
        except Exception as e:
            raise ValueError(f"Cấu trúc pickle không hợp lệ: thiếu docstore hoặc docstore không đọc được: {e}")

        index_to_docstore_id = getattr(store_data, "index_to_docstore_id", None)
        if index_to_docstore_id:
            texts = []
            metadatas = []
            for did in index_to_docstore_id:
                doc = doc_dict.get(did) or doc_dict.get(str(did)) or doc_dict.get(int(did))
                if doc is None:
                    texts.append("")
                    metadatas.append({})
                else:
                    if hasattr(doc, "page_content"):
                        texts.append(doc.page_content)
                        metadatas.append(getattr(doc, "metadata", {}))
                    elif isinstance(doc, dict) and "page_content" in doc:
                        texts.append(doc.get("page_content", ""))
                        metadatas.append(doc.get("metadata", {}))
                    else:
                        texts.append(str(doc))
                        metadatas.append({})
            logger.info("Load index.pkl: extracted from object.docstore + index_to_docstore_id.")
        else:
            # fallback iterate doc_dict
            texts = []
            metadatas = []
            for doc in doc_dict.values():
                if hasattr(doc, "page_content"):
                    texts.append(doc.page_content)
                    metadatas.append(getattr(doc, "metadata", {}))
                elif isinstance(doc, dict) and "page_content" in doc:
                    texts.append(doc.get("page_content", ""))
                    metadatas.append(doc.get("metadata", {}))
                else:
                    texts.append(str(doc))
                    metadatas.append({})
            logger.info("Load index.pkl: extracted from object.docstore values (no index mapping).")

    # Case C: store_data is tuple or list - try to find docstore / mapping inside
    elif isinstance(store_data, (tuple, list)):
        found = False
        # try each element
        for part in store_data:
            if isinstance(part, dict) and "docstore" in part:
                # reuse logic for dict
                tmp = part
                docstore = tmp["docstore"]
                try:
                    doc_dict = _extract_from_docstore_obj(docstore)
                except Exception:
                    continue
                index_to_docstore_id = tmp.get("index_to_docstore_id", None)
                if index_to_docstore_id:
                    texts = []
                    metadatas = []
                    for did in index_to_docstore_id:
                        doc = doc_dict.get(did) or doc_dict.get(str(did))
                        if doc is None:
                            texts.append("")
                            metadatas.append({})
                        else:
                            if hasattr(doc, "page_content"):
                                texts.append(doc.page_content)
                                metadatas.append(getattr(doc, "metadata", {}))
                            elif isinstance(doc, dict) and "page_content" in doc:
                                texts.append(doc.get("page_content", ""))
                                metadatas.append(doc.get("metadata", {}))
                            else:
                                texts.append(str(doc))
                                metadatas.append({})
                    found = True
                    break
                else:
                    # fallback iterate doc_dict
                    texts = []
                    metadatas = []
                    for doc in doc_dict.values():
                        if hasattr(doc, "page_content"):
                            texts.append(doc.page_content)
                            metadatas.append(getattr(doc, "metadata", {}))
                        elif isinstance(doc, dict) and "page_content" in doc:
                            texts.append(doc.get("page_content", ""))
                            metadatas.append(doc.get("metadata", {}))
                        else:
                            texts.append(str(doc))
                            metadatas.append({})
                    found = True
                    break
            # if part itself has docstore attribute
            if hasattr(part, "docstore"):
                try:
                    doc_dict = _extract_from_docstore_obj(part.docstore)
                except Exception:
                    continue
                index_to_docstore_id = getattr(part, "index_to_docstore_id", None)
                if index_to_docstore_id:
                    texts = []
                    metadatas = []
                    for did in index_to_docstore_id:
                        doc = doc_dict.get(did) or doc_dict.get(str(did))
                        if doc is None:
                            texts.append("")
                            metadatas.append({})
                        else:
                            if hasattr(doc, "page_content"):
                                texts.append(doc.page_content)
                                metadatas.append(getattr(doc, "metadata", {}))
                            elif isinstance(doc, dict) and "page_content" in doc:
                                texts.append(doc.get("page_content", ""))
                                metadatas.append(doc.get("metadata", {}))
                            else:
                                texts.append(str(doc))
                                metadatas.append({})
                    found = True
                    break
                else:
                    texts = []
                    metadatas = []
                    for doc in doc_dict.values():
                        if hasattr(doc, "page_content"):
                            texts.append(doc.page_content)
                            metadatas.append(getattr(doc, "metadata", {}))
                        elif isinstance(doc, dict) and "page_content" in doc:
                            texts.append(doc.get("page_content", ""))
                            metadatas.append(doc.get("metadata", {}))
                        else:
                            texts.append(str(doc))
                            metadatas.append({})
                    found = True
                    break
        if not found:
            raise ValueError("Cấu trúc pickle là tuple/list nhưng không tìm thấy docstore hay định dạng mong đợi.")
    else:
        raise ValueError("Không nhận dạng được cấu trúc file index.pkl. Vui lòng kiểm tra định dạng file.")

    # Ensure lists exist and lengths are consistent
    texts, metadatas = _normalize_text_and_metadata_list(texts, metadatas)

    # If index_to_docstore_id exists and length mismatch with faiss index, log a warning
    try:
        ntotal = index.ntotal
    except Exception:
        ntotal = None

    if ntotal is not None and len(texts) != ntotal:
        logger.warning(
            f"Số lượng vectors trong FAISS ({ntotal}) và số texts lấy được ({len(texts)}) không khớp. "
            "Sự không khớp có thể do lưu index khác cách. Kết quả truy vấn sẽ cố gắng sử dụng chỉ số trả về từ FAISS."
        )

    logger.info(f"✅ Đã load vectorstore: {len(texts)} items (FAISS ntotal={ntotal}).")
    return index, texts, metadatas


# -----------------------------------------------------------------------------
# TRUY VẤN VECTORSTORE BẰNG EMBED_QUERY()
# -----------------------------------------------------------------------------
def query_vectorstore(user_query: str, top_k: int = 5, selected_model: str = "text-embedding"):
    """
    Tính embedding cho câu hỏi bằng CustomEmbedding.embed_query()
    rồi truy vấn FAISS index và in kết quả (score, content, metadata).
    """
    # Khởi tạo embedding client (dùng biến môi trường nếu không truyền tham số)
    embedding_model = CustomEmbedding(selected_embedding_model=selected_model, use_cosine=True)

    # Tạo embedding query
    query_vector = embedding_model.embed_query(user_query)
    print(f"query_vector len: {len(query_vector)}")
    
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    print(f"query_vector len [2]: {len(query_vector)}")
    
    # Load index + metadata
    index, texts, metadatas = load_vectorstore(INDEX_FILE, PICKLE_FILE)

    # Thực hiện tìm kiếm
    scores, indices = index.search(query_vector, top_k)

    print(f"\n🔹 Kết quả Top-{top_k} cho câu hỏi: {user_query}\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        # FAISS trả -1 nếu không tìm thấy
        if idx == -1:
            continue
        # Nếu idx vượt quá texts length, in cảnh báo và skip
        if idx >= len(texts):
            logger.warning(f"Index trả về ({idx}) vượt quá chiều dài texts ({len(texts)}). Bỏ qua.")
            continue
        content = texts[idx]
        metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
        print(f"--- Document {rank} ---")
        print(f"Score: {score:.6f}")
        print(f"Content: {content[:1000]}{'...' if len(content) > 1000 else ''}")
        print(f"Metadata: {metadata}\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        query = input("Nhập câu hỏi của bạn: ").strip()
        if not query:
            print("⚠️ Câu hỏi không được để trống.")
        else:
            # bạn có thể đổi selected_model="cohere-multilingual" nếu index được build bằng cohere
            query_vectorstore(query, top_k=5, selected_model="text-embedding")
    except Exception as e:
        logger.error(f"❌ Lỗi khi chạy truy vấn: {e}")
        raise
