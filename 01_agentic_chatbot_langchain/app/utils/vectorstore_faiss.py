import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Tuple

from app.config.env_loader import logger  # Load .env một lần duy nhất
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from app.utils.custom_embedding import CustomEmbedding #CustomCohereEmbedding  

class VectorstoreFaiss:
    """
    Utility class for creating, saving, loading, and querying a FAISS vectorstore
    from a DataFrame that contains precomputed embedding vectors.

    Attributes:
        vectorstore_path (str): Path to the directory where the FAISS vectorstore is saved or loaded.
        embedding_model (Embeddings): Custom embedding model (e.g., CustomCohereEmbedding).
        vectorstore (FAISS): The FAISS vectorstore instance.

    Methods:
        from_dataframe(df, input_col="input", embedding_col="embedding"):
            Build a FAISS vectorstore from a DataFrame containing text and its corresponding embeddings,
            and save it to the specified path.

        corpus_query(query, k=3):
            Query the vectorstore with a user input string and return the top-k most similar
            documents along with their similarity scores.

    Example:
        >>> vs = VectorstoreFaiss("vectorstores/my_index")
        >>> vs.from_dataframe(df, input_col="text", embedding_col="embedding")
        >>> results = vs.corpus_query("credit card benefits", k=3)
        >>> for content, score in results:
        >>>     print(f"{score:.4f} | {content[:100]}...")
    """
    def __init__(self, vectorstore_path: str):
        """
        Initialize the VectorstoreFaiss instance.

        Args:
            vectorstore_path (str): Directory path for saving or loading the FAISS vectorstore.
        """
        self.vectorstore_path = vectorstore_path
        self.embedding_model = CustomEmbedding()
        self.vectorstore = None

    def from_dataframe(self, df: pd.DataFrame, input_col="input", embedding_col="embedding"):
        """
        Create and save a FAISS vectorstore from a DataFrame containing texts and embeddings.

        Args:
            df (pd.DataFrame): A DataFrame with a text column and an embedding column.
            input_col (str): Name of the column containing input text.
            embedding_col (str): Name of the column containing embeddings (as stringified lists).
        """
        # 1. Lấy văn bản & vector embedding
        texts = df[input_col].astype(str).tolist()
        embeddings = df[embedding_col].apply(eval).tolist()  # Nếu là chuỗi dạng list

        # 2. Tạo Documents & metadata
        docs = []
        ids = []
        for i, text in enumerate(texts):
            doc_id = str(df.index[i])
            metadata = df.drop(columns=[input_col, embedding_col], errors="ignore").loc[i].to_dict()
            docs.append(Document(page_content=text, metadata=metadata))
            ids.append(doc_id)

        # 3. Tạo FAISS index
        embedding_matrix = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embedding_matrix)  # Chuẩn hóa cho cosine similarity
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embedding_matrix)

        # 4. Tạo docstore & index mapping
        docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in zip(ids, docs)})
        index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

        # 5. Tạo FAISS vectorstore
        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True
        )

        # 6. Lưu lại
        self.vectorstore.save_local(self.vectorstore_path)
        logger.info(f"✅ Vectorstore đã được lưu tại: {self.vectorstore_path}")

    def from_dataframe_with_embedding(self, df: pd.DataFrame, input_col="input"):
        """
        Create and save a FAISS vectorstore from a DataFrame containing only texts.
        Automatically generates embeddings using the embedding model.

        Args:
            df (pd.DataFrame): DataFrame containing a text column.
            input_col (str): Name of the column containing input text.
        """
        # 1. Lấy văn bản
        texts = df[input_col].astype(str).tolist()

        # 2. Tạo embeddings bằng model
        embeddings = self.embedding_model.embed_documents(texts)  # List[List[float]]

        # 3. Tạo Documents & metadata
        docs = []
        ids = []
        for i, text in enumerate(texts):
            doc_id = str(df.index[i])
            metadata = df.drop(columns=[input_col], errors="ignore").loc[i].to_dict()
            docs.append(Document(page_content=text, metadata=metadata))
            ids.append(doc_id)

        # 4. Tạo FAISS index (cosine similarity)
        embedding_matrix = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embedding_matrix)  # Chuẩn hóa cho cosine similarity
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embedding_matrix)

        # 5. Tạo docstore & mapping
        docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in zip(ids, docs)})
        index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

        # 6. Tạo FAISS vectorstore object
        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True
        )

        # 7. Lưu lại
        self.vectorstore.save_local(self.vectorstore_path)
        logger.info(f"✅ Vectorstore đã được lưu tại: {self.vectorstore_path}")


    def _load_vectorstore(self):
        """
        Load the FAISS vectorstore from disk if it hasn't been loaded into memory yet.
        """
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                folder_path=self.vectorstore_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )

    # def corpus_query(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
    #     """
    #     Query the FAISS vectorstore to retrieve the top-k most similar documents.

    #     Args:
    #         query (str): The input query string to search for.
    #         k (int): The number of top similar results to return.

    #     Returns:
    #         List[Tuple[str, float]]: A list of tuples containing document content and similarity score.
    #     """
    #     self._load_vectorstore()
    #     logger.info("Load vectorstore done...")
    #     try:
    #         results = self.vectorstore.similarity_search_with_score(query, k=k)
    #         logger.info(f"Query vectorstore done. Retrieved {len(results)} results.")
    #     except Exception as e:
    #         logger.error(f"❌ Lỗi khi truy vấn vectorstore: {e}")
    #         raise   
    #     return [(doc.page_content, score) for doc, score in results]


    def corpus_query(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        self._load_vectorstore()
        logger.info("Load vectorstore done...")

        try:
            # ✅ Debug kích thước embedding
            logger.info(f"[DEBUG] Embedding vector dimension check... \n query={query}")
            query_vec = self.embedding_model.embed_query(query)
            print(f"[DEBUG] Query embedding dim = {len(query_vec)}")
            print(f"[DEBUG] Index embedding dim = {self.vectorstore.index.d}")

            # kiểm tra nếu mismatch thì raise rõ ràng
            if len(query_vec) != self.vectorstore.index.d:
                raise ValueError(
                    f"❌ Dimension mismatch: query vector ({len(query_vec)}) "
                    f"!= index ({self.vectorstore.index.d}). "
                    "Hãy đảm bảo model embedding trùng với model khi build FAISS."
                )

            # tiếp tục nếu hợp lệ
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Query vectorstore done. Retrieved {len(results)} results.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi truy vấn vectorstore: {e}")
            raise
        return [(doc.page_content, score) for doc, score in results]
