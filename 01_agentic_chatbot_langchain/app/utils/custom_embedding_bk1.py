from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI
import boto3
import os
import json
import numpy as np
import faiss
from typing import List, Optional
from botocore.exceptions import ClientError
from app.config.env_loader import logger


class CustomEmbedding(Embeddings):
    """
    Custom embedding class supporting both:
    - Cohere multilingual embedding via AWS Bedrock.
    - Azure OpenAI text embedding (e.g. text-embedding-3-small or 3-large).

    The selected model is determined by `selected_embedding_model`:
        - 'cohere-multilingual' => Bedrock Cohere.
        - 'text-embedding'      => Azure OpenAI embedding.

    Attributes:
        use_cosine (bool): Whether to normalize embeddings using L2 norm.
        selected_embedding_model (str): Model selector string.
    """

    def __init__(
        self,
        selected_embedding_model: str = "text-embedding",  # hoặc 'cohere-multilingual'
        use_cosine: bool = True      
    ):
        """
        Initialize the embedding class.

        Args:
            selected_embedding_model (str): Choose between 'cohere-multilingual' or 'text-embedding'.
            use_cosine (bool): Apply L2 normalization for cosine similarity.
            azure_endpoint (str): Azure OpenAI endpoint URL.
            azure_api_key (str): Azure OpenAI API key.
            azure_api_version (str): Azure OpenAI API version.
            azure_deployment_name (str): Azure OpenAI deployment name for text embedding.
        """
        self.selected_embedding_model = selected_embedding_model
        self.use_cosine = use_cosine

    # -------------------------------------------------------------------------
    # INTERNAL API CALLER
    # -------------------------------------------------------------------------
    def _call_endpoint(self, text: str) -> List[float]:
        """
        Send a request to the selected embedding provider and return the embedding vector.
        """
        if isinstance(text, list):
            text = text[0]
        text = text.strip()

        logger.info(f"[Embedding] Model={self.selected_embedding_model}, Text={text[:80]}...")

        # ------------------ CASE 1: Cohere via AWS Bedrock ------------------
        if self.selected_embedding_model == "cohere-multilingual":
            try:
                bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                    aws_session_token=os.getenv("AWS_SSN_TOKEN"),
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                    verify=False,
                )

                model_id = "cohere.embed-multilingual-v3"
                json_params = {
                    "texts": [text[:2048]],
                    "truncate": "END",
                    "input_type": "search_document",
                }

                json_body = json.dumps(json_params)
                result = bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json_body,
                    contentType="application/json",
                )

                response = json.loads(result["body"].read().decode())
                return response["embeddings"][0]

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ExpiredTokenException":
                    logger.info("❌ Token AWS đã hết hạn. Vui lòng làm mới credentials.")
                logger.info(f"⚠️ AWS Bedrock Error: {error_code}")
                raise e

        # ------------------ CASE 2: Azure OpenAI text-embedding -------------
        elif self.selected_embedding_model == "text-embedding":
            try:
               
                azure_client = AzureOpenAI(
                    azure_endpoint=os.getenv('AZ_OPENAI_EMBEDDING_API_ENDPOINT') ,
                    api_key=os.getenv('AZ_OPENAI_EMBEDDING_API_KEY'),
                    api_version=os.getenv('AZ_OPENAI_EMBEDDING_DEPLOYMENT_VERSION')     
                )
                
                response = azure_client.embeddings.create(
                    model=os.getenv('AZ_OPENAI_EMBEDDING_DEPLOYMENT_NAME'),
                    input=[text],
                )

                embedding_vector = response.data[0].embedding
                return embedding_vector

            except Exception as e:
                logger.info(f"❌ Azure embedding error: {e}")
                raise e

        else:
            raise ValueError(
                f"Unsupported embedding model '{self.selected_embedding_model}'. "
                f"Valid options: 'cohere-multilingual', 'text-embedding'."
            )

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACES
    # -------------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            vector = self._call_endpoint(text)
            emb_arr = np.array(vector, dtype=np.float32).reshape(1, -1)
            if self.use_cosine:
                faiss.normalize_L2(emb_arr)
            embeddings.append(emb_arr.ravel().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector (normalized if use_cosine=True).
        """
        emb = self._call_endpoint(text)
        emb_arr = np.array(emb, dtype=np.float32).reshape(1, -1)
        if self.use_cosine:
            faiss.normalize_L2(emb_arr)
        return emb_arr.ravel().tolist()
