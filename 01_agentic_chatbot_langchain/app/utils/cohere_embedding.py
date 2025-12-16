from langchain_core.embeddings import Embeddings
import boto3
import os
import json
import numpy as np
import faiss
from typing import List
from botocore.exceptions import ClientError
from app.config.env_loader import logger  # Load .env một lần duy nhất

class CustomCohereEmbedding(Embeddings):
    """
    Custom embedding class for integrating Cohere multilingual embedding via AWS Bedrock.

    This class provides methods to obtain embedding vectors from the Cohere model
    hosted on AWS Bedrock. It supports both document embedding and query embedding.
    If cosine similarity is used, L2 normalization is automatically applied.

    Attributes:
        use_cosine (bool): Whether to normalize the embeddings using L2 norm
                           for cosine similarity compatibility.

    Methods:
        _call_endpoint(text: str) -> List[float]:
            Sends a request to Cohere via AWS Bedrock to retrieve the embedding vector.

        embed_documents(texts: List[str]) -> List[List[float]]:
            Embeds a list of documents and returns a list of embedding vectors.

        embed_query(text: str) -> List[float]:
            Embeds a single query and returns a normalized embedding vector (if cosine is used).
    """
    
    def __init__(self, use_cosine: bool = True):
        """
        Initialize the embedding class.

        Args:
            use_cosine (bool): If True, apply L2 normalization to the output embeddings
                               (recommended when using cosine similarity).
        """
        self.use_cosine = use_cosine

    def _call_endpoint(self, text: str) -> List[float]:
        """
        Internal method to send text to the Cohere model via AWS Bedrock and receive the embedding.

        Args:
            text (str): A single input string to be embedded.

        Returns:
            List[float]: The embedding vector received from Cohere.

        Raises:
            ClientError: If there is an error with AWS credentials or API call.
        """
        if isinstance(text, list):
            text = text[0]

        logger.info(f"[Embedding] Text: {text}")
        response = ""

        try:
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
                aws_session_token=os.getenv('AWS_SSN_TOKEN'),
                region_name="us-east-1",
                verify=False
            )

            model_id = "cohere.embed-multilingual-v3"
            json_params = {
                'texts': [text[:2048]],
                'truncate': "END",
                'input_type': "search_document"
            }

            json_body = json.dumps(json_params)
            result = bedrock_client.invoke_model(
                modelId=model_id,
                body=json_body,
                contentType="application/json"
            )

            response = json.loads(result['body'].read().decode())
            return response["embeddings"][0]

        except ClientError as e:
            logger.info(f"ClientError: {e}")
            error_code = e.response['Error']['Code']
            if error_code == 'ExpiredTokenException':
                logger.info("❌ Token đã hết hạn. Vui lòng làm mới AWS credentials.")
                raise e
            else:
                logger.info(f"⚠️ Lỗi khác: {error_code} - {e.response['Error']['Message']}")
                raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (or input strings).

        Args:
            texts (List[str]): A list of text strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            vector = self._call_endpoint([text])
            emb_arr = np.array(vector, dtype=np.float32).ravel()
            embeddings.append(emb_arr.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text (str): The query to embed.

        Returns:
            List[float]: The embedding vector (normalized if use_cosine=True).
        """
        emb = self._call_endpoint(text)
        emb_arr = np.array(emb, dtype=np.float32).reshape(1, -1)
        if self.use_cosine:
            faiss.normalize_L2(emb_arr)
        return emb_arr.ravel().tolist()
