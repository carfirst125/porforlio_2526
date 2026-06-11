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
import tiktoken

class CustomEmbedding(Embeddings):
    """
    Custom embedding class supporting both:
    - Cohere multilingual embedding via AWS Bedrock.
    - Azure OpenAI text embedding (e.g. text-embedding-3-small or 3-large).

    The selected model is determined by `selected_embedding_model`:
        - 'cohere-multilingual' => Bedrock Cohere.
        - 'text-embedding'      => Azure OpenAI embedding.
    """

    def __init__(
        self,
        selected_embedding_model: str = "text-embedding",
        use_cosine: bool = True,
        # ---- Azure OpenAI parameters ----
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        # ---- AWS Bedrock parameters ----
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: str = "us-east-1",
    ):
        """
        Initialize the embedding class.

        Args:
            selected_embedding_model (str): 'cohere-multilingual' or 'text-embedding'.
            use_cosine (bool): Apply L2 normalization for cosine similarity.

            azure_endpoint (str): Azure OpenAI endpoint URL.
            azure_api_key (str): Azure OpenAI API key.
            azure_api_version (str): Azure OpenAI API version.
            azure_deployment_name (str): Azure OpenAI deployment name for embeddings.

            aws_access_key (str): AWS Access Key ID for Bedrock.
            aws_secret_key (str): AWS Secret Access Key for Bedrock.
            aws_session_token (str): AWS Session Token (optional).
            aws_region (str): AWS region name (default: 'us-east-1').
        """
        self.selected_embedding_model = selected_embedding_model
        self.use_cosine = use_cosine

        # Azure OpenAI configs
        self.azure_endpoint = azure_endpoint or os.getenv("AZ_OPENAI_EMBEDDING_API_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZ_OPENAI_EMBEDDING_API_KEY")
        self.azure_api_version = azure_api_version or os.getenv("AZ_OPENAI_EMBEDDING_DEPLOYMENT_VERSION")
        self.azure_deployment_name = azure_deployment_name or os.getenv("AZ_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        # AWS Bedrock configs
        self.aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY")
        self.aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_KEY")
        self.aws_session_token = aws_session_token or os.getenv("AWS_SSN_TOKEN")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")

        # Pre-initialize clients for better performance
        self._init_clients()

    # -------------------------------------------------------------------------
    # CLIENT INITIALIZATION
    # -------------------------------------------------------------------------
    def _init_clients(self):
        """Initialize clients for selected embedding provider."""
        if self.selected_embedding_model == "text-embedding":
            if not all([self.azure_endpoint, self.azure_api_key, self.azure_api_version, self.azure_deployment_name]):
                raise ValueError("Azure OpenAI configuration is incomplete for text-embedding model.")
            self.azure_client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )
            logger.info("✅ Azure OpenAI client initialized successfully.")

        elif self.selected_embedding_model == "cohere-multilingual":
            if not all([self.aws_access_key, self.aws_secret_key]):
                raise ValueError("AWS credentials are incomplete for Cohere multilingual model.")
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region,
                verify=False,
            )
            logger.info("✅ AWS Bedrock client initialized successfully.")

        else:
            raise ValueError(
                f"Unsupported embedding model '{self.selected_embedding_model}'. "
                f"Valid options: 'cohere-multilingual', 'text-embedding'."
            )

    # -------------------------------------------------------------------------
    # Calculate number of tokens
    # ------------------------------------------------------------------------- 
    def num_tokens_in_text(self, text: str, model_name="text-embedding-3-small") -> int:

        """
        Count the number of tokens in a given text using the specified model.

        Args:
            text (str): Input text to be tokenized.
            model_name (str): Name of the model to use for tokenization.
                Defaults to 'text-embedding-3-small'.
        Returns:
            int: Number of tokens in the input text.
        """   
        #logger.info(f"[Embedding] Counting tokens for model: {model_name}, text: {text[:50]}...")
        encoding = tiktoken.encoding_for_model(model_name)
        #logger.info(f"[Embedding] Encoding used: {encoding.name}")
        num_tokens = len(encoding.encode(text))    
        return num_tokens
    
    # -------------------------------------------------------------------------
    # INTERNAL API CALLER
    # -------------------------------------------------------------------------
    def _call_endpoint(self, text: str) -> List[float]:
        """Send a request to the selected embedding provider and return the embedding vector."""
        if isinstance(text, list):
            text = text[0]
        
        text = text.strip()
        logger.info(f"[Embedding] Embedding text: {text} characters.")
        num_tokens = self.num_tokens_in_text(text)
        logger.info(f"[Embedding] Model={self.selected_embedding_model} \n Document length: {len(text)}/num-token: {num_tokens}: {text[:80]}...")
                    
        # logger.info(f"[Embedding] Model={self.selected_embedding_model}, Text={text[:80]}...")

        # CASE 1: Cohere multilingual via AWS Bedrock
        if self.selected_embedding_model == "cohere-multilingual":
            try:
                model_id = "cohere.embed-multilingual-v3"
                json_params = {
                    "texts": [text[:3000]],
                    "truncate": "END",
                    "input_type": "search_document",
                }
                json_body = json.dumps(json_params)
                result = self.bedrock_client.invoke_model(
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

        # CASE 2: Azure OpenAI text embedding
        elif self.selected_embedding_model == "text-embedding":
            logger.info("[Embedding] Calling Azure OpenAI embedding endpoint...")
            try:
                response = self.azure_client.embeddings.create(
                    model=self.azure_deployment_name,
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
        """Embed a list of documents."""
        embeddings = []
        logger.info(f"[Embedding] Embedding {len(texts)} documents...")
        
        for ind, text in enumerate(texts):
            logger.info(f"[Embedding] Document {ind+1}")
            vector = self._call_endpoint(text)
            emb_arr = np.array(vector, dtype=np.float32).reshape(1, -1)
            if self.use_cosine:
                faiss.normalize_L2(emb_arr)
            embeddings.append(emb_arr.ravel().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        emb = self._call_endpoint(text)
        emb_arr = np.array(emb, dtype=np.float32).reshape(1, -1)
        if self.use_cosine:
            faiss.normalize_L2(emb_arr)
        return emb_arr.ravel().tolist()
