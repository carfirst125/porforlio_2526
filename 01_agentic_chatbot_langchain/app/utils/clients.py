import os
import boto3
import logging
import json
from botocore.exceptions import ClientError
from langchain_community.chat_models import AzureChatOpenAI
#from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import BedrockChat

import sys

logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn cần chi tiết hơn
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # In ra terminal
)
logger = logging.getLogger(__name__)

class DBConnection:
    src_db_dict = {
        'DBHOST': os.getenv("DBHOST_DP"),
        'DBPORT': os.getenv("DBPORT_DP"),
        'DBNAME': os.getenv("DBNAME_DP"),
        'DBUSER': os.getenv("DBUSER_DP"),
        'DBPASS': os.getenv("DBPASS_DP")
    }

    agentic_db_dict = {
        'DBHOST': os.getenv("DBHOST_DPN"),
        'DBPORT': os.getenv("DBPORT_DPN"),
        'DBNAME': os.getenv("DBNAME_DPN"),
        'DBUSER': os.getenv("DBUSER_DPN"),
        'DBPASS': os.getenv("DBPASS_DPN")
    }

class OpenAIGPT:
    
    # LLM_CONFIG = {
    #     "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    #     "deployment_name": "intbot-prod-gpt-4o",
    #     "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    #     "temperature": 0
    # }  

    LLM_CONFIG = {
        "azure_endpoint": os.getenv("AZ_OPENAI_LLM_API_ENDPOINT"),
        "openai_api_version": os.getenv("AZ_OPENAI_LLM_API_VERSION"),
        "deployment_name": os.getenv('AZ_OPENAI_LLM_API_DEPLOYMENT_NAME'),
        "openai_api_key": os.getenv("AZ_OPENAI_LLM_API_SUBSCRIPTION_KEY"),
        "temperature": 0
    } 
        
    # llm = AzureChatOpenAI(
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     deployment_name="intbot-prod-gpt-4o",
    #     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     temperature=0
    # )  

'''
class OpenAIGPT:
    LLM_CONFIG = {
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment_name": "intbot-prod-gpt-4o",
        "model": "gpt-4o",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
'''

class BedrockLLM:
    COHERE_MULTI_LANGUAGE = 'cohere.embed-multilingual-v3'
    HAIKU_35 = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-haiku-20241022-v1:0')
    '''
    def __init__(self, **kwargs):  
        # Cho phép nhận bất kỳ tham số nào nhưng chỉ dùng cái cần
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
            aws_session_token=os.getenv('AWS_SSN_TOKEN'),
            region_name="ap-southeast-1",
            verify=False   # ⚠️ chỉ dùng dev, test
        )
        self.model_id = kwargs.get("model_id", self.HAIKU_35)

        self.llm = BedrockChat(
            client=self.bedrock_client,
            model_id=self.model_id
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt) 
    '''
    def get_embedding_vector(self, text, engine: str = 'cohere.embed-multilingual-v3'):
        """get embedding vector by cohere
        
        Args:
            text (str): message or text string that you would like to get embedding vector.
            [engine, genai_type]: [COHERE_MULTI_LANGUAGE, 'cohere']
            
        Returns:
            vector: vector array
        """
        try:                
            model_id = engine
            json_params = {
                'texts': [text[:2048]], #[text[:2048]]
                'truncate': "END", 
                'input_type': "search_document"
            }
            
            json_body = json.dumps(json_params)
            params = {'body': json_body, 'modelId': model_id}
            result = self.bedrock_client.invoke_model(**params)
            response = json.loads(result['body'].read().decode())
        
        
        except ClientError as e:
            logger.info(f"ClientError: {e}")
            error_code = e.response['Error']['Code']
            if error_code == 'ExpiredTokenException':
                logger.info("❌ Token đã hết hạn. Vui lòng làm mới AWS credentials.")
                sys.exit(1)
            else:
                logger.info(f"⚠️ Lỗi khác: {error_code} - {e.response['Error']['Message']}")
            
        return response["embeddings"][0]
