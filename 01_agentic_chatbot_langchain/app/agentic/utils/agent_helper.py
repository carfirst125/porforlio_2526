import json
import os
from app.utils.clients import OpenAIGPT, BedrockLLM
from langchain_community.chat_models import AzureChatOpenAI


class AgentTypes:
    CREDIT_CARD_CONSULTANT = "CREDIT_CARD_CONSULTANT"
    PRODUCT_INFO = "PRODUCT_INFO"
    OTHERS = "OTHERS"

    # Có thể thêm các nhóm khác nếu cần
    DEFAULT_AGENT = "DEFAULT_AGENT"

class AgentHelper:
    def __init__(self, user_cdcard_info_path: str):
        self.user_cdcard_info_path = user_cdcard_info_path

    def check_processing(self):
        if not os.path.exists(self.user_cdcard_info_path):
            return None  # hoặc raise Exception nếu cần

        try:
            with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("processing") == 1:
                return AgentTypes.CREDIT_CARD_CONSULTANT
            else:
                return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None



class AgentLLM:    
    def __init__(self, which_llm: str = 'azure_openai'):
        self.which_llm = which_llm
        self.llm = self.get_llm() 

    def get_llm(self):
        if 'bedrock' in self.which_llm:
            #BedrockLLM_obj = BedrockLLM()   
            return BedrockLLM().llm
        elif 'openai' in self.which_llm:
            return AzureChatOpenAI(**OpenAIGPT.LLM_CONFIG)