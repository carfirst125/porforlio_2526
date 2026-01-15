import os
import json
import re
import logging
from datetime import datetime
from app.agentic.tools.tools_rag import ToolsRAG
#from app.agentic.tools.tools_creditcard import AgenticToolsCreditCard

from typing import Any, Dict, List, Optional, Literal
#from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pdfminer.high_level import extract_text
from app.config.env_loader import logger  # Load .env một lần duy nhất

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from app.datastore.functionsstore import FmCcyRate, SavingCoreRate, FunctionsStore
from app.utils.clients import DBConnection, OpenAIGPT

from langchain.callbacks.base import BaseCallbackHandler


# Schema để extract
class SavingQuery(BaseModel):
    """
    Pydantic schema for extracting saving deposit information from user input.
    Vietnamese abbreviations in string values are kept as-is.
    """
    saving_term: str = Field(
        description="Term of deposit. If there is no term length, default is 1M (1 month). Separated by commas, rounds down to total weeks, months, or years. For examples: 2W (meaning: 2 weeks), 3M (meaning: 3 months), 1Y (meaning: 1 year)."
    )
    saving_amount: Optional[float] = Field(
        description="Balance of deposit. If there is no amount, omit this. With abbeviation in Vietnamese: for example: 10tr (meaning: 10000000), 500tr (meaning: 500000000), 1tỷ (meaning: 1000000000)"
    )
    saving_currency: str = Field(
        description="Currency type of deposit. Default is VND"
    )
    saving_type: str = Field(
        description="Method(s) of deposit, separated by commas: online/trực tuyến (ONLINE), at counter/tại quầy (COUNTER). Default value is ONLINE "
    )
    client_type: str = Field(
        description="Customer type: individual/cá nhân (CN), or business/doanh nghiệp (DN). Default is CN "
    )
    product_type: str = Field(
        description="Product type of the deposit: automatic/tự động (401), installment/gửi góp (502), targeted/mục tiêu (506), online/trực tuyến (524), upfront interest/lãi đầu kỳ (526), idepo (528), term/kỳ hạn (TDE), flexible/linh hoạt (FFD), or foreign currency/ngoại tệ (TMS). Notice: If method of deposit is online, the product_type MUST be 524 (prioritized)."
    )

class ExchangeQuery(BaseModel):
    """
    Pydantic schema for extracting currency exchange information from user input.
    Vietnamese abbreviations in string values are kept as-is.
    """
    exchange_type: Literal["SELL", "BUY"] = Field(
        description="Type of exchange: sell or buy. Default: BUY"
    )
    rate_type: Literal["CA", "TRS"] = Field(
        description="Type of rate: cash (CA), or transfer (TRS). Default: CA"
    )
    exchange_date: Optional[str] = Field(
        description="Date of exchange"
    )
    from_ccy: Literal["USD", "AUD", "CAD", "EUR", "GBP", "JPY", "SGD", "CHF", "DKK", "HKD", "NOK"] = Field(
        description="Source currency type"
    )
    to_ccy: Literal["VND", "USD", "AUD", "CAD", "EUR", "GBP", "JPY", "SGD", "CHF", "DKK", "HKD", "NOK"] = Field(
        description="Destination currency type"
    )
    exchange_amount: Optional[float] = Field(
        description="Amount which customer would like to exchange. If there is no amount, omit this. With abbeviation in Vietnamese: for example: 10tr (meaning: 10000000), 500tr (meaning: 500000000), 1tỷ (meaning: 1000000000)"
    )


###################################################
# Logging

def get_logger(name="agent_usage"):
    """
    Create and return a logger for agent usage events. Ensures no duplicate handlers.
    """
    log_file = os.path.join(os.getcwd(), "agent_usage.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.info("Logger started")
    return logger


class UsageLoggerHandler(BaseCallbackHandler):
    """
    Callback handler for logging agent and LLM usage events.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        """
        Log usage statistics when the LLM finishes responding.
        """
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.logger.info(json.dumps({
                "event": "usage",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens
            }, ensure_ascii=False))

    def on_agent_action(self, action, **kwargs):
        """
        Log each step the agent takes.
        """
        self.logger.info(json.dumps({
            "event": "agent_action",
            "tool": action.tool,
            "tool_input": str(action.tool_input),
            "log": action.log
        }, ensure_ascii=False))

    def on_tool_end(self, output, **kwargs):
        """
        Log the result returned from a tool.
        """
        self.logger.info(json.dumps({
            "event": "tool_end",
            "output": str(output)
        }, ensure_ascii=False))

    def on_chain_end(self, outputs, **kwargs):
        """
        Log when a chain finishes.
        """
        self.logger.info(json.dumps({
            "event": "chain_end",
            "outputs": outputs
        }, ensure_ascii=False))

logger = get_logger()

###################################################
# Build agent RAG

class AgenticOthers(ToolsOthers):

    
    def __init__(self, user_id: str):
        super().__init__(vectorstore=vectorstore)       
        self.user_id = user_id
        self.chat_history = self._load_history_from_file(user_id)
        self.agent_executor = self._build_agent()
        
    def _load_history_from_file(self):
        """
        Load the chat history for the given user from a local JSON file.
        Args:
            user_id (str): The user identifier.
        Returns:
            List[Dict[str, str]]: List of chat history entries.
        """        
        path = f"chat_history_{self.user_id}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history_to_file(self):
        """
        Save the current chat history for the user to a JSON file.
        """
        path = f"chat_history_{self.user_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)

    def _build_agent(self):
        """
        Initialize the LangChain agent with pre-configured tools and memory.
        Returns:
            agent_executor: The initialized LangChain agent executor.
        """
        callback_handler = UsageLoggerHandler(logger=logger)
        llm_dict = self.llm.__dict__.copy()
        llm_dict.pop('callbacks', None)  # Remove 'callbacks' key if it exists

        self.llm = self.llm.__class__(  # Reinitialize the LLM with updated dictionary
            **llm_dict,
            callbacks=[callback_handler]
        )
        
        tools_dict = self.tools_define()
        #tool_cdcard_dict = self.AgenticToolsCreditCard.tools_define()
        
        # Compose the list of tools for the agent
        tools = [
            tools_dict.get('retriever_tool', ""),
            tools_dict.get('pdf_reader_tool', ""),
            tools_dict.get('saving_interest_pipeline_tool', ""),
            tools_dict.get('exchange_rate_pipeline_tool', "")
        ]
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        for msg in self.chat_history:
            memory.chat_memory.add_user_message(msg["user"])
            memory.chat_memory.add_ai_message(msg["ai"])
            
        # Create the agent executor
        agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            callbacks=[callback_handler]
        )
        return agent_executor

    def _format_history(self) -> str:
        """
        Convert the latest Q&A pairs into a string format for prompt injection.
        Only the last 2 pairs are included.
        Returns:
            str: Formatted chat history string.
        """
        lines = []
        for item in self.chat_history[-2:]:  # Only take up to the last 2 pairs
            question = item.get("user", "").strip()
            answer = item.get("ai", "").strip()
            lines.append(f"H: {question}\nA: {answer}")
        return "\n".join(lines)

    def run(self, **kwargs) -> str:
        """
        Run inference with a prompt template and dynamic variables.
        Args:
            kwargs: Must contain 'prompt_template' and variables for placeholders in the template.
        Returns:
            str: The agent's answer.
        Example:
            agent.run(prompt_template="Xin chào {name}, đây là lịch sử: {HISTORY}", name="Tuấn")
        """
        if "prompt_template" not in kwargs:
            raise ValueError("Thiếu 'prompt_template' trong kwargs")
            
        prompt_template = kwargs.pop("prompt_template")
        # Find placeholders in the template
        placeholders = re.findall(r"\${(.*?)}", prompt_template)
        logger.info(f"placeholders: {placeholders}")
        
        for key in placeholders:
            if key == "HISTORY":
                history_text = self._format_history()
                history_text = "Đây là câu hỏi đầu tiên của Khách hàng, chưa có lịch sử hội thoại." if history_text == "" else history_text
                prompt_template = prompt_template.replace("${HISTORY}", history_text)
            elif key in kwargs:
                value = str(kwargs[key])
                prompt_template = prompt_template.replace(f"${{{key}}}", value)
            else:
                raise ValueError(f"Thiếu giá trị cho biến '{key}' trong prompt_template")
        # Call the agent executor
        answer = self.agent_executor.run(prompt_template)
        answer = answer.replace("**", "")
        # Update and save chat history
        self.chat_history.append({"user": kwargs['QUESTION'], "ai": answer})
        self._save_history_to_file()
        return answer


''' USERGUIDE
# Giả sử đã có vectorstore
from langchain_community.vectorstores import FAISS

# Load vectorstore đã lưu
embedding_model = CustomCohereEmbedding()
vectorstore = FAISS.load_local(
    folder_path="vectorstores/cohere_index_cosine",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Khởi tạo AgenticLangOpenai
agent = AgenticLangOpenai(vectorstore=my_vectorstore, user_id="user123")
response = agent.run({'prompt_template':"your-prompt",
                    'QUESTION':'"Tôi muốn biết hạn mức thẻ tín dụng là gì?"})
print(response)
'''