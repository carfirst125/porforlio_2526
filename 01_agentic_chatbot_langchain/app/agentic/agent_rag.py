import os
import json
import re
import logging
from datetime import datetime
from app.agentic.tools.tools_rag import ToolsRAG
# from app.datastore.functionsstore import FmCcyRate, SavingCoreRate, FunctionsStore
from app.utils.clients import DBConnection, OpenAIGPT
from app.agentic.utils.agent_helper import AgentTypes, AgentHelper
from app.utils.logger import logger, UsageLoggerHandler

from typing import Any, Dict, List, Optional, Literal
#from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pdfminer.high_level import extract_text

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain.callbacks.base import BaseCallbackHandler
import warnings
warnings.filterwarnings("ignore")

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
# Build agent RAG

class AgenticRAG(ToolsRAG):
    """
    An agent-based conversational system powered by Azure OpenAI and LangChain.

    This class integrates Azure OpenAI (via LangChain) with a FAISS vectorstore and
    supports conversational memory. It enables context-aware question answering by
    using tool-augmented function-calling and persistent chat history for each user.
    Vietnamese strings in prompts and logic are preserved as required.
    """
    
    def __init__(self, vectorstore: FAISS, userid: str, which_llm: str = 'azure_openai'):
        """
        Initialize the AgenticLangOpenai agent.
        Args:
            llm: The language model instance.
            vectorstore (FAISS): The FAISS vectorstore for document retrieval.
            userid (str): Unique identifier for the user session.
        """
        super().__init__(vectorstore=vectorstore, which_llm=which_llm)
        self.vectorstore = vectorstore
        self.userid = userid
        self.chat_history = self._load_history_from_file()
        self.agent_executor = self._build_agent()
        # connection = self._get_db_connection()
        # self.functions_store = FunctionsStore(connection=connection)
        
    def _get_db_connection(self):
        """
        Build the database connection string from environment variables.
        Returns:
            str: SQLAlchemy connection string for PostgreSQL.
        """
        dbuser = DBConnection.agentic_db_dict.get('DBUSER')
        dbpass = DBConnection.agentic_db_dict.get('DBPASS')
        dbhost = DBConnection.agentic_db_dict.get('DBHOST')
        dbport = DBConnection.agentic_db_dict.get('DBPORT')
        dbname = DBConnection.agentic_db_dict.get('DBNAME')
        return f"postgresql+pg8000://{dbuser}:{dbpass}@{dbhost}:{dbport}/{dbname}"

    def _load_history_from_file(self):
        """
        Load the chat history for the given user from a local JSON file.
        Args:
            userid (str): The user identifier.
        Returns:
            List[Dict[str, str]]: List of chat history entries.
        """        
        path = f"chat_history_{self.userid}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history_to_file(self):
        """
        Save the current chat history for the user to a JSON file.
        """
        path = f"chat_history_{self.userid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)

    def _build_agent(self):
        """
        Initialize the LangChain agent with pre-configured tools and memory.
        Returns:
            agent_executor: The initialized LangChain agent executor.
        """
        params = {"step":"Agent-RAG"}
        callback_handler = UsageLoggerHandler(logger=logger, **params)
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
            # tools_dict.get('saving_interest_pipeline_tool', ""),
            # tools_dict.get('exchange_rate_pipeline_tool', "")
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
        logger.info(f"question: {kwargs.get('QUESTION', '')}")
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
        logger.info(f"agent response: {answer}")
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
agent = AgenticLangOpenai(vectorstore=my_vectorstore, userid="user123")
response = agent.run({'prompt_template':"your-prompt",
                    'QUESTION':'"Tôi muốn biết hạn mức thẻ tín dụng là gì?"})
print(response)
'''
'''
class AgenticRAG(ToolsRAG):
    """
    An agent-based conversational system powered by Azure OpenAI/Bedrock and LangChain.
    """

    def __init__(self, vectorstore: FAISS, userid: str, which_llm: str = 'azure_openai'):
        """
        Initialize the AgenticLangOpenai agent.
        Args:
            vectorstore (FAISS): The FAISS vectorstore for document retrieval.
            userid (str): Unique identifier for the user session.
            which_llm (str): 'azure_openai' | 'openai' | 'bedrock'
        """
        super().__init__(vectorstore=vectorstore, which_llm=which_llm)
        self.vectorstore = vectorstore
        self.userid = userid
        self.chat_history = self._load_history_from_file()
        self.agent_executor = self._build_agent()
        connection = self._get_db_connection()
        self.functions_store = FunctionsStore(connection=connection)

    def _get_db_connection(self):
        dbuser = DBConnection.agentic_db_dict.get('DBUSER')
        dbpass = DBConnection.agentic_db_dict.get('DBPASS')
        dbhost = DBConnection.agentic_db_dict.get('DBHOST')
        dbport = DBConnection.agentic_db_dict.get('DBPORT')
        dbname = DBConnection.agentic_db_dict.get('DBNAME')
        return f"postgresql+pg8000://{dbuser}:{dbpass}@{dbhost}:{dbport}/{dbname}"

    def _load_history_from_file(self):
        path = f"chat_history_{self.userid}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history_to_file(self):
        path = f"chat_history_{self.userid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)

    # --- Helper: bảo đảm self.llm là LangChain Runnable, không phải boto3 client ---
    # --- Helper: bảo đảm self.llm là LangChain Runnable, không phải boto3 client ---
    def _ensure_langchain_llm(self):
        """
        Với Bedrock: nếu self.llm là boto3 client thì bọc thành LangChain ChatBedrock.
        Với OpenAI/Azure: giữ nguyên.
        """
        which = getattr(self, "which_llm", "azure_openai")

        if which != "bedrock":
            return  # OpenAI/Azure: thường đã là ChatOpenAI/AzureChatOpenAI (Runnable)

        # Bedrock: tự detect nếu đang là boto3 client
        try:
            from botocore.client import BaseClient as _BotoBaseClient
            is_boto_client = isinstance(self.llm, _BotoBaseClient)
        except Exception:
            is_boto_client = False

        # Nếu đã là Runnable (có .invoke/.generate/.bind_tools), bỏ qua:
        if hasattr(self.llm, "invoke") or hasattr(self.llm, "generate"):
            return

        # Ưu tiên langchain_aws.ChatBedrock (mới). Fallback sang langchain_community.BedrockChat
        chat_bedrock_cls = None
        use_aws = False
        try:
            from langchain_aws import ChatBedrock  # bản mới
            chat_bedrock_cls = ChatBedrock
            use_aws = True
        except Exception:
            try:
                from langchain_community.chat_models import BedrockChat  # bản cũ
                chat_bedrock_cls = BedrockChat
                use_aws = False
            except Exception:
                raise RuntimeError(
                    "Không tìm thấy ChatBedrock/BedrockChat. "
                    "Cài gói `langchain-aws` hoặc update `langchain_community`."
                )

        # Đọc cấu hình từ ENV để linh hoạt
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        region = os.getenv("AWS_REGION", os.getenv("BEDROCK_REGION", "us-east-1"))

        if use_aws:
            # ChatBedrock (langchain-aws) → nhận temperature trực tiếp
            self.llm = chat_bedrock_cls(
                model_id=model_id,
                region_name=region,
                temperature=0
            )
        else:
            # BedrockChat (community) → chỉ cho phép model_kwargs
            self.llm = chat_bedrock_cls(
                model_id=model_id,
                model_kwargs={"temperature": 0}
            )


    def _build_agent(self):
        """
        Initialize the LangChain agent with pre-configured tools and memory.
        Returns:
            agent_executor: The initialized LangChain agent executor.
        """
        # 1) Gắn callback KHÔNG re-init từ __dict__ để tránh lỗi _serializer (Bedrock)
        params = {"step": "Agent-RAG"}
        callback_handler = UsageLoggerHandler(logger=logger, **params)

        # Đảm bảo self.llm là LangChain Runnable (đặc biệt với Bedrock)
        self._ensure_langchain_llm()

        # Gắn callback theo LCEL
        try:
            self.llm = self.llm.with_config(callbacks=[callback_handler])
        except Exception:
            # Fallback: một số wrapper cũ có thể không có with_config
            setattr(self.llm, "callbacks", [callback_handler])

        # 2) Chuẩn bị tools (lọc tool rỗng)
        tools_dict = self.tools_define()
        tools = [
            tools_dict.get('retriever_tool'),
            tools_dict.get('pdf_reader_tool'),
            tools_dict.get('saving_interest_pipeline_tool'),
            tools_dict.get('exchange_rate_pipeline_tool'),
        ]
        tools = [t for t in tools if t]

        # 3) Memory (Bedrock/React nên dùng chuỗi thay vì messages)
        which = getattr(self, "which_llm", "azure_openai")
        return_msgs = False if which == "bedrock" else True
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=return_msgs
        )
        for msg in self.chat_history:
            if "user" in msg:
                memory.chat_memory.add_user_message(msg["user"])
            if "ai" in msg:
                memory.chat_memory.add_ai_message(msg["ai"])

        # 4) Chọn agent type theo provider
        if which in ("openai", "azure_openai"):
            agent_type = AgentType.OPENAI_FUNCTIONS
        else:
            # Bedrock/Anthropic: dùng ReAct mô tả tool
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION

        # 5) Khởi tạo agent executor
        agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=agent_type,
            memory=memory,
            verbose=True,
            callbacks=[callback_handler],
        )
        return agent_executor

    def _format_history(self) -> str:
        lines = []
        for item in self.chat_history[-2:]:
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
            agent.run(prompt_template="Xin chào ${name}, đây là lịch sử: ${HISTORY}", name="Tuấn", QUESTION="...")
        """
        if "prompt_template" not in kwargs:
            raise ValueError("Thiếu 'prompt_template' trong kwargs")

        prompt_template = kwargs.pop("prompt_template")
        if "QUESTION" in kwargs:
            logger.info(f"question: {kwargs.get('QUESTION', '')}")

        placeholders = re.findall(r"\${(.*?)}", prompt_template)
        logger.info(f"placeholders: {placeholders}")

        for key in placeholders:
            if key == "HISTORY":
                history_text = self._format_history()
                history_text = (
                    "Đây là câu hỏi đầu tiên của Khách hàng, chưa có lịch sử hội thoại."
                    if history_text == "" else history_text
                )
                prompt_template = prompt_template.replace("${HISTORY}", history_text)
            elif key in kwargs:
                prompt_template = prompt_template.replace(f"${{{key}}}", str(kwargs[key]))
            else:
                raise ValueError(f"Thiếu giá trị cho biến '{key}' trong prompt_template")

        answer = self.agent_executor.run(prompt_template)
        logger.info(f"agent response: {answer}")
        answer = answer.replace("**", "")

        user_q = kwargs.get("QUESTION")
        if user_q is not None:
            self.chat_history.append({"user": user_q, "ai": answer})
            self._save_history_to_file()

        return answer
'''