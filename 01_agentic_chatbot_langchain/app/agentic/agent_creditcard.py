import os
import json
import re
import logging
from datetime import datetime

from app.agentic.tools.tools_creditcard import ToolsCreditCard
from app.datastore.functionsstore import FmCcyRate, SavingCoreRate, FunctionsStore
from app.utils.clients import DBConnection, OpenAIGPT
#from app.config.env_loader import logger 
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


###################################################
# Build agent

class AgenticCreditCard(ToolsCreditCard):
    """
    An agent-based conversational system powered by Azure OpenAI and LangChain.

    This class integrates Azure OpenAI (via LangChain) with a FAISS vectorstore and
    supports conversational memory. It enables context-aware question answering by
    using tool-augmented function-calling and persistent chat history for each user.
    Vietnamese strings in prompts and logic are preserved as required.
    """
    
    def __init__(self, vectorstore: FAISS, userid: str):
        """
        Initialize the AgenticLangOpenai agent.
        Args:
            llm: The language model instance.
            vectorstore (FAISS): The FAISS vectorstore for document retrieval.
            userid (str): Unique identifier for the user session.
        """
        super().__init__(knowledge_base=vectorstore, userid=userid)
        self.vectorstore = vectorstore
        #self.userid = userid
        self.chat_history = self._load_history_from_file()
        self.agent_executor = self._build_agent()
        # connection = self._get_db_connection()
        # self.functions_store = FunctionsStore(connection=connection)


    # def _get_db_connection(self):
    #     """
    #     Build the database connection string from environment variables.
    #     Returns:
    #         str: SQLAlchemy connection string for PostgreSQL.
    #     """
    #     dbuser = DBConnection.agentic_db_dict.get('DBUSER')
    #     dbpass = DBConnection.agentic_db_dict.get('DBPASS')
    #     dbhost = DBConnection.agentic_db_dict.get('DBHOST')
    #     dbport = DBConnection.agentic_db_dict.get('DBPORT')
    #     dbname = DBConnection.agentic_db_dict.get('DBNAME')
    #     return f"postgresql+pg8000://{dbuser}:{dbpass}@{dbhost}:{dbport}/{dbname}"

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
        params = {"step":"Agent-CREDITCARD"}
        callback_handler = UsageLoggerHandler(logger=logger, **params)
        llm_dict = self.llm.__dict__.copy()
        llm_dict.pop('callbacks', None)  # Remove 'callbacks' key if it exists

        self.llm = self.llm.__class__(  # Reinitialize the LLM with updated dictionary
            **llm_dict,
            callbacks=[callback_handler]
        )
                
        tool_cdcard_dict = self.tools_define()
        
        # Compose the list of tools for the agent
        tools = [
            tool_cdcard_dict.get('do_cdcard_consult', "")
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
agent = AgenticLangOpenai(vectorstore=my_vectorstore, userid="user123")
response = agent.run({'prompt_template':"your-prompt",
                    'QUESTION':'"Tôi muốn biết hạn mức thẻ tín dụng là gì?"})
print(response)
'''