from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import json
import pandas as pd

from langchain_community.vectorstores import FAISS

from app.config.env_loader import logger
from app.agentic.agent_rag import AgenticRAG
from app.agentic.agent_creditcard import AgenticCreditCard
from app.utils.vectorstore_faiss import VectorstoreFaiss
from app.utils.custom_embedding import CustomEmbedding #CustomCohereEmbedding
from app.agentic.utils.agent_helper import AgentTypes
from app.agentic.agent_router import AgentRouter

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="VIB Credit Card & RAG API", version="1.0.0")

######################################
# 1- check document vectorstore is existed or not
# 2- if not, create vectorstore from documents.parquet, else load existing vectorstore
######################################

# Load vectorstore global
VECTORSTORE_PATH = "app/vectorstore/docs_index_cosine"
embedding_model = CustomEmbedding()

# 1- check vectorstore
if not os.path.exists(VECTORSTORE_PATH + "/index.faiss"):
    logger.info(f"[startup] Vectorstore not found. Initializing...")
    df = pd.read_parquet("app/data/documents.parquet")
    logger.info(f"[startup] Loaded {len(df)} documents from parquet.")
    VectorstoreFaiss_obj = VectorstoreFaiss(vectorstore_path=VECTORSTORE_PATH)
    #VectorstoreFaiss_obj.from_dataframe(df=df, input_col="input", embedding_col="embedding")
    VectorstoreFaiss_obj.from_dataframe_with_embedding(df=df, input_col="input")

# 2- load vectorstore
vectorstore = FAISS.load_local(
    folder_path=VECTORSTORE_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

def llm_response_process(text: str) -> str:
    """Remove markdown links and clean newlines."""
    import re
    pattern = r'\[.*?\]\((https?://[^\s)]+)\)'

    def replace_markdown_link(match):
        return match.group(1)

    new_text = re.sub(pattern, replace_markdown_link, text)
    new_text = new_text.replace('\n\n', '\n')
    return new_text


class QuestionRequest(BaseModel):
    userid: str = "UNKNOWN"
    question: str

@app.get("/")
def root():
    return {"message": "Agentic Chatbot API is OK"}

@app.post("/chat")
def ask_question(payload: QuestionRequest):
    """
    Main endpoint to handle customer questions.
    It routes question to the right agent (RAG, CreditCard, Others).
    """
    userid = payload.userid
    question = payload.question
    file_path = f"./credit_info_{userid}.json"

    AgentRouter_obj = AgentRouter(userid=userid)
    label = AgentRouter_obj.meta_agent_router(question=question)
    logger.info(f"[router] label={label}")

    response = ""

    # Case 1: RAG Agent
    if AgentTypes.PRODUCT_INFO in label:
        agent_rag = AgenticRAG(vectorstore=vectorstore, userid=userid, which_llm="azure_openai")#"bedrock") #
        logger.info(f"[agent] RAG is working...")
        prompt_template = """Hãy sử dụng những tool được cung cấp để đưa ra câu trả lời phù hợp nhất cho khách hàng có userid là ${USERID}
        Lịch sử hội thoại của khách hàng:`
        ${HISTORY}
        
        Câu hỏi hiện tại của khách hàng:
        ${QUESTION}
        
        Rules: 
        1 - Xuất ra chỉ phần trả lời, không xuất phần diễn giải, lời dẫn hay lý luận, giải thích kết quả nhé!
        2 - Trả lời bằng tiếng việt, không trả lời bằng tiếng Anh.
        """
        agent_prompt = {
            "prompt_template": prompt_template,
            "QUESTION": question,
            "USERID": userid
        }
        response = agent_rag.run(**agent_prompt)

    # Case 2: Credit Card Agent
    elif AgentTypes.CREDIT_CARD_CONSULTANT in label:
        if not os.path.exists(file_path):
            initial_data = {
                "muc_dich": None,
                "thu_nhap": None,
                "han_muc": None,
                "processing": 1
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=4)
            logger.info(f"[file] Created {file_path} with default data.")

        agent_cdcard = AgenticCreditCard(vectorstore=vectorstore, userid=userid)
        logger.info(f"[agent] CreditCard Consultant is working...")
        prompt_template = """Bạn là một tư vấn viên ngân hàng, chuyên tư vấn về thẻ tín dụng. Bạn đang hỗ trợ khách hàng có userid là ${USERID}
        Lịch sử hội thoại của khách hàng:
        ${HISTORY}
        
        Câu hỏi/câu trả lời hiện tại của khách hàng: 
        ${QUESTION}
        
        Rules: 
        1 - Xuất ra chỉ phần trả lời, không xuất phần diễn giải, lời dẫn hay lý luận, giải thích kết quả nhé!
        2 - Trả lời bằng tiếng việt, không trả lời bằng tiếng Anh.
        """
        agent_prompt = {
            "prompt_template": prompt_template,
            "QUESTION": question,
            "USERID": userid
        }
        response = agent_cdcard.run(**agent_prompt)

    # Case 3: Others
    elif AgentTypes.OTHERS in label:
        logger.info(f"[agent] OTHERS is working...")
        _, response = AgentRouter_obj.question_others_response(question=question)

    response = llm_response_process(response)
    return {"userid": userid, "question": question, "answer": response}


# Run
# uvicorn app.chatbot:app --reload --port 8001
# uvicorn app.chatbot:app --host 0.0.0.0 --port 8001