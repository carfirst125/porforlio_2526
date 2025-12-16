from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import json
import pandas as pd
import asyncio
import aiofiles

from langchain_community.vectorstores import FAISS

from app.config.env_loader import logger
from app.agentic.agent_rag import AgenticRAG
from app.agentic.agent_creditcard import AgenticCreditCard
from app.utils.vectorstore_faiss import VectorstoreFaiss
from app.utils.custom_embedding import CustomEmbedding
from app.agentic.utils.agent_helper import AgentTypes
from app.agentic.agent_router import AgentRouter

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="VIB Credit Card & RAG API", version="1.0.0")

# ======================================================
# Start-up: Load vectorstore 1 lần duy nhất
# ======================================================

VECTORSTORE_PATH = "app/vectorstore/docs_index_cosine"
embedding_model = CustomEmbedding()

if not os.path.exists(VECTORSTORE_PATH + "/index.faiss"):
    logger.info("[startup] Vectorstore not found. Initializing...")
    df = pd.read_parquet("app/data/documents.parquet")
    logger.info(f"[startup] Loaded {len(df)} documents.")
    VectorstoreFaiss_obj = VectorstoreFaiss(vectorstore_path=VECTORSTORE_PATH)
    VectorstoreFaiss_obj.from_dataframe_with_embedding(df=df, input_col="input")

vectorstore = FAISS.load_local(
    folder_path=VECTORSTORE_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ======================================================
# Utility function: clean markdown
# ======================================================
def llm_response_process(text: str) -> str:
    import re
    pattern = r'\[.*?\]\((https?://[^\s)]+)\)'

    def replace_markdown_link(match):
        return match.group(1)

    new_text = re.sub(pattern, replace_markdown_link, text)
    new_text = new_text.replace('\n\n', '\n')
    return new_text


# ======================================================
# Request schema
# ======================================================
class QuestionRequest(BaseModel):
    userid: str = "UNKNOWN"
    question: str


@app.get("/")
def root():
    return {"message": "Agentic Chatbot API is OK"}

# ======================================================
# MAIN ENDPOINT — WITH FIX FOR TIMEOUT
# ======================================================
@app.post("/chat")
async def ask_question(payload: QuestionRequest):
    userid = payload.userid
    question = payload.question

    try:
        AgentRouter_obj = AgentRouter(userid=userid)
        label = AgentRouter_obj.meta_agent_router(question=question)
        logger.info(f"[router] label={label}")

        # ==================================================
        # RAG Agent
        # ==================================================
        if AgentTypes.PRODUCT_INFO in label:
            logger.info("[agent] RAG is working...")

            agent_rag = AgenticRAG(
                vectorstore=vectorstore,
                userid=userid,
                which_llm="azure_openai"
            )

            prompt_template = """
            Hãy sử dụng những tool được cung cấp để đưa ra câu trả lời phù hợp nhất cho khách hàng có userid là ${USERID}
            Lịch sử hội thoại của khách hàng: ${HISTORY}
            Câu hỏi hiện tại của khách hàng: ${QUESTION}

            Rules:
            1 - Chỉ xuất phần trả lời.
            2 - Trả lời bằng tiếng Việt.
            """

            agent_prompt = {
                "prompt_template": prompt_template,
                "QUESTION": question,
                "USERID": userid
            }

            # Prevent timeout — max 25 seconds
            response = await asyncio.wait_for(
                asyncio.to_thread(agent_rag.run, **agent_prompt),
                timeout=25
            )

        # ==================================================
        # Credit Card Agent
        # ==================================================
        elif AgentTypes.CREDIT_CARD_CONSULTANT in label:
            logger.info("[agent] CreditCard Consultant is working...")

            file_path = f"./credit_info_{userid}.json"

            # Safe async file IO
            if not os.path.exists(file_path):
                initial_data = {
                    "muc_dich": None,
                    "thu_nhap": None,
                    "han_muc": None,
                    "processing": 1
                }
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(initial_data, ensure_ascii=False, indent=4))

            agent_cc = AgenticCreditCard(vectorstore=vectorstore, userid=userid)

            prompt_template = """
            Bạn là tư vấn viên ngân hàng hỗ trợ khách hàng userid=${USERID}
            Lịch sử hội thoại: ${HISTORY}
            Câu hỏi hiện tại: ${QUESTION}

            Rules:
            1 - Chỉ xuất câu trả lời.
            2 - Trả lời bằng tiếng Việt.
            """

            agent_prompt = {
                "prompt_template": prompt_template,
                "QUESTION": question,
                "USERID": userid
            }

            response = await asyncio.wait_for(
                asyncio.to_thread(agent_cc.run, **agent_prompt),
                timeout=25
            )

        # ==================================================
        # Other questions
        # ==================================================
        else:
            logger.info("[agent] OTHERS is working...")
            _, response = AgentRouter_obj.question_others_response(question=question)

        # CLEAN
        response = llm_response_process(response)
        return {
            "userid": userid,
            "question": question,
            "answer": response
        }

    except asyncio.TimeoutError:
        logger.error("[timeout] LLM processing timeout > 25 seconds")
        return {
            "userid": userid,
            "question": question,
            "answer": "Xin lỗi, hệ thống đang bận xử lý quá lâu. Bạn vui lòng hỏi lại sau 1 chút nhé!"
        }

    except Exception as e:
        logger.error(f"[error] {e}")
        return {
            "userid": userid,
            "question": question,
            "answer": f"Hệ thống gặp lỗi nội bộ: {str(e)}"
        }
