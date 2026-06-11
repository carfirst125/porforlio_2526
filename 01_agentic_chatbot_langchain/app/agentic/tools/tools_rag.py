import os
import json
import re
import logging
from datetime import datetime

from typing import Any, Dict, List, Optional, Literal
from langchain_community.chat_models import AzureChatOpenAI
#from langchain_openai import AzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import FAISS
from pdfminer.high_level import extract_text
from app.config.env_loader import logger  # Load .env một lần duy nhất
from app.utils.clients import OpenAIGPT, BedrockLLM
from app.agentic.utils.agent_helper import AgentLLM

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


import warnings
warnings.filterwarnings("ignore")

# Schema để extract
class SavingQuery(BaseModel):
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


class ToolsRAG(AgentLLM):
    def __init__(self, vectorstore: FAISS, which_llm: str = 'azure_openai'):
        super().__init__(which_llm=which_llm) 
        self.vectorstore=vectorstore

    # def get_llm(self):
    #     if self.which_llm == 'bedrock':
    #         return BedrockLLM.client
    #     elif self.which_llm == 'azure_openai':
    #         return AzureChatOpenAI(**OpenAIGPT.LLM_CONFIG)        
    
    # #----------------------------------
    # # EXCHANGE
    # # Tool: Extract info from customer query
    # def extract_exchange_info_func(self, user_query: str) -> dict:
    #     parser = PydanticOutputParser(pydantic_object=ExchangeQuery)
    #     prompt = ChatPromptTemplate.from_template(
    #         """Trích xuất thông tin từ câu hỏi về đổi tiền tệ/ tỷ giá tiền tệ.
    #         Câu hỏi: {question}
    #         {format_instructions}"""
    #     )
    #     # print('A:',parser.get_format_instructions())
    #     chain = prompt | self.llm | parser
    #     result = chain.invoke({"question": user_query, "format_instructions": parser.get_format_instructions()})
    #     # print('B:', result.dict())
    #     return result.dict()
    
    # def exchange_rate_pipeline_func(self, query: str) -> str:
    #     # 1. Dùng LLM extract params
    #     extracted = self.extract_exchange_info_func(query)
    #     from_ccy = extracted['from_ccy']
    #     to_ccy = extracted['to_ccy']                  
    #     rate_type = extracted['rate_type'] 
    #     #effective_date = extracted['effective_date']
        
    #     effective_date = datetime.now().date().strftime("%Y-%m-%d")
    #     exchange_data = {}              
    #     exchange_data['exchange_rate'] = self.functions_store.get_exchange_rate(from_ccy, to_ccy, rate_type) #, effective_date)
    #     print(f"saving_data['exchange_rate']: {len(exchange_data['exchange_rate'])}")            
    #     if len(exchange_data['exchange_rate']) > 100:
    #         exchange_data['exchange_rate']  = exchange_data['exchange_rate'][:100]
    #     exchange_data['exchange_amount'] = extracted['exchange_amount']
        
    #     return exchange_data    
    
    # #----------------------------------
    # # SAVING/DEPOSIT
    # # Tool: Extract info from customer query
    # def extract_saving_info_func(self, user_query: str) -> dict:
    #     parser = PydanticOutputParser(pydantic_object=SavingQuery)
    #     prompt = ChatPromptTemplate.from_template(
    #         """Trích xuất thông tin từ câu hỏi gửi tiết kiệm.
    #         Câu hỏi: {question}
    #         {format_instructions}"""
    #     )
    #     #print('A:',parser.get_format_instructions())
    #     chain = prompt | self.llm | parser
    #     result = chain.invoke({"question": user_query, "format_instructions": parser.get_format_instructions()})
    #     #print('B:', result.dict())
    #     return result.dict()    
    

    # #----------------------------------
    # # Saving
    # # Tool: Trả lời câu hỏi về tiết kiệm [Flow]
    
    # # Đây là flow nhiều bước cho 1 process, đóng gói nó vào 1 hàm
    # # ví dụ: extract params --> get data --> answer composing
    # def saving_interest_pipeline_func(self, query: str) -> str:
    #     # 1. Dùng LLM extract params
    #     extracted = self.extract_saving_info_func(query)
    #     product_type = extracted['product_type']
    #     saving_type = extracted['saving_type']                  
    #     client_type = extracted['client_type'] 
    #     term_type = extracted['saving_term']
    #     ccy = extracted['saving_currency']
        
    #     saving_data = {}              
    #     saving_data['interest_rate'] = self.functions_store.get_savings_rate(product_type,saving_type,client_type,term_type,ccy)
    #     print(f"saving_data['interest_rate']: {len(saving_data['interest_rate'])}")            
    #     if len(saving_data['interest_rate']) > 500:
    #         saving_data['interest_rate']  = saving_data['interest_rate'][:200]
    #     saving_data['saving_amount'] = extracted['saving_amount']
        
    #     return saving_data

    #----------------------------------
    # Tool 02
    def calculator_saving_profit_func(self, expr: str) -> str:
        try:
            return str(eval(expr))
        except:
            return "Lỗi tính toán."
        
    #----------------------------------
    # Tool 03
    def translate(self, text: str) -> str:
        return f"Translated: {text} → English"

    #----------------------------------
    # Tool 05
    def retrieve_context(self, query: str, k: int = 5) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Nội dung tham khảo từ tài liệu:\n{context}"

    #----------------------------------
    # Tool 06
    def file_reader(self, file_path: str) -> str:
        try:
            text = extract_text(file_path)
            return text.strip() if text else "Không đọc được nội dung từ PDF."
        except Exception as e:
            return f"Lỗi khi đọc file PDF: {e}"
            
    ########################################        
    def tools_define(self):
        # extract_exchange_info_tool = Tool.from_function(
        #     func=self.extract_exchange_info_func,
        #     name="extract_exchange_info",
        #     description="Trích xuất exchange_type, rate_type, exchange_date, from_ccy, to_ccy từ câu hỏi của khách hàng."
        # )        
        
        # exchange_rate_pipeline_tool = Tool.from_function(
        #     func=self.exchange_rate_pipeline_func,
        #     name="exchange_rate_pipeline",
        #     description="""Xử lý câu hỏi về tỷ giá đổi ngoại tệ, tỷ giá chuyển đổi tiền tệ. Đọc hiểu câu hỏi thuộc loại nào trong 2 loại sau và trả lời cho chính xác: 
        #     1. Câu hỏi về số tiền nhận được khi chuyển đổi một lương tiền tệ cụ thể (from_ccy) sang một đồng tiền tệ khác (to_ccy). Với câu hỏi này bạn cần tìm ra tỷ giá chuyển đổi giữa 2 đồng tiền muốn đổi dựa trên dữ liệu về tỷ giá được cung cấp. Nếu là tỷ giá chuyển đổi giữa 1 đồng tiền khác với VND thì đơn giản lấy trực tiếp trong dữ liệu. Nhưng nếu tỷ giá chuyển đổi là giữa 2 đồng tiền không có VND thì bạn phải tính tỷ giá chuyển đổi giữa 2 đồng tiền thông qua trung gian VND, tức là lấy tỷ giá chuyển đổi của from_ccy với VND chia cho tỷ giá chuyển đổi của to_ccy với VND sẽ ra tỷ giá chuyển đổi của from_ccy so với to_ccy. Sau đó, khi có tỷ giá chuyển đổi, bạn tính toán số tiền có thể đổi được theo công thức: (số tiền gởi * tỷ giá chuyển đổi giữa đồng tiền cần đổi so với đồng tiền muốn đổi)
        #     Chú ý, câu trả lời là một câu tổng hợp với yêu cầu là gì và kết quả là gì, ko chỉ trả lời kết quả only.
        #     ví dụ 1: H: đổi 100 USD được bao nhiêu VND
        #             A: tỷ giá chuyển đổi USD sang VND là 1 USD = 25000 VND, số tiền đổi được của 100 USD = 25 triệu VND.
        #     2. Câu hỏi về tỷ giá chuyển đổi huyển đổi giữa 2 đồng tiền tệ, tỷ giá chuyển đổi từ đồng tiền này (from_ccy) sang một đồng tiền tệ khác (to_ccy). Nếu câu hỏi không có from_ccy hay to_ccy thì mắc định giá trị của trường thiếu đó là VND.
        #     Xuất ra câu trả lời là một câu hoàn chỉnh ví dụ như sau:
        #     Ví dụ 2: H: tỷ giá USD đang bao nhiêu.
        #             A: tỷ giá USD hiện đang là 25000 VND"""
        # )        
           
        # extract_saving_info_tool = Tool.from_function(
        #     func=self.extract_saving_info_func,
        #     name="extract_saving_info",
        #     description="Trích xuất saving term, saving amount, saving currency, saving type, client type, product type từ câu hỏi của khách hàng."
        # )
        
        # saving_interest_pipeline_tool = Tool.from_function(
        #     func=self.saving_interest_pipeline_func,
        #     name="saving_interest_pipeline",
        #     description="""Xử lý câu hỏi về lãi gửi tiết kiệm. Đọc hiểu câu hỏi thuộc loại nào trong 2 loại sau và trả lời cho chính xác: 
        #     1. Câu hỏi về số tiền lời nhận được của một số tiền gởi cụ thể (saving amount) ở kỳ hạn N tuần/tháng (saving term). Với câu hỏi này bạn cần tìm ra lãi suất của tiền gởi ứng với kỳ hạn gởi (saving interest rate), sau đó tính toán số tiền lời theo công thức: (số tiền gởi * lãi suất tiền gởi theo năm / 12)* số tháng gởi.
        #     Chú ý, câu trả lời là một câu tổng hợp với yêu cầu là gì và kết quả là gì, ko chỉ trả lời kết quả only.
        #     ví dụ 1: H: gởi 100 triệu 3 tháng lãi bao nhiêu?
        #             A: gởi tiết kiệm kỳ hạn 3 tháng thì lãi suất là 5%/năm và lợi tức thu được là 1.25 triệu.
        #     2. Câu hỏi về lãi suất tiền gởi (saving interest rate) hoặc lãi suất tiền gởi cho 1 kỳ hạn gởi nào đó. Trích suất trong data tìm ra lãi suất tiền gởi tương ứng với kỳ hạn yêu cầu trong câu hỏi.
        #     Xuất ra câu trả lời là một câu hoàn chỉnh ví dụ như sau:
        #     Ví dụ 2: H: gởi tiết kiệm kỳ hạn 6 tháng lãi bao nhiêu?
        #             A: gởi tiết kiệm kỳ hạn 6 tháng thì lãi suất là 5.1%/năm"""
        # )
        
        calculator_saving_month_profit_tool = Tool.from_function(
            func=self.calculator_saving_profit_func,
            name="calculator_saving_profit",
            description="tính toán lãi tiền gởi tiết kiệm dựa trên lợi suất và tiền gởi. Theo đó tổng tiền lãi gởi tiết kiệm hàng tháng sẽ bằng (lợi suất * tiền gởi)/12"
        )
    
        translator_tool = Tool.from_function(
            func=self.translate,
            name="translator",
            description="Dịch văn bản từ tiếng Việt sang tiếng Anh"
        )
    
        retriever_tool = Tool.from_function(
            func=self.retrieve_context,
            name="retriever",
            description="Tìm kiếm thông tin liên quan từ tài liệu nội bộ để hỗ trợ trả lời câu hỏi về sản phẩm, dịch vụ của ngân hàng VIB."
        )

        pdf_reader_tool = Tool.from_function(
            func=self.file_reader,
            name="pdf_reader",
            description="Đọc nội dung file PDF. VD: './data/file.pdf'"
        )

        return {#'extract_exchange_info_tool'      : extract_exchange_info_tool,
                #'exchange_rate_pipeline_tool'   : exchange_rate_pipeline_tool,
                #'extract_saving_info_tool'      : extract_saving_info_tool,
                #'saving_interest_pipeline_tool' : saving_interest_pipeline_tool,
                'calculator_saving_month_profit_tool': calculator_saving_month_profit_tool,
                'retriever_tool'                : retriever_tool,
                'pdf_reader_tool'               : pdf_reader_tool}
    
