import os
import json
import re
import logging
from datetime import datetime

from typing import Any, Dict, List, Optional, Literal
from langchain_community.chat_models import AzureChatOpenAI
# from langchain_openai import AzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import FAISS
from pdfminer.high_level import extract_text
from app.config.env_loader import logger  # Load .env một lần duy nhất
from app.utils.clients import OpenAIGPT

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage

from langchain.schema import HumanMessage
import warnings
warnings.filterwarnings("ignore")

class RequiredCreditInfo(BaseModel):
    """
    Pydantic model representing the required information for credit card advisory.

    Attributes:
        muc_dich (Optional[str]): 
            The intended purpose of using the credit card, such as online shopping,
            supermarket/restaurant payments, flight bookings, reward accumulation, travel, etc.
        thu_nhap (Optional[str]): 
            The customer's monthly income, e.g., "20 million VND", "30tr", "500 USD".
        han_muc (Optional[str]): 
            The desired credit card limit, which reflects spending needs.
            Examples: "50 million VND", "100tr", "2000 USD".
    """
    muc_dich: Optional[str] = Field(
        description=(
            "Mục đích sử dụng thẻ tín dụng hoặc lý do thường dùng thẻ tín dụng "
            "(ví dụ: mua sắm online, thanh toán tại siêu thị/nhà hàng, "
            "thanh toán vé máy bay, tích điểm, du lịch, v.v.)."
        )
    )
    thu_nhap: Optional[str] = Field(
        description="Thu nhập hàng tháng của khách hàng, ví dụ: '20 triệu', '30tr', '500 USD'."
    )
    han_muc: Optional[str] = Field(
        description=(
            "Hạn mức thẻ mong muốn, phản ánh nhu cầu chi tiêu cao hay thấp. "
            "Ví dụ: '50 triệu', '100tr', '2000 USD'."
        )
    )


class CreditCardFields:
    """
    A class to manage credit card information fields during the advisory flow.

    Attributes:
        muc_dich (str): The purpose of using the credit card.
        thu_nhap (str): The customer's monthly income.
        han_muc (str): The desired credit card limit.
        processing (int): Processing status flag (1: collecting info, 0: not collecting).
    """
    def __init__(self):
        """Initialize credit card fields with None and set processing to 0."""
        self.muc_dich = None
        self.thu_nhap = None
        self.han_muc = None
        self.processing = 0

    def to_dict(self):
        """
        Convert the credit card fields into a dictionary.

        Returns:
            dict: Dictionary containing all field values.
        """
        return {
            "muc_dich": self.muc_dich,
            "thu_nhap": self.thu_nhap,
            "han_muc": self.han_muc,
            "processing": self.processing
        }

    def update_field(self, field, value):
        """
        Update a specific field with a given value.

        Args:
            field (str): The field name to update.
            value (Any): The new value for the field.
        """
        if hasattr(self, field):
            setattr(self, field, value)


class ToolsCreditCard:
    """
    Main class to handle credit card advisory workflow including information extraction,
    updating customer info, asking questions, and providing card recommendations.

    Attributes:
        knowledge_base: VectorStore retriever or FAISS instance used for similarity search.
        llm: Large Language Model instance (AzureChatOpenAI).
        userid (str): Identifier for the customer.
        state (CreditCardFields): Stores the current state of collected customer info.
        user_cdcard_info_path (str): File path to store customer credit info in JSON format.
    """
    def __init__(self, knowledge_base, userid):
        """
        Initialize ToolsCreditCard with knowledge base and customer ID.

        Args:
            knowledge_base: VectorStore retriever or FAISS instance.
            userid (str): Customer identifier.
        """
        
        self.llm = AzureChatOpenAI(**OpenAIGPT.LLM_CONFIG)
        self.knowledge_base = knowledge_base       
        self.userid = userid
        self.state = CreditCardFields()
        self.user_cdcard_info_path = f"credit_info_{self.userid}.json"
        
    
    def extract_credit_info_func(self, customer_answer: str) -> dict:
        """
        Extract credit card information from the customer's answer using LLM.

        Args:
            customer_answer (str): The customer's answer in free text.

        Returns:
            dict: Extracted fields {"muc_dich": ..., "thu_nhap": ..., "han_muc": ...}.
        """
        parser = PydanticOutputParser(pydantic_object=RequiredCreditInfo)
        prompt = ChatPromptTemplate.from_template(
            """
            Trích xuất các thông tin cần thiết để tư vấn mở thẻ tín dụng từ câu trả lời của khách hàng.
            Nếu khách hàng không đề cập một thông tin nào đó, để giá trị là null.

            Câu trả lời của khách hàng: {answer}

            {format_instructions}
            """
        )

        chain = prompt | self.llm | parser
        result = chain.invoke({
            "answer": customer_answer,
            "format_instructions": parser.get_format_instructions()
        })
        return result.dict()
            
    
    def next_question(self) -> Optional[str]:
        """
        Determine the next question to ask the customer if required data is missing.

        Returns:
            Optional[str]: Next question string, or None if all data is collected.
        """
        for field, value in self.state.to_dict().items():
            if value is None:
                return self.get_question_for_field(field)
        return None

    def get_question_for_field(self, field):
        """
        Get a predefined question for a specific field.

        Args:
            field (str): The field name (muc_dich, thu_nhap, han_muc).

        Returns:
            str: Question string for the field.
        """
        questions = {
            "muc_dich": "Bạn muốn mở thẻ tín dụng để tiêu dùng cho những mục đích hay sản phẩm nào? (mua sắm, du lịch, tích điểm...)",
            "thu_nhap": "Thu nhập hàng tháng của bạn khoảng bao nhiêu?",
            "han_muc": "Bạn muốn hạn mức thẻ bao nhiêu?"
        }
        return questions.get(field, "")

    def update_customer_info(self) -> str:
        """
        Update stored customer information by reading and writing the JSON file.

        Returns:
            str: Confirmation message or error message.
        """
        try:
            # Read data from the provided JSON file
            with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update state with the data from the file
            for field, value in data.items():
                self.state.update_field(field, value)

            # Save updated state back to the same file
            with open(self.user_cdcard_info_path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=4)

            return "Đã lưu thông tin khách hàng thành công."
        except Exception as e:
            return f"Lỗi khi cập nhật thông tin: {e}"

    def recommend_credit_card(self, _: str) -> str:
        """
        Recommend the most suitable credit card based on collected customer information.

        Returns:
            str: Recommendation text including the card name and highlights.
        """
        

        def _extract_json_from_response(response):
            # Nếu là AIMessage
            if isinstance(response, AIMessage):
                text = response.content
            # Nếu là dict (có key 'content')
            elif isinstance(response, dict) and "content" in response:
                text = response["content"]
            else:
                text = str(response)

            # Bỏ ```json ... ```
            clean_text = re.sub(r"```json|```", "", text).strip()
            
            # Convert thành dict
            return json.loads(clean_text)

    
    
        if not all(v is not None for v in self.state.to_dict().values()):
            return "Chưa đủ thông tin để tư vấn."

        query = f"Tư vấn thẻ tín dụng phù hợp với yêu cầu: {json.dumps(self.state.to_dict(), ensure_ascii=False)}"
        docs = self.knowledge_base.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Bạn là chuyên viên tư vấn thẻ tín dụng.
        Yêu cầu khách hàng: {json.dumps(self.state.to_dict(), ensure_ascii=False)}

        Thông tin thẻ tín dụng trong hệ thống:
        {context}

        Hãy chọn loại thẻ phù hợp nhất, lưu ý là tư vấn phải nêu ra được tên thẻ tín dụng của VIB nhé! Giải thích lý do bằng cách nêu ra 3 tới 5 điểm nổi bật đáng chú ý của thẻ phù hợp với nhu cầu mở thẻ của khách hàng. 
        OUTPUT xuất ra có format như sau {{"cardname": "[Tên thẻ]", "answer": "[Câu trả lời tư vấn của bot]"}}
        Lưu ý: không viết dài dòng, chỉ viết vắn tắt, cô đọng nhất có thể, nhấn mạnh tên thẻ tín dụng bạn tư vấn là quan trọng.        
        """
        
        response = self.llm.invoke(prompt)  
        result = _extract_json_from_response(response)
        self.update_key_value(key='recommend', value=result['cardname'])
        
        return result['answer']

    #######################################
    def check_info_enough(self) -> bool:
        """
        Check if all required fields in credit_info_<userid>.json have values.
        Args:
            userid (str): The user ID.
        Returns:
            bool: True if all fields have values, False otherwise.
        """

        try:
            with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return all(value is not None for value in data.values())
        except FileNotFoundError:
            return False

    
    def confirm_credit_info(self) -> str:
        """
        Generate a confirmation message summarizing the customer's provided information.

        Returns:
            str: Confirmation text with collected customer data.
        """
        #if not os.path.exists(self.user_cdcard_info_path):
        #    return "Hiện tại tôi chưa có thông tin về nhu cầu mở thẻ của bạn."

        with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return (
            f"Hiện tôi đang có thông tin nhu cầu mở thẻ của bạn như sau:\n"
            f"- Mục đích sử dụng thẻ: {data.get('muc_dich', 'chưa có')}\n"
            f"- Thu nhập: {data.get('thu_nhap', 'chưa có')}\n"
            f"- Hạn mức mong muốn: {data.get('han_muc', 'chưa có')}\n\n"            
            "Bạn có muốn thay đổi các thông tin này hay không? "
            "Chúng tôi sẽ dựa vào thông tin này để tư vấn thẻ phù hợp cho bạn."
        )
        
    def update_key_value(self, key, value):        
        #def update_processing_status(self, status_value):
        # status_value=1: enable update value, status_value=0: disable update value to credit_info
        with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data[key] = value
        
        with open(self.user_cdcard_info_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)        
    
    def get_processing_status(self):
        with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data['processing']

    
    def do_cdcard_consult(self, customer_answer: str) -> str:
        """
        Perform the credit card advisory process:
        1. Extract info from customer answer.
        2. Update JSON file with new info.
        3. If enough info is collected, provide a recommendation.
        4. Otherwise, ask the next question.

        Args:
            customer_answer (str): Customer's input text.

        Returns:
            str: Either a recommendation or the next question.
        """
 
        if self.get_processing_status():
                              
            # 1. Extract thông tin mới từ câu trả lời khách
            extracted_info = self.extract_credit_info_func(customer_answer)

            # 2. Đọc dữ liệu hiện tại từ file hoặc tạo mới nếu chưa có
            if not os.path.exists(self.user_cdcard_info_path):
                data = {
                    "muc_dich": None,
                    "thu_nhap": None,
                    "han_muc": None
                }
            else:
                with open(self.user_cdcard_info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # 3. Cập nhật các trường mới nếu extract ra được giá trị khác None
            for field, value in extracted_info.items():
                if value is not None:
                    data[field] = value

            # 4. Ghi lại vào file
            with open(self.user_cdcard_info_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # 5. Cập nhật self.state từ file
            self.update_customer_info()

            # 6. Kiểm tra đủ thông tin chưa
            if self.check_info_enough():
                self.update_key_value(key='processing', value=0)
                return self.recommend_credit_card(self.user_cdcard_info_path)                
            else:
                next_question = self.next_question()
                if next_question:
                    return next_question
                else:
                    self.update_key_value(key='processing', value=0)
                    return self.recommend_credit_card(self.user_cdcard_info_path)                    
                
        else:
            if self.check_info_enough():               
                self.update_key_value(key='processing', value=1)
                return self.confirm_credit_info()
                
    ###########################################
    def tools_define(self):
        """
        Define tools for orchestrating the credit card advisory workflow.

        Returns:
            dict: A dictionary of tool functions (ask, update, recommend, consult).
        """
        ask_tool = Tool.from_function(
            func=self.next_question,
            name="ask_next_question",
            description="Hỏi khách hàng câu hỏi tiếp theo để thu thập thông tin mở thẻ tín dụng."
        )

        update_tool = Tool.from_function(
            func=self.update_customer_info,
            name="update_customer_info",
            description="Cập nhật thông tin khách hàng. Input là JSON dạng {\"field\": \"muc_dich\", \"value\": \"du lịch\"}."
        )

        recommend_tool = Tool.from_function(
            func=self.recommend_credit_card,
            name="recommend_credit_card",
            description="Dựa vào thông tin đã thu thập, tư vấn loại thẻ tín dụng phù hợp."
        )

        do_cdcard_consult = Tool.from_function(
            func=self.do_cdcard_consult,
            name="do_cdcard_consult",
            description=f"Dựa vào câu hỏi của Khách hàng có USERID {self.userid}, câu hỏi của khách hàng đang nằm trong luồng của tư vấn thẻ tín dụng thì kiểm tra xem các trường dữ liệu cần thu thập để tư vấn khách hàng về thẻ tín dụng đã đầy đủ chưa. Dựa vào thông tin thu thập được từ khách hàng, lấy tài liệu phù hợp từ knowledge database và đưa ra tư vấn cho khách hàng."
        )
        
        return {
            "do_cdcard_consult": do_cdcard_consult
        }