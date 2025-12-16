import os
import json
import re
import logging

from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

from app.config.env_loader import logger 
from app.utils.clients import OpenAIGPT
from app.agentic.utils.agent_helper import AgentTypes, AgentHelper
from app.utils.logger import logger, UsageLoggerHandler

import warnings
warnings.filterwarnings("ignore")

###################################################
# Router agent

class AgentRouter:
    
    def __init__(self, userid: str): 
        """
        Initialize the AgenticLangOpenai agent.
        Args:
            llm: The language model instance.
            userid (str): Unique identifier for the user session.
        """
        self.log_handler = UsageLoggerHandler(step="AgentRouter")
        self.llm = AzureChatOpenAI(**OpenAIGPT.LLM_CONFIG, callbacks=[self.log_handler])
        self.userid = userid
        self.chat_history = self._load_history_from_file()

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
            
    
    def meta_agent_router(self, question):   
            
        print("meta_agent_router begin")  # debug

        helper = AgentHelper(f"credit_info_{self.userid}.json")
        agent_type = helper.check_processing()
        print("agent_type: ",agent_type)
        if agent_type == AgentTypes.CREDIT_CARD_CONSULTANT:
            return agent_type
        else:
            classification_prompt = PromptTemplate(
                input_variables=["HISTORY", "QUESTION"],
                template="""
                Bạn là bộ phân loại câu hỏi. Bạn có nhiệm vụ phân loại câu hỏi của khách hàng thuộc loại nào trong các loại sau:
                1- CREDIT_CARD_CONSULTANT: Nếu câu hỏi cho thấy Khách hàng cần tư vấn để chọn thẻ tín dụng, hay mở thẻ tín dụng (CREDIT_CARD_CONSULTANT) → trả về CREDIT_CARD_CONSULTANT.
                2- PRODUCT_INFO: Những yêu cầu sau của Khách hàng sẽ được trả về là PRODUCT_INFO:
                   Câu hỏi thuộc về hỏi thông tin dịch vụ, sản phẩm của ngân hàng VIB, bao gồm cả thông tin thẻ mà không phải hỏi tư vấn mở thẻ (PRODUCT_INFO) 
                   Câu hỏi về mở thẻ thanh toán, thẻ thanh toán phải là thẻ tín dụng nên trả về PRODUCT_INFO.
                   Câu hỏi về cổ phiếu, tình hình kinh doanh của VIB, ban lãnh đạo, kế hoạch, sách phát triển, hoạt động xã hội hay bất cứ nội dung gì có liên quan tới ngân hàng VIB.
                   Gởi tiết kiệm, đổi ngoại tệ, cho vay mua ô tô, vay mua nhà, mở hay đóng thẻ tín dụng, thẻ thanh toán, đều là sản phẩm của VIB.
                   Thanh lý nhà đất, máy móc, thiết bị, xe ô tô cũng là sản phẩm của VIB.
                   Thông tin liên hệ với VIB để giải đáp thắc mắc.
                3- OTHERS: Nếu câu hỏi thuộc dạng chào hỏi, phàn nàn, phản hồi, chia sẻ một vấn đề cá nhân, hay một chủ đề không liên quan (OTHERS) → trả về OTHERS.
                Chú ý cố gắng kỹ câu hỏi của khách hàng, để phân loại cho chính xác, nhiều câu khách hàng có nhắc thẻ tín dụng nhưng ko phải là hỏi về tư vấn mà là phàn nàn về sản phẩm thẻ (OTHERS), hoặc hỏi thông tin sản phẩm thẻ (PRODUCT_INFO) thì bạn nhớ phân loại cho đúng.     
                
                Lịch sử hội thoại:
                {HISTORY}
                
                Câu hỏi: 
                {QUESTION}
                
                Dựa vào lịch sử hội thoại của Khách hàng và câu hỏi, hãy thực hiện phân loại và trả ra output theo JSON format: {{"response": "[Loại câu hỏi]"}}.
                Lưu ý: Chỉ trả về đúng JSON, không giải thích gì thêm.
                """
            )
            
            history_text = self._format_history()
            history_text = "Đây là câu hỏi đầu tiên của Khách hàng, chưa có lịch sử hội thoại." if history_text == "" else history_text
            #print("history_text:", history_text)  # debug
            #print("question:", question)  # debug
            
            formatted_prompt = classification_prompt.format(
                HISTORY=history_text,
                QUESTION=question
            )
    
            #print("formatted_prompt:", formatted_prompt)  # debug
            result = self.llm.invoke(formatted_prompt)#[[HumanMessage(content=formatted_prompt)]],callbacks=[self.log_handler])
            #print("result:", result)  # debug
            raw_text = result.content if hasattr(result, "content") else str(result)
            print(f"raw_text:\n {raw_text}")  # debug
            
            # Dùng regex để lấy JSON trong output (phòng khi LLM in thêm text)
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not match:
                raise ValueError(f"Không tìm thấy JSON trong output: {raw_text}")
            
            parsed = json.loads(match.group())
            label = parsed.get("response", "OTHERS")  # fallback nếu thiếu key
            return label


    def question_others_response(self, question):    
    
        #log_handler = UsageLoggerHandler(step="AgentRouter.question_others_response")
        
        classification_prompt = PromptTemplate(
            input_variables=["HISTORY", "QUESTION"],
            template="""
            Bạn là trợ lý ảo ViePro của VIB, chuyên tư vấn giải đáp thắc mắc của Khách hàng về sản phẩm dịch vụ của VIB. Sau đây bạn phân loại câu hỏi khách hàng thuộc loại này trong 4 loại sau và đưa ra câu trả lời phù hợp
            1 - CUSTOMER_FEEDBACK: Câu hỏi khách hàng là một phản hồi về sản phẩm/dịch vụ của VIB, hay phản hồi về chất lượng câu trả lời của chatbot. Bạn hãy phản hồi phù hợp cho lời khen chê đó của Khách hàng một các lịch sự. Nếu khách hàng gay gắt trong việc chê hay phê bình thậm tệ sản phẩm dịch vụ VIB, bạn hãy tìm cách trả lời khéo léo theo hướng làm dịu cảm xúc của khách hàng và lúc nào của tỏ ra thiện chí lắng nghe khách hàng để hỗ trợ thêm.
            2 - PERSONAL_SHARING: Câu hỏi khách hàng chia sẻ một vấn đề cá nhân, không liên quan tới sản phẩm dịch vụ của VIB. Bạn hãy trả lời chia sẻ sự đồng cảm và hướng Khách hàng về với sản phẩm dịch vụ của VIB.
            3 - UNRELATED: Câu hỏi khách hàng hoàn toàn ở một đề tài khác, nằm ngoại phạm vi nội dung tư vấn hỗ trợ về sản phẩm dịch vụ của VIB như hỏi về giải toán, hỏi về thời tiết, hỏi về chính trị, hỏi về bạo lực, vân vân... Bạn hãy trả lời Khách hàng bạn là ai và đang sẵn sàng hỗ trợ khách hàng về vấn đề gì.
            4 - GREETING_FAREWELL:Câu hỏi khách hàng là câu chào hỏi, câu bắt đầu câu chuyện thì bạn hãy phản hồi lại lịch sự và hỏi khách hàng cần hỗ trợ gì không. Nếu là câu tạm biệt, câu đồng ý, cảm ơn thì bạn cũng lịch sự cảm ơn lại khách hàng đã tin tưởng dùng sản phẩm của vib, và mong muốn được phục vụ, hỗ trợ, giải đáp thắc mắc cho khách hàng.
            
            Lịch sử hội thoại:
            {HISTORY}
            
            Câu hỏi: 
            {QUESTION}
            
            Dựa vào lịch sử hội thoại của Khách hàng và câu hỏi, hãy thực hiện phân loại và trả ra output tương ứng theo JSON format: {{"qtype": "[Loại câu hỏi CUSTOMER_FEEDBACK, PERSONAL_SHARING, UNRELATED, GREETING_FAREWELL]", "response": "[câu trả lời của llm]"}}.
            Lưu ý: Chỉ trả về đúng JSON, không giải thích gì thêm.
            """
        )
       
        history_text = self._format_history()
        history_text = "Đây là câu hỏi đầu tiên của Khách hàng, chưa có lịch sử hội thoại." if history_text == "" else history_text
        #print("history_text:", history_text)  # debug
        #print("question:", question)  # debug
        
        formatted_prompt = classification_prompt.format(
            HISTORY=history_text,
            QUESTION=question
        )
        #formatted_prompt = self.formatted_prompt(formatted_prompt)
        
        #print("formatted_prompt:", formatted_prompt)  # debug
        result = self.llm.invoke(formatted_prompt)#[[HumanMessage(content=formatted_prompt)]],callbacks=[self.log_handler])
        #print("result:", result)  # debug
        raw_text = result.content if hasattr(result, "content") else str(result)
        print(f"raw_text:\n {raw_text}")  # debug
        
        # Dùng regex để lấy JSON trong output (phòng khi LLM in thêm text)
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"Không tìm thấy JSON trong output: {raw_text}")
        
        parsed = json.loads(match.group())
        qtype = parsed.get("qtype", "UNKNOWN")  # fallback nếu thiếu key
        answer = parsed.get("response", "UNKNOWN")  # fallback nếu thiếu key
        print(f"qtype: {qtype}, answer: {answer}")  # debug
        self.chat_history.append({"user": question, "ai": answer})
        self._save_history_to_file()
        return qtype, answer
        
