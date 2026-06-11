"""
field_definitions_draft.py — AUTO-GENERATED, DO NOT USE IN PRODUCTION

Generated : 2026-06-08 11:55
Source    : ../documents_bgem3.parquet
Script    : scripts/extract_fields.py

Review this file, make any corrections, then copy the contents
into src/knowledge_graph/field_definitions.py.
"""

DOMAIN_FIELDS: dict[str, dict[str, str]] = {
    "credit_card": {
        "is_student": (
            "Bạn là sinh viên không? (Ví dụ: để xem xét các sản phẩm thẻ phù hợp với độ "
            "tuổi của bạn)"
        ),
        "airline_benefits": (
            "Bạn có thường xuyên bay với Vietnam Airlines và muốn tích lũy dặm thưởng "
            "không? (Ví dụ: để giới thiệu thẻ Premier Boundless)"
        ),
        "online_payment": (
            "Bạn có nhu cầu sử dụng các tính năng thanh toán, giao dịch trực tuyến qua "
            "app hoặc website không? (Ví dụ: thẻ nào hỗ trợ tiện ích này)"
        ),
        "annual_fee": (
            "Bạn có sẵn sàng chi trả một khoản phí thường niên nhất định cho thẻ tín "
            "dụng của mình không? (Ví dụ: năm đầu miễn phí, các năm sau 299.000 VNĐ)"
        ),
        "installment_option": (
            "Bạn có đang tìm kiếm một sản phẩm thẻ cho phép thanh toán mua sắm theo kỳ "
            "hạn với lãi suất ưu đãi không? (Ví dụ: để xem xét các gói trả góp hấp dẫn)"
        ),
    },
    "insurance": {
        "age": (
            "Bạn sinh năm nào? (ví dụ: 1980)"
        ),
        "payment_term": (
            "Bạn dự định đóng phí bảo hiểm trong bao lâu? (ví dụ: 5-10 năm)"
        ),
        "sum_insured": (
            "Mức độ bảo vệ tài chính bạn cần là bao nhiêu? (ví dụ: từ 50 triệu đồng)"
        ),
        "health_condition": (
            "Có bất kỳ vấn đề về sức khỏe nào không ảnh hưởng đến việc tham gia bảo "
            "hiểm?"
        ),
        "occupation_status": (
            "Bạn đang làm việc trong lĩnh vực gì? (ví dụ: công nhân, văn phòng, kinh "
            "doanh)"
        ),
    },
    "savings": {
        "thu_nhap": (
            "Thưa Quý khách, để chúng tôi tư vấn sản phẩm phù hợp nhất, xin Quý khách "
            "cho biết thu nhập hàng tháng của mình là bao nhiêu? (Ví dụ: 10 triệu "
            "đồng/tháng)"
        ),
        "muon_thu_hien": (
            "Quý khách đang có kế hoạch sử dụng số tiền nhàn rỗi để làm gì? (Ví dụ: Mua "
            "nhà, mua xe, dự trữ cho con đi học)"
        ),
        "ky_hu_khong_gian": (
            "Quý khách mong muốn gửi tiết kiệm trong khoảng thời gian bao lâu? (Ví dụ: 1 "
            "tháng, 3 tháng, 6 tháng)"
        ),
        "nguy_co_chap_nhan_duoc": (
            "Quý khách có sẵn sàng chấp nhận mức độ rủi ro nào trong quá trình đầu tư? "
            "(Ví dụ: Rất thấp, trung bình)"
        ),
        "loai_tai_khoan": (
            "Quý khách có đang tìm kiếm một loại hình sản phẩm tiết kiệm hoặc đầu tư cụ "
            "thể nào không? (Ví dụ: Tiết kiệm bậc thang, gửi USD)"
        ),
    },
    "general": {
        "loai_san_pham": (
            "Bạn đang quan tâm đến sản phẩm hay dịch vụ nào của VIB ạ? "
            "(thẻ, vay, bảo hiểm, tiết kiệm, hay dịch vụ khác?)"
        ),
    },
}

# Domain display names (for confirm messages)
DOMAIN_LABELS = {
    "credit_card": "Thẻ Tín Dụng / Thẻ Ghi Nợ",
    "insurance": "Bảo hiểm",
    "loan": "loan",
    "savings": "Tiết kiệm / Đầu tư",
    "general": "sản phẩm ngân hàng",
}

# Domain → retrieval bias keywords
DOMAIN_KEYWORDS = {
    "credit_card": "thẻ tín dụng VIB",
    "insurance": "Bảo hiểm VIB",
    "loan": "loan VIB",
    "savings": "Sản phẩm tiết kiệm / đầu tư",
    "general": "sản phẩm dịch vụ VIB",
}


def get_fields(domain: str) -> dict[str, str]:
    return DOMAIN_FIELDS.get(domain, DOMAIN_FIELDS["general"])


def get_domain_keyword(domain: str) -> str:
    return DOMAIN_KEYWORDS.get(domain, "VIB ngân hàng")
