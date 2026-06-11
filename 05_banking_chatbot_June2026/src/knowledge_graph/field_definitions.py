"""
Knowledge Graph — required fields per advisory domain.

Each domain has a dict: {field_name: question_to_ask_customer}
The advisor pipeline uses these to drive multi-turn info collection.
"""

DOMAIN_FIELDS: dict[str, dict[str, str]] = {
    "credit_card": {
        "thu_nhap_hang_thang": (
            "Thu nhập hàng tháng của bạn khoảng bao nhiêu ạ? "
            "(ví dụ: dưới 10 triệu, 10–20 triệu, trên 20 triệu)"
        ),
        "muc_chi_tieu_chu_yeu": (
            "Bạn thường chi tiêu chủ yếu vào lĩnh vực nào? "
            "(ăn uống/cà phê, mua sắm online, du lịch, xăng xe, siêu thị...)"
        ),
        "uu_tien_quyen_loi": (
            "Bạn ưu tiên quyền lợi nào từ thẻ tín dụng? "
            "(hoàn tiền cashback, tích điểm/dặm bay, trả góp 0%, ưu đãi nhà hàng/ăn uống...)"
        ),
        "co_the_hien_tai": (
            "Bạn hiện đang dùng thẻ tín dụng ngân hàng nào chưa ạ?"
        ),
    },
    "insurance": {
        "tuoi": (
            "Bạn cho mình hỏi bạn đang ở độ tuổi khoảng bao nhiêu ạ?"
        ),
        "tinh_trang_gia_dinh": (
            "Bạn đã có gia đình hoặc con nhỏ chưa ạ?"
        ),
        "muc_dich_bao_hiem": (
            "Bạn muốn bảo hiểm để làm gì chủ yếu ạ? "
            "(bảo vệ sức khỏe/bệnh viện, tích lũy tiết kiệm, bảo vệ nhân thọ cho gia đình, tai nạn...)"
        ),
        "ngan_sach_hang_thang": (
            "Bạn dự kiến dành khoảng bao nhiêu tiền mỗi tháng cho bảo hiểm ạ?"
        ),
    },
    "loan": {
        "muc_dich_vay": (
            "Bạn cần vay để làm gì ạ? "
            "(mua nhà/đất, mua xe, tiêu dùng cá nhân, kinh doanh, hay mục đích khác?)"
        ),
        "so_tien_can_vay": (
            "Bạn cần vay khoảng bao nhiêu tiền ạ?"
        ),
        "thu_nhap_hang_thang": (
            "Thu nhập hàng tháng của bạn (hoặc hộ gia đình) khoảng bao nhiêu ạ?"
        ),
        "tai_san_the_chap": (
            "Bạn có tài sản thế chấp không ạ? "
            "(sổ đỏ/sổ hồng, ô tô, sổ tiết kiệm, hay muốn vay tín chấp không cần thế chấp?)"
        ),
    },
    "savings": {
        "so_tien_gui": (
            "Bạn dự kiến gửi khoảng bao nhiêu tiền ạ?"
        ),
        "thoi_han_gui": (
            "Bạn muốn gửi trong bao lâu ạ? "
            "(1 tháng, 3 tháng, 6 tháng, 12 tháng, hay dài hạn hơn?)"
        ),
        "muc_tieu": (
            "Mục tiêu chính của bạn là gì ạ? "
            "(lãi suất cao nhất, an toàn và linh hoạt rút trước hạn, hay tích lũy dài hạn?)"
        ),
    },
    "general": {
        "loai_san_pham": (
            "Bạn đang quan tâm đến sản phẩm hay dịch vụ nào của VIB ạ? "
            "(thẻ, vay, bảo hiểm, tiết kiệm, hay dịch vụ khác?)"
        ),
    },
}

# Domain display names (for prompts)
DOMAIN_LABELS = {
    "credit_card": "thẻ tín dụng / thẻ ghi nợ",
    "insurance": "bảo hiểm",
    "loan": "vay vốn",
    "savings": "tiết kiệm / đầu tư",
    "general": "sản phẩm ngân hàng",
}

# Domain → search keywords (used to bias retrieval)
DOMAIN_KEYWORDS = {
    "credit_card": "thẻ tín dụng VIB",
    "insurance": "bảo hiểm VIB",
    "loan": "vay vốn VIB",
    "savings": "tiết kiệm VIB lãi suất",
    "general": "sản phẩm dịch vụ VIB",
}


def get_fields(domain: str) -> dict[str, str]:
    return DOMAIN_FIELDS.get(domain, DOMAIN_FIELDS["general"])


def get_domain_keyword(domain: str) -> str:
    return DOMAIN_KEYWORDS.get(domain, "VIB ngân hàng")
