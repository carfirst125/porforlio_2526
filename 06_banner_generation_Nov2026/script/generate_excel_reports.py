"""
Đọc file design_feedback_topics.json và xuất ra 2 file Excel:

1. topics_to_ids.xlsx
   Schema: ["topics", "existed_in_ids", "count_ids"]
   Mỗi dòng là 1 topic, "existed_in_ids" là danh sách ID (ngăn cách bởi ", "),
   "count_ids" là số lượng ID trong list đó.

2. id_to_topics.xlsx
   Schema: ["id", "topics", "count_topics", "quick_fit_topic"]
   Mỗi dòng là 1 ID thiết kế, "topics" là danh sách các topic mà ID đó xuất hiện
   (ngăn cách bởi "; "), "count_topics" là số lượng topic mà ID đó thuộc về.
   "quick_fit_topic" = 1 nếu ID đó thuộc ít nhất 1 trong nhóm "quick fit" topic
   (danh sách QUICK_FIT_TOPICS bên dưới — các lỗi sửa nhanh, không cần regen ảnh:
   đổi thuật ngữ Autobill, CTA quá dài/quá ngắn/quá nhỏ, gợi ý đổi câu chữ CTA,
   thiếu nút CTA, không dùng từ viết tắt/tiếng Anh); ngược lại = 0.

Cách chạy:
    pip install openpyxl --break-system-packages   # nếu chưa có
    python generate_excel_reports.py

Mặc định script tìm file JSON cùng thư mục (design_feedback_topics.json) và xuất
2 file .xlsx cũng vào thư mục đó. Có thể đổi đường dẫn qua biến JSON_PATH / OUTPUT_DIR
bên dưới.
"""

import json
import os
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# ---- Cấu hình đường dẫn ----------------------------------------------------
# Dùng thư mục hiện hành (nơi bạn gõ lệnh `python generate_excel_reports.py`),
# KHÔNG dùng thư mục chứa file script. Nghĩa là: đặt file design_feedback_topics.json
# vào đúng thư mục bạn đang mở terminal/cmd, rồi chạy script từ đó — file kết quả
# .xlsx cũng sẽ được tạo ngay tại đó.
BASE_DIR = os.getcwd()
JSON_PATH = os.path.join(BASE_DIR, "design_feedback_topics.json")
TOPICS_XLSX_PATH = os.path.join(BASE_DIR, "topics_to_ids.xlsx")
IDS_XLSX_PATH = os.path.join(BASE_DIR, "id_to_topics.xlsx")

# Nhóm topic được coi là "quick fit" (sửa nhanh trên creative, không cần regen ảnh).
# Giá trị phải khớp CHÍNH XÁC với field "topic" trong design_feedback_topics.json.
QUICK_FIT_TOPICS = {
    "Đổi thuật ngữ 'Autobill' thành 'Thanh toán tự động'",
    "CTA quá dài, cần rút ngắn còn 2-3 từ",
    "CTA quá ngắn, chưa đủ 2 từ",
    "CTA quá nhỏ hoặc vị trí chưa nổi bật",
    "Gợi ý đổi câu chữ CTA cụ thể",
    "Thiếu nút CTA",
    "Không dùng từ viết tắt/tiếng Anh (VD: ADSL)",
}

FONT_NAME = "Arial"
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(name=FONT_NAME, bold=True, color="FFFFFF", size=11)
BODY_FONT = Font(name=FONT_NAME, size=11)


def load_topics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["topics"]


def style_header_row(ws, row_idx, num_cols):
    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def autosize_columns(ws, widths):
    for col_idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def build_topics_to_ids_workbook(topics, output_path):
    """Bảng 1: mỗi topic -> list ID + số lượng ID."""
    wb = Workbook()
    ws = wb.active
    ws.title = "topics_to_ids"

    headers = ["topics", "existed_in_ids", "count_ids"]
    ws.append(headers)
    style_header_row(ws, 1, len(headers))

    for item in topics:
        topic_name = item["topic"]
        ids = item["ids"]
        row = [topic_name, ", ".join(ids), len(ids)]
        ws.append(row)

    # Định dạng phần thân bảng
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    ws.freeze_panes = "A2"
    autosize_columns(ws, [45, 90, 12])

    wb.save(output_path)
    return output_path


def build_id_to_topics_workbook(topics, output_path):
    """Bảng 2: mỗi ID -> list topic + số lượng topic."""
    id_to_topics = defaultdict(list)
    for item in topics:
        topic_name = item["topic"]
        for design_id in item["ids"]:
            id_to_topics[design_id].append(topic_name)

    # Sắp xếp ID theo số thứ tự (A001, A003, ... A135)
    def id_sort_key(design_id):
        digits = "".join(ch for ch in design_id if ch.isdigit())
        return int(digits) if digits else 0

    sorted_ids = sorted(id_to_topics.keys(), key=id_sort_key)

    wb = Workbook()
    ws = wb.active
    ws.title = "id_to_topics"

    headers = ["id", "topics", "count_topics", "quick_fit_topic"]
    ws.append(headers)
    style_header_row(ws, 1, len(headers))

    for design_id in sorted_ids:
        topic_list = id_to_topics[design_id]
        is_quick_fit = 1 if (set(topic_list) & QUICK_FIT_TOPICS) else 0
        row = [design_id, "; ".join(topic_list), len(topic_list), is_quick_fit]
        ws.append(row)

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    ws.freeze_panes = "A2"
    autosize_columns(ws, [10, 100, 14, 16])

    wb.save(output_path)
    return output_path


def main():
    if not os.path.isfile(JSON_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file '{JSON_PATH}'.\n"
            f"Hãy đặt file design_feedback_topics.json vào cùng thư mục bạn đang chạy "
            f"lệnh `python generate_excel_reports.py` (thư mục hiện hành: {BASE_DIR})."
        )

    topics = load_topics(JSON_PATH)

    path1 = build_topics_to_ids_workbook(topics, TOPICS_XLSX_PATH)
    print(f"Đã tạo: {path1}")

    path2 = build_id_to_topics_workbook(topics, IDS_XLSX_PATH)
    print(f"Đã tạo: {path2}")


if __name__ == "__main__":
    main()
