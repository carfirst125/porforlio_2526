# 1. Project Description — DCO Dynamic Creative Optimization (`version_1`)

## Mục đích

Tự động hoá việc tạo ra banner/creative quảng cáo cho OCB (một ngân hàng Việt
Nam) bằng Image GenAI, thay vì thiết kế thủ công. Người dùng (marketing/design
team) cung cấp 1 `request.json` mô tả: sản phẩm/dịp lễ gì, đối tượng khách
hàng, các asset sẵn có (theme/background, nhân vật, sản phẩm), nội dung text
cần hiển thị, và định dạng banner mong muốn (`format_type`, tương ứng 1 design
guideline có sẵn). App đọc request này, tra guideline, dựng prompt, gọi song
song nhiều Image GenAI model khác nhau (Gemini, GPT-Image-2, MAI-Image,
FLUX...), rồi trả về nhiều phiên bản thiết kế để người dùng so sánh, chọn, và
"release" phiên bản ưng ý — có thể "clone" (nhân bản) design đã release sang
nhiều kích thước hiển thị khác nhau (banner app, mạng xã hội, web...) mà không
cần thiết kế lại từ đầu.

## Phạm vi

**Trong phạm vi (in-scope):**
- Đọc `request.json`, tra guideline theo `format_type`, dựng prompt tổng hợp
  (brief + guideline + design rules + asset).
- Gọi 7 Image GenAI model (Gemini-3.1-flash-image, Gemini-3-pro-image,
  GPT-Image-2, MAI-Image-2.5/-Flash/-Pro, FLUX.2-Pro) + 1 LLM (mặc định
  gpt-5-mini, swappable) để tinh gọn prompt.
- Sinh nhiều "version" sáng tạo khác nhau cho cùng 1 request (`num_version`).
- Crop vùng `generation.safe_area` (safe area) ra khỏi `generation.canvas`
  sinh ra, theo `output.crop`/`output.size` khai báo trong request.json.
- Quản lý output: lưu ảnh, log prompt, cache ảnh đã giảm mẫu, theo đúng quy
  ước thư mục/tên file đã thống nhất.
- Release (chọn ảnh ưng ý) + Clone (nhân bản 1 ảnh đã release sang nhiều
  kích thước khác, dùng lại chính ảnh đó làm reference).
- REST API (FastAPI) phục vụ webapp Streamlit đã có sẵn.
- CLI runner (`run_cli.py`) chạy độc lập không cần webapp.

**Ngoài phạm vi (out-of-scope):**
- Không tự train/fine-tune model — chỉ gọi API model có sẵn (cloud-hosted).
- Không có hệ thống authentication/authorization (app chạy nội bộ, tin cậy
  người dùng gọi trực tiếp).
- Không có database — mọi state là file trên đĩa (`input/`, `output/`,
  `config/`).
- Không tối ưu cho nhiều người dùng đồng thời chỉnh sửa CÙNG 1 request cùng
  lúc (single-operator workflow là giả định chính, xem "Giả định" bên dưới).

## Yêu cầu chức năng chính

1. Đọc & validate `request.json` theo schema cố định (`app/models/schemas.py`).
2. Tra cứu design guideline theo `format_type`, gộp cả `general_guide` bắt buộc.
3. Giảm mẫu (downsample) toàn bộ ảnh đính kèm trước khi gửi cho model, để
   giảm dung lượng request.
4. Dựng 1 prompt tổng hợp DÙNG CHUNG cho mọi model/version trong 1 lần chạy
   (không dựng lại prompt cho từng model).
5. Chạy generate theo 5 "mode": `default` / `quality` / `eco` / `custom` /
   `all`, resolve model list qua `config/model_usage.json`.
6. Crop `generation.safe_area` (`output.crop.area` nếu có khai báo, mặc định
   dùng lại chính `generation.safe_area`) ra khỏi `generation.canvas` sinh
   được, lưu kết quả đúng bằng `output.size`.
7. Cho phép "clone" 1 ảnh đã release sang nhiều kích thước hiển thị khác.
8. Toàn bộ tiến trình phải log ra terminal + file, MỌI exception phải được
   bắt và ghi log (dump), không được để crash cả tiến trình chạy.

## Yêu cầu phi chức năng

- **Khả năng đổi LLM provider** (cho bước tinh gọn prompt) chỉ qua config,
  không sửa code (`app/llm/` — xem `2-project-technote.md`).
- **Khả năng phục hồi lỗi từng phần**: 1 model/version lỗi không được làm
  hỏng cả batch — các model/version khác vẫn tiếp tục chạy.
- **Khả năng audit**: mọi prompt thực sự gửi đi phải được lưu lại
  (`output/<request>/prompt/prompt-<model>.txt`) để xem lại được.
- **Portable**: chạy được trên máy dev (Windows, đường dẫn tuyệt đối trong
  `.env`) lẫn trong container Linux (path tương đối, fallback tự động).

## Giả định & ràng buộc

- Áp dụng: mỗi request_name chỉ có 1 người chạy `Run` tại 1 thời điểm qua
  webapp (webapp tự khoá nút Run khi đang chạy). App vẫn tự bảo vệ khỏi việc
  2 lần chạy trùng giây bằng cơ chế đặt tên file chống đè
  (`output_manager.unique_image_output_path`, xem
  `4-project-evaluation-results.md`), nhưng KHÔNG có cơ chế lock cấp
  request để ngăn 2 người bấm Run cùng lúc cho cùng 1 request — đây là rủi ro
  đã biết, chấp nhận được ở quy mô sử dụng hiện tại (1 team nhỏ).
- Model name/API shape của các model rất mới (Gemini-3.x, MAI-Image,
  FLUX.2-Pro...) được implement theo format phổ biến nhất cho lớp API đó —
  chưa xác nhận 100% khớp tài liệu chính thức (model quá mới tại thời điểm
  viết code). Xem ghi chú "Cần double-check" trong `2-project-technote.md`.
- `.env` không thể chỉnh sửa trực tiếp qua công cụ remote-write (bị chặn vì
  lý do bảo mật) — mọi giá trị cấu hình mới thêm vào phải có default hợp lý
  hoặc alias tương thích ngược để không yêu cầu người dùng phải tự sửa `.env`
  ngay (xem alias `azure_gpt5mini` trong `app/llm/factory.py`).
- **request.json schema (quan trọng, xem `4-project-evaluation-results.md`
  2026-08-20):** schema CHÍNH THỨC là template lồng nhau (`generation.canvas`
  / `generation.safe_area` + `output.size` là object `{width, height}`) in
  literal ở đầu REQUEST.md, khớp với file thật `input/DB_NOTI_AUTOPAY_NOEL.json`
  hiện tại. Có 1 đợt implement trước đó (2026-08-19) đã dựng theo 1 schema
  phẳng khác (`output.size` dạng string "W,H", `design_options.protect_area`)
  vì lúc đó lấy nhầm căn cứ từ 1 file `input/*.json` cũ user quên xoá — đã
  được sửa lại đúng theo REQUEST.md, KHÔNG dùng field `protect_area` nữa.
