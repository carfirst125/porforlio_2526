File mô tả yêu cầu thực hiện Python project gồm backend (python/fastapi) và frontend (streamlit)
Thực hiện code add nằm trong thư mục version_1

# Danh sách các Image GenAI model thực hiện chạy thử nghiệm.
## Mô hình sinh ảnh (Image GenAI Model)
Hosted in GCP
Gemini-3.1-flash-image
Gemini-3-pro-image

Hosted in Azure
GPT-Image-2 


## Mô hình sinh ngôn ngữ (LLM)
Hosted in Azure
gpt-5-mini 

# Template cho request.json để app chạy

{
  "creative_id": "DB_DEMO_AUTOPAY_NOEL",
  "format_type": "banner_popup_noti",

  "design_brief": "Thiết kế banner noti về tính năng thanh toán tự động trên app OMNI của OCB. Thiết kế cần gây chú ý và ấn tượng người xem ở thông điệp đưa ra. Thiết kế hiển thị trong dịp Giáng sinh 2026.",

  "buckets": {
    "theme": "Bucket_1/db_theme_01.png",
    "context": [],
    "product": "Bucket_3/db_phone_autopayment.png"
  },

  "reference": "",

  "text": {
    "headline": "",
    "message": "Tự động thanh toán. Một chạm dễ dàng.",
    "cta_label": "",
    "cta_deep_link": "banner/notification/autopay"
  },

  "guidelines": "",

  "num_version": 3,

  "design_options": {
    "generate_person": false,
    "has_logo": false
  },

  "generation": {
    "canvas": {
      "width": 1200,
      "height": 800
    },

    "safe_area": {
      "width": 1035,
      "height": 420,
      "position": "center",
      "x": 82,
      "y": 190
    },

    "instruction": "Generate the design on the full 1200x800 canvas. Keep all important visual elements, text, product, and composition fully contained within the centered 1035x420 safe area. The outer area may contain background extension or decorative elements, but no essential design information should be placed outside the safe area."
  },

  "output": {
    "size": {
      "width": 1035,
      "height": 420
    },
    "crop": {
      "source": "generation.canvas",
      "area": {
        "x": 82,
        "y": 190,
        "width": 1035,
        "height": 420
      }
    },
    "clone": []
  }
}


# Mô tả nội dung cần làm:
Tôi muốn thực hiện project app với mô tả như sau:

## Mô tả
Thực hiện Image generation AI sử dụng LangGraph với workflow được mô tả như sau:
[1] Đọc file request.json: file này chứa tất cả thông tin cần thiết cho yêu cầu tạo ảnh bao gôm: 
    - "design_brief": chứa mô tả yêu cầu tạo ảnh cho sản phẩm gì, dịp lễ nào hay ngày thường, cho đối tượng là ai, phong cách thiết kế ra sao.
	- "bucket": gồm 3 bucket theme, context, product. Theme chứa ảnh background, context chứa ảnh nhân vật, product chứa ảnh sản phẩm. Mỗi bucket cũng có thể nhận vào mô tả bằng text thay vì đường dẫn ảnh (ví dụ "theme": "không gian xanh lá gradient, quầy tính tiền...") -- khi đó image model sẽ tự vẽ background/nhân vật/sản phẩm theo đúng mô tả đó thay vì dùng ảnh đính kèm (xem `app/core/asset_resolver.py::looks_like_image_path()`).
	- "text": chứa nội dung text cần xuất hiện trong thiết kế ở các vị trí Headline, Message và CTA
	- "generation": chứa nội dung về size thiết kế (canvas) và vùng đặt content thiết kế (safe_area), kèm "instruction": canvas là size thiết kế image model sẽ generate ra, trong thiết kế thì nội dung chỉ được đặt trong safe area. Vùng ngoài safe area chỉ là background, không có decorative elements, no design information.
	- "output": "size" kích thước mong muốn xuất ra. Chính là kích thước trong safe_area. "crop" hướng dẫn kích thước crop ở trục x và y lùi vào đều ở mép trái phải, trên dưới.
	- "num_version": 1 (default), là số thiết kế người dùng muốn tạo ra với 1 image model, Nếu num_version > 1, ý muốn tăng sự sáng tạo của image model để tạo ra 1 version thiết kế khác dựa trên prompt cũ, ví dụ như cũng những theme, context, product đó nhưng layout có thể khác đi, có thể thêm một vài decorator khác vào hình, hoặc thay đổi tư thế biểu cảm của nhân vật ở context khác đi để sáng tạo ra một thiết kế mới.
                     "output": kích thước ngõ ra ở "size", ở "clone" lưu các kích thước khác muốn nhân bản thêm để có thể hiển thị ở nhiều platform khác nhau.
	- "design_options": có 2 option là "has_logo" cho phép trong thiết kế có logo hay không, và "generate_person" là true sẽ yêu cầu image model tự thêm nhân vật vào thiết kế trong trường hợp bucket.context rỗng (không cung cấp nhân vật)
	- "format_type": đây là trường quan trọng chỉ ra ảnh đó sử dụng cho mục đích gì, với mỗi format_type sẽ có một design_guideline tương ứng, quy định trong ./lib/guidelines/guidelines.json với tên là giá trị của format_type. Các guidelines content này cần được đưa vào prompt, và attach thêm file guideline vào prompt để gởi tới image model.
[2] Lấy guideline: dựa vào format_type trong request.json để tra cứu design_guideline tương ứng. Tra cứu thông qua ./lib/guidelines/guidelines.json với tên là giá trị của format_type. Các guidelines content này cần được đưa vào prompt, và attach thêm file guideline thêm vào prompt gởi tới image model. 
  Lưu ý quan trọng: trong ./lib/guidelines/guidelines.json có field "general_guide" với value chính là các format_type chung bắt buộc với design guideline cũng cần đưa vào prompt. Nên bạn phải tìm vào tổng hợp hết, không được xót guideline.
[3] Viết phần prompt hướng dẫn thiết kế phải đặt trong safe area, mô tả sao cho image model hiểu theo instruction.
[4] Xử lý giảm mẫu/độ phân giải danh sách hình ảnh cần đính kèm gởi Image model. Sau bước [1][2], bạn có danh sách các ảnh cần đính kèm, bạn thực hiện giảm mẫu tất cả ảnh này để giảm dung lượng lại phù hợp dể gởi tới model.
[5] Tạo prompt cho image model: Tổng hợp từ thông tin tạo thành prompt tổng gồm: 
    (a) Thông tin từ [2] [3] [4]
	(b) Thêm vào phần design rules mô tả trong ./prompt/design_rules.md 
	(c) Kiểm tra phần design_options để thêm prompt hướng dẫn về việc có tạo logo hay không, và có nhờ Image model hỗ trợ generate nhân vật hay không (thứ tự ưu tiên nếu bucket đã có hình nhân vật trong context thì Image model không tạo, nếu không có thì model sẽ tạo nếu generate_person=True).
	(d) Đính kèm các hình guideline từ [2], hình bucket từ json input với ghi chú rõ ràng.
	(e) Tổng hợp lại toàn bộ prompt với kiến trúc prompt rõ ràng chi tiết phù hợp cho Image model
     Lưu ý quan trọng: trong prompt phải làm sao guide được là guideline ở [2] là áp dụng cho thiết kế trong phần safe_area chứ ko phải cho canvas. vì phần safe_area sẽ được crop ra dùng.
	 Hình ảnh sử dụng đính kèm là hình ảnh đã qua xử lý giảm mẫu ở [4]
	 
[6] Sau khi hình ảnh trả về từ Image model, bạn thực hiện gọi crop để xử lý lấy ra phần safe area và lưu ở output.

Output được lưu trong thư mục output --> <request_name> (tương ứng với request_name.json) --> display size --> image_model (tên mô hình tạo ra thiết kế) --> ảnh thiết kế output, tên đặt theo <creative_id>_<display-size>_<image_model_name>_<YYYY-mm-hh-mm> với YYYY-mm-hh-mm lấy ở thời điểm của phiên chạy.
Tạo thư mục prompt trong output như đường dẫn: output --> <request_name> (tương ứng với request_name.json) --> prompt --> prompt-model-A.txt để lưu log. Tất cả sub-prompt tạo bởi các bước xử lý đều được lưu tại đây. Prompt sau tổng hợp cũng lưu tại đây
Các hình ảnh sau khi giảm mẫu làm nhẹ để đính kèm lưu tại: output --> <request_name> (tương ứng với request_name.json) --> image_processing --> .png

Bạn xem xét sử dụng thêm GPT-5-MINI trong các tác vụ cần thiết đế sinh text linh hoạt như prompt thay vì chốt cứng. Hoặc dùng để viết lại prompt tổng cho cô đọng và có cấu trúc hơn.

## [1] Input - Yêu cầu đầu vào
File request.json input sẽ đặt trong folder input, app sẽ nhận file input để xử lý.

## [2] Output và cấu trúc lưu trữ  
Trong request.json phần output.size chỉ chỉ quy định về output size output, chỉ tạo ra thiết kế với 1 size duy nhất.
Output được lưu trong thư mục output --> <request_name> (tương ứng với request_name.json) --> display size --> image_model (tên mô hình tạo ra thiết kế) --> ảnh thiết kế output, tên đặt theo <creative_id>_<display-size>_<image_model_name>_<YYYY-mm-hh-mm> với YYYY-mm-hh-mm lấy ở thời điểm của phiên chạy.

## [3] Prompt cho Image model
 Các ý đã nêu rõ bên trên. Đây là vài ý nhắc lại và thêm về prompt.
 3a) đưa cho image model gồm (a) thông tin yêu cầu thiết kế gì, (b) bắt buộc sử dụng require image liệt kê trong "bucket" gồm theme, context, product image nào vào thiết kế, (c) input guildline, (d) các RULE MUST và NOT MUST trong một file prompt tên cố định do DEV cung cấp.
 3b) Một số rules thiết kế cần xem xét đưa vào prompt:
       App thể hiện sự sáng tạo của mình trong việc add thêm các ý tưởng thiết kế decoration thêm cho banner tạo ra điểm nhấn theo hướng kích thích sự tò mò của người xem, thúc đẩy hành động.
       Thiết kế phải cho thấy sự mềm mại không khô cứng khi xem, thiết kế thoáng, không chằng chịt đối tượng trên đó.
       Hình ảnh tạo ra có logo hay không tùy vào theme có logo hay không
       Hình ảnh nhân vật trong Context có thể được thay đổi tư thế, biểu cảm dựa vào mô tả design_brief.
 3c) Prompt gồm text và append ảnh sẽ có dung lượng khá lớn do ảnh. Do đó cần phải có bước xử lý ảnh giảm mẫu, giảm phân giải về nhỏ hơn trước khi append vào prompt để gởi tới Image GenAI model API.
 3d) Prompt sau khi xử lý giảm mẫu ảnh để nhẹ ok rồi thì chỉ nên xuất 1 lần và lưu output --> <request_name> (tương ứng với request_name.json) --> prompt --> prompt-version-1.txt 
     (ở những version khác nhau có thể có mô tả trong prompt khác nhau nên lưu thành nhiều file, và mỗi file này sẽ dùng chung cho tất cả image model). Nghĩa là lần sau chạy với model khác với cùng 1 version thì không cần tạo lại prompt mà đọc ở đây ra request API.
 
## [4] Runing mode:
Bạn tạo giúp tôi một file model_usage để quản lý các model tương ứng với running mode
{
	"default": {"Gemini-3.1-flash"
				},
	"quality": {"GPT-Image-2",
				"MAI-Image-2.5-pro",
				"Gemini-3-pro"
				},
	"eco":		{"FLUX.2-pro",
				 "MAI-Image-2.5",
				 "Gemini-3.1-flash"
				 },
	"custom": 	{<user enter model name>}
	"all": {"quality","eco"}
}
quyết định là chạy generate ảnh như thế nào. Cụ thể:
 --mode=default thì code sẽ chạy tạo ảnh bằng tên model trong default của model_usage
 --mode=quality thì sẽ chạy tạo ảnh cùng 1 prompt chạy 3 lần cho 3 model lưu trong quality
 --mode=custom thì sẽ chạy tạo ảnh bằng 1 model hoặc list model mà user nhập vào ví dụ mode=custom,gemini-3-pro,mai-image-25 thì code sẽ chạy tạo ảnh cùng 1 prompt cho 2 model gemini-3-pro,mai-image-25
 --mode=all thì sẽ chạy tạo ảnh cùng request.json với từng model trong nhóm "quality" và "eco"

Thêm một mode nữa là 
 --mode=clone,path-to-select-image,select-model: code sẽ lấy thiết kế ở path-to-select-image và chạy với select image model để tạo ra tất cả các size đang có trong output.clone của request.json, lưu lại theo cây thư mục như [2]
 
## [5] Biến môi trường dùng cho app
AZ_GPI_5_MINI_ENDPOINT
AZ_GPI_5_MINI_KEY

GCP_GEMINI_31_FLASH_IMAGE_ENDPOINT
GCP_GEMINI_3_PRO_IMAGE_ENDPOINT
GCP_VERTEX_API_KEY

AZ_MAI_IMAGE_25_ENDPOINT
AZ_MAI_IMAGE_25_KEY

AZ_FLUX_2_PRO_ENDPOINT
AZ_FLUX_2_PRO_KEY

AZ_MAI_IMAGE_25_FLASH_ENDPOINT
AZ_MAI_IMAGE_25_FLASH_KEY

AZ_GPT_IMAGE_2_ENDPOINT
AZ_GPT_IMAGE_2_KEY

AZ_GPI_5_MINI_ENDPOINT
AZ_GPI_5_MINI_KEY

## [6] APP API

App được deploy sử dung FastAPI (uvicorn). Bạn chú ý xem xét toàn bộ dự án trên code app version 1 bẳng workflow sử dụng LangGraph nhé.
Webapp như streamlit sử dụng app thông qua gọi RESTful API. Code streamlit tôi đã viết sẵn.

IMPORTANT NOTE:
Trên toàn bộ app bạn thực hiện lưu log và in ra terminal tiến trình các bước xứ lý đang ở đâu.
Với các exception, dự đoán trước và code bắt toàn bộ, không để ERROR. Thay bằng dump exception.





