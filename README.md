# porforlio_2526
This is my portfolio repository performing my competences at period of 2025-2026. 


## Portfolio Summary

This portfolio showcases three end-to-end AI & Machine Learning demo projects that demonstrate my ability to design, build, and deploy production-oriented data and AI systems, covering the full lifecycle from experimentation to cloud deployment.

### Covered Domains & Skills

#### LLM & Agentic AI Systems

- Agentic chatbot architecture (LangChain)
- Retrieval-Augmented Generation (RAG)
- Vector search (FAISS)
- LLM integration and tool-based reasoning

#### Machine Learning & Modeling

- End-to-end ML workflows: data preparation, feature engineering, model training, and evaluation
- Classical ML models and scorecard-style prediction systems
- Model inference via REST APIs

#### MLOps & Workflow Orchestration

- ML pipeline orchestration (Amazon SageMaker Pipelines, Prefect)
- Experiment tracking and model versioning (MLflow)
- Reproducible and modular ML project structure

#### Cloud, Infrastructure & Deployment

- Containerized ML/AI applications using Docker
- Infrastructure as Code (Terraform)
- CI/CD pipelines for automated build and deployment
- Deployment on cloud platforms (Azure, AWS)



## Project Highlights

### Project 01 – Agentic Chatbot (LLM + Cloud Deployment)

- Designed and implemented an agentic LLM chatbot using LangChain, supporting Retrieval-Augmented Generation (RAG) with FAISS vector store and embedding-based similarity search.

- Built a production-ready FastAPI inference service, containerized the application using Docker, and managed container images via Azure Container Registry (ACR).

- Provisioned and deployed the entire cloud infrastructure using Terraform (Infrastructure as Code), and implemented CI/CD pipelines with Azure DevOps Pipelines to automate container build, registry push, and deployment to Azure Container Apps (ACA).

### Project 02 – Training ML Pipeline with Amazon SageMaker

- Implemented an end-to-end machine learning pipeline using Amazon SageMaker Pipelines, modeling the workflow as a DAG composed of multiple ProcessingSteps.

- Utilized ScriptProcessor for data ingestion, preprocessing, and feature engineering, with all intermediate outputs and model artifacts persisted to Amazon S3.

- Demonstrated best practices for modular ML pipeline design, reproducible execution, and separation of data processing, training logic, and artifact management in a managed cloud ML platform.

### Project 03 – ML Application with Progressive MLOps Architecture

- Developed an end-to-end machine learning application that evolves from a standalone Python training script to an orchestrated ML workflow using Prefect for task and flow management.

- Integrated MLflow for experiment tracking, model versioning, and artifact management, enabling reproducible training and controlled model promotion.

- Exposed trained models through a RESTful inference API using FastAPI, supporting real-time prediction requests and demonstrating the transition from experimental ML code to a production-oriented MLOps workflow.

### Project 06 – Banner Generation (Multi-model Image GenAI + LangGraph Pipeline)

- Built a Dynamic Creative Optimization (DCO) system that turns a single `request.json` (product, audience, assets, text, banner format) into multiple banner designs using Image GenAI models (Gemini, GPT-Image-2, MAI-Image, FLUX) behind one provider registry.

- Orchestrated the prompt pipeline with LangGraph (request, assets, design guideline, image preparation, prompt), with a swappable text-LLM layer (OpenAI / Claude / Llama / DeepSeek / Ollama) and a generate → validate → repair loop with safe-area aware cropping.

- Delivered the system as a FastAPI service and a Streamlit review/release/clone webapp, containerized with Docker, and covered by a pytest evaluation suite (unit / smoke / stress) with a documented change log of real-run findings.

Details: [`06_banner_generation_Nov2026`](06_banner_generation_Nov2026/README.md)

# END
Xây dựng Voice Bot tiếng Việt: kiến trúc, công nghệ và phương pháp đánh giá
Essay kỹ thuật — cập nhật tháng 10/2026. Phạm vi: toàn bộ đường đi của một câu nói, từ micro của khách hàng, qua nhận dạng giọng nói (STT), xử lý ở backend, đến tổng hợp giọng nói (TTS) trả ngược về thiết bị; kèm phương pháp đánh giá để biết hệ thống có "ổn" và "có chất lượng" hay không.

Tóm tắt nhanh
Kiến trúc: nối tầng STT → LLM → TTS là mặc định hợp lý cho tiếng Việt; speech-to-speech là đường tắt, nhưng bằng chứng tiếng Việt còn mỏng.
Truyền dữ liệu: gửi âm thanh lên máy chủ. WebRTC cho trình duyệt/app, SIP cho điện thoại; WebSocket/gRPC giữa các dịch vụ. Kafka không phải kênh âm thanh, nó thuộc lớp sự kiện (bản chữ, số đo, kiểm toán, QA).
Trước STT: AEC/khử nhiễu ở thiết bị, VAD (Silero là điểm khởi đầu) và phát hiện hết lượt (Smart Turn v3 hỗ trợ tiếng Việt; bộ của LiveKit và Deepgram Flux thì không).
STT: không có bảng xếp hạng trung lập bao cả model mở và thương mại. Top 5 model mở có số đo: PhoWhisper-medium, Zipformer-30M, PhoWhisper-small, Qwen3-ASR-1.7B, parakeet-ctc-0.6b-Vietnamese. Số STT thương mại là do nhà cung cấp tự công bố.
LLM: top 5 model mở theo SEA-HELM tiếng Việt là Gemma 4 31B, Qwen 3.5 122B-A10B, Qwen 3.6 27B, SEA-LION v4.5 27B, Muse Glimmer 30B; model đóng chưa có bảng tiếng Việt trung lập.
TTS và "nhả chữ": cắt mệnh đề đầu rồi cắt theo câu, chuẩn hóa số/ngày/tiền trước khi đọc, đưa chữ lên giao diện theo audio đã phát, và cắt ngữ cảnh đúng chỗ khi bị ngắt lời. VieNeu-TTS là lựa chọn mở có streaming; chưa có MOS/WER công bố.
Đánh giá: theo tầng và theo lát cắt (giọng vùng miền, nhiễu, điện thoại 8 kHz, số/thực thể), WER tách dấu thanh, kiểm thử turn-taking, độ trễ phân vị đo ở client, mô phỏng bot-to-bot với pass^k, tải/soak/chaos, cổng hồi quy và giám sát sản xuất.
Lưu ý về chất lượng bằng chứng: mỗi số liệu chính được gắn nhãn 🟢 (kiểm chứng được) / 🟡 (nhà cung cấp tự công bố) / ⚪ (suy luận hoặc kiến thức chung). Mục 11.2 liệt kê những gì mình không đọc được.
0. Lời mở đầu và cách đọc số liệu
Một voice bot trông đơn giản: người dùng nói, bot trả lời bằng giọng nói. Nhưng "nói – nghe – trả lời" là một chuỗi gồm ít nhất sáu thành phần phải khớp nhau trong chưa đầy một giây: thu âm và lọc nhiễu, phát hiện tiếng nói, quyết định người dùng đã nói xong chưa, nhận dạng thành chữ, suy nghĩ để tạo câu trả lời, rồi đọc câu trả lời ra loa trong khi vẫn sẵn sàng bị ngắt lời. Mỗi thành phần có hàng chục lựa chọn, và với tiếng Việt, nhiều lựa chọn "tốt nhất thế giới" không hề tốt nhất — thậm chí không hỗ trợ — tiếng Việt.

Bài viết này trả lời lần lượt các câu hỏi của bạn:

Âm thanh được thu và xử lý ở thiết bị như thế nào, cần những model phát hiện giọng nói nào ngoài STT (mục 2).
Truyền về backend là âm thanh hay chữ, và dùng Kafka hay công nghệ gì để streaming (mục 3).
Top model STT cho tiếng Việt cùng các số liệu benchmark (mục 4).
Backend xử lý văn bản ra sao và top model cho khâu này (mục 5), cùng phương án speech-to-speech thay thế (mục 6).
Trả kết quả ngược về thiết bị: TTS, streaming, "nhả chữ", top model TTS tiếng Việt (mục 7).
Triển khai ở Việt Nam: vùng đặt máy chủ và dữ liệu cá nhân (mục 8).
Phương pháp đánh giá chất lượng và độ ổn định của cả hệ thống (mục 9).
Áp dụng vào dự án demo hiện có của bạn (mục 10).
Kết luận và những điều mình chưa kiểm chứng được (mục 11), kèm danh sách nguồn.
Ba điều cần biết trước khi tin vào bất kỳ con số nào
Thứ nhất, hầu hết benchmark công khai không phải benchmark cho voice bot tiếng Việt. Bảng xếp hạng STT phổ biến nhất của Hugging Face (Open ASR Leaderboard) chỉ có tiếng Anh và năm ngôn ngữ châu Âu (Đức, Pháp, Ý, Tây Ban Nha, Bồ Đào Nha); tiếng Việt không có trong đó [S15]. Bảng của Artificial Analysis cho STT dùng dữ liệu hội thoại, bài phát biểu nghị viện châu Âu và báo cáo tài chính — không nói rõ có tiếng Việt [S16]. Vì vậy mình dựa vào những nguồn có số liệu tiếng Việt thật: Vietnamese Open ASR Leaderboard [S1], bài báo PhoWhisper [S2], VietASR [S4], báo cáo kỹ thuật Qwen3-ASR [S5], và bảng so sánh tiếng Việt của ElevenLabs [S8].

Thứ hai, cùng một model có thể ra các con số rất khác nhau tùy cách đo. Whisper large-v3 trên FLEURS tiếng Việt được báo 8,59% [S4], 9,1% [S10] và 10,7% [S8] — ba nhóm đo khác nhau, cùng model. Trên Vietnamese Open ASR Leaderboard, mỗi model có hai cột (O-WER và N-WER) chênh nhau khoảng 1,6–2,6 lần (PhoWhisper-medium: 16,73 và 7,61) [S1]; trang không định nghĩa hai cột này, nên mình chỉ ghi nhận hiện tượng, không diễn giải. Nguyên nhân thường gặp trong ngành ⚪ là cách chuẩn hóa văn bản trước khi tính lỗi: hoa/thường, dấu câu, cách viết số ("25" hay "hai mươi lăm"), chữ không dấu. Kết luận thực tế: chỉ so sánh các số được đo trong cùng một bảng.

Thứ ba, mình dán nhãn độ tin cậy cho mỗi số liệu quan trọng:

Nhãn	Ý nghĩa
🟢	Từ bảng xếp hạng/bài báo/tài liệu kỹ thuật có thể kiểm chứng, không do nhà bán sản phẩm tự công bố
🟡	Do chính nhà cung cấp hoặc tác giả model tự công bố (thiên lệch có thể có)
⚪	Kiến thức chung của người viết hoặc suy luận, không tra nguồn trong lần nghiên cứu này
Có hai bảng xếp hạng liên quan trực tiếp mà mình không đọc được nội dung: Voice Arena (bảng STT tiếng Việt) [S46] và Multilingual TTS Arena của Artificial Analysis (có tiếng Việt) [S45]. Cả hai là trang tải bằng JavaScript hoặc chặn truy cập tự động. Bạn nên mở trực tiếp hai trang này; chúng có thể là nguồn số liệu tiếng Việt tốt nhất hiện có và mình không muốn bịa số thay chúng.

1. Bức tranh tổng thể
1.1 Hai họ kiến trúc
Kiến trúc cascaded (nối tầng): STT → LLM/logic nghiệp vụ → TTS, mỗi khâu là một dịch vụ riêng. Ưu điểm: chọn được model tốt nhất cho từng khâu (đặc biệt quan trọng với tiếng Việt), tự host từng phần, tuân thủ quy định dữ liệu, quan sát và đo lường từng tầng. Nhược điểm: tích lũy độ trễ qua nhiều bước, mất thông tin giọng điệu (cảm xúc, ngập ngừng) khi chuyển thành chữ.

Kiến trúc speech-to-speech (một model nghe và nói trực tiếp): như các API realtime của OpenAI và Google. Ưu điểm: đơn giản, độ trễ thấp, xử lý ngắt lời tự nhiên. Nhược điểm: ít kiểm soát, khó kiểm toán (không có bản ghi chữ trung gian để kiểm tra), ít lựa chọn giọng đọc tiếng Việt, và bạn phụ thuộc hoàn toàn vào chất lượng tiếng Việt của một nhà cung cấp. Báo cáo WildASR còn lập luận rằng ngay cả khi speech-to-speech phổ biến, lớp ASR tường minh vẫn có giá trị như "lớp có thể kiểm toán và chốt chặn an toàn" cho agent thực hiện hành động [S17].

Nhà cung cấp Deepgram mô tả sự đánh đổi này tương tự: API gộp qua một WebSocket giảm số kết nối và các lỗi tranh chấp giữa các bước; còn ghép nhiều dịch vụ cho phép chọn model rộng hơn, tự host, cách ly tuân thủ và quan sát từng khâu [S25] 🟡.

1.2 Luồng đầu–cuối của kiến trúc cascaded
 THIẾT BỊ / TRÌNH DUYỆT / TỔNG ĐÀI                       BACKEND
┌────────────────────────────┐          ┌────────────────────────────────────────────────┐
│ Micro                      │          │  Media gateway (SFU / SBC)                     │
│  → AEC + NS + AGC          │  audio   │   ↓ audio frame 20 ms                          │
│  → VAD (gác cổng, tùy chọn)│ ───────► │  VAD + Turn detection                          │
│  → mã hóa Opus (hoặc G.711)│ WebRTC   │   ↓ audio                                      │
│                            │ /SIP/WS  │  STT streaming ──► partial / final transcript  │
│                            │          │   ↓ text (khi hết lượt)                        │
│ Loa ◄── giải mã + jitter   │          │  LLM / Dialog manager (+ RAG, công cụ nghiệp vụ)│
│         buffer + phát      │ ◄─────── │   ↓ token stream                               │
│  (cắt phát khi bị ngắt lời)│  audio   │  Chunker (cắt câu) + Chuẩn hóa văn bản tiếng Việt│
└────────────────────────────┘          │   ↓ text từng câu                              │
                                        │  TTS streaming ──► audio frame                 │
                                        │                                                 │
                                        │  ──── mặt phẳng sự kiện (ngoài đường nóng) ────│
                                        │  Kafka/Redpanda/NATS: transcript, metrics, log, │
                                        │  ghi âm, QA, đánh giá, CRM                      │
                                        └────────────────────────────────────────────────┘

Điểm mấu chốt của sơ đồ: có hai mặt phẳng. Mặt phẳng media (âm thanh thời gian thực) đi qua WebRTC/SIP/WebSocket và nhạy cảm với từng chục mili-giây. Mặt phẳng sự kiện (bản ghi chữ, số liệu, kiểm toán) đi qua hàng đợi như Kafka và chịu được độ trễ cao hơn. Nhầm lẫn giữa hai mặt phẳng này là lỗi kiến trúc phổ biến nhất (xem mục 3.4).

1.3 Ngân sách độ trễ
Người gọi bắt đầu cảm thấy trễ ở khoảng 800 ms, và phần lớn thời gian nằm ở chốt cuối lượt của STT và thời gian ra token đầu tiên của LLM [S25] 🟡. Hamming (hãng bán công cụ kiểm thử voice agent) đưa ra các ngưỡng tham chiếu từ dữ liệu sản xuất của họ [S42] 🟡:

Thành phần	Mục tiêu	Mức nghiêm trọng
STT	< 300 ms	> 500 ms
LLM — thời gian ra token đầu tiên (TTFT)	< 400 ms	> 600 ms
TTS — đến âm thanh đầu tiên	< 200 ms	> 400 ms
Thời gian đến từ đầu tiên (TTFW)	< 400 ms	—
Độ trễ lượt, P50 / P90 / P95 (đã gồm mạng, điện thoại)	< 1,5 s / < 2,5 s / < 3,5 s	> 1,7 s / > 3,0 s / > 5,0 s
Trang này tự mâu thuẫn ở vài chỗ (chỗ khác nói P95 dưới 5 giây là đủ), nên hãy coi đây là khung tham chiếu để bạn tự đặt SLO, không phải chuẩn ngành. Cộng ba thành phần đạt mục tiêu (300 + 400 + 200 = 900 ms) cho thấy ngân sách rất sát — vì vậy streaming ở mọi khâu và gối đầu giữa các khâu là bắt buộc, không phải tùy chọn.

Còn một yếu tố hay bị quên: vị trí máy chủ. Bài của LiveKit nêu ví dụ Singapore – US-East có RTT thực tế khoảng 230–280 ms trước khi tính xử lý [S23] 🟡. Với người dùng ở Việt Nam, đặt toàn bộ chuỗi STT/LLM/TTS ở châu Mỹ sẽ tốn gần hết ngân sách chỉ cho đường truyền (xem mục 8).

2. Từ micro đến backend: thu âm, tiền xử lý và các model "phát hiện" giọng nói
2.1 Thu âm
Định dạng chuẩn cho STT: PCM 16-bit, mono, 16 kHz. Gemini Live yêu cầu đúng định dạng này cho đầu vào và trả đầu ra 24 kHz [S29] 🟢; ElevenLabs Scribe nhận PCM 8–48 kHz và mu-law 8 kHz cho điện thoại, khuyến nghị gói khoảng 100 ms (cho phép 20–250 ms) [S9] 🟡.
Khung (frame) 20–30 ms: đơn vị xử lý của VAD và codec; dự án demo của bạn dùng 30 ms.
Băng thông: PCM 16 kHz/16-bit/mono là 32 000 byte/giây (256 kbps). Opus thường dùng một phần nhỏ băng thông này ⚪, nên với mạng di động nên nén bằng Opus (WebRTC làm sẵn) thay vì gửi PCM thô.
Điện thoại: tổng đài thường đưa audio G.711 8 kHz. Các model tiếng Việt được huấn luyện phần lớn trên âm thanh băng rộng ⚪, nên giảm xuống 8 kHz/điện thoại là một trục suy giảm cần đo riêng; WildASR liệt kê "phone codec" là một trong các điều kiện làm lệch kết quả giữa các model [S17] 🟢.
2.2 Tiền xử lý âm thanh
Ba bộ lọc cổ điển — khử vọng (AEC), khử nhiễu (NS), tự chỉnh âm lượng (AGC) — đã có sẵn trong đường ống media của WebRTC, chạy trước khi âm thanh lên mạng [S23, S24] 🟢. Trong số ba bộ lọc, AEC quan trọng nhất cho voice bot: nếu micro thu lại tiếng loa phát giọng bot, STT sẽ nghe thấy chính bot nói, và tính năng cho người dùng ngắt lời (barge-in) sẽ tự kích hoạt sai. Bài của LiveKit lưu ý AEC phải bám theo âm thanh vừa phát ra [S23] 🟡.

Khử nhiễu bằng mạng nơ-ron (RNNoise, DeepFilterNet, Krisp, các dịch vụ như LiveKit enhanced noise cancellation) giúp ở môi trường ồn ⚪, nhưng khử nhiễu quá mạnh có thể làm méo âm vị và làm tăng WER ⚪. Cách làm đúng: A/B test bật/tắt trên bộ dữ liệu có nhiễu thật của bạn (mục 9), không bật mặc định theo cảm tính.

2.3 Phát hiện hoạt động giọng nói (VAD)
VAD trả lời câu hỏi "khung âm thanh này có tiếng người không". Nó có ba vai trò: (1) cắt bỏ khoảng lặng để tiết kiệm băng thông và tiền (STT tính theo giờ audio), (2) cho biết người dùng bắt đầu nói (để kích hoạt ngắt lời), (3) làm tín hiệu đầu vào cho phát hiện hết lượt.

Năm lựa chọn VAD đáng cân nhắc:

#	Lựa chọn	Đặc điểm đã xác minh	Ghi chú cho tiếng Việt
1	Silero VAD	Huấn luyện trên kho dữ liệu hơn 6000 ngôn ngữ; một khối 30+ ms xử lý < 1 ms trên một luồng CPU; mô hình JIT khoảng 2 MB; hỗ trợ 8 kHz và 16 kHz; giấy phép MIT (trang có một huy hiệu ghi CC BY-NC 4.0 mâu thuẫn — hãy kiểm tra giấy phép trong repo trước khi dùng thương mại) [S20] 🟢	LiveKit dùng Silero làm VAD mặc định [S22] (Pipecat cũng hỗ trợ ⚪); VAD mang tính âm học nên không cần bản riêng cho tiếng Việt, nhưng vẫn phải chỉnh ngưỡng trên dữ liệu ồn của bạn
2	TEN VAD	Thư viện 306 KB (Linux) so với 2,16–2,22 MB của Silero; RTF 0,0086–0,0150 trên CPU x86 (Silero 0,0127 trên Xeon Gold 6348); hop 10 ms hoặc 16 ms; Apache 2.0 kèm điều kiện bổ sung [S19] 🟡	Tác giả nói TEN "chính xác hơn Silero và WebRTC VAD" và Silero trễ "vài trăm mili-giây" khi chuyển từ nói sang im, nhưng chỉ đưa đường cong precision-recall trên bộ test của chính họ, không công bố số — cần tự kiểm chứng
3	WebRTC VAD	VAD cổ điển, rất nhẹ, 4 mức "khắt khe"; chỉ nhận mono 8/16/32/48 kHz, khung 10/20/30 ms (đúng như code demo của bạn)	Dễ bắt tiếng ồn tạp làm tiếng nói; dùng được cho prototype, thường được thay bằng Silero/TEN ở production ⚪
4	VAD của nhà cung cấp API realtime	Gemini Live bật VAD tự động theo mặc định; khi phát hiện ngắt lời, phần sinh đang chạy bị hủy [S29] 🟢; ElevenLabs Scribe Realtime phân đoạn theo khoảng lặng, khuyến nghị 200–400 ms cho agent phản hồi nhanh và 800–1200 ms cho đọc chính tả [S9] 🟡	Tiện, nhưng bạn không kiểm soát được thuật toán và độ trễ
5	Cobra (Picovoice)	Thương mại (Picovoice); chưa kiểm chứng trong lần nghiên cứu này ⚪	Cân nhắc nếu cần chạy on-device với hợp đồng hỗ trợ
Kết luận thực dụng: bắt đầu bằng Silero VAD; đo thử TEN VAD trên dữ liệu của bạn nếu cần chạy trên thiết bị yếu hoặc trình duyệt (bản web 277 KB) [S19].

2.4 Phát hiện hết lượt (turn detection / end-of-utterance)
VAD chỉ biết "đang có tiếng/không có tiếng"; nó không biết người dùng đã nói xong ý hay chỉ ngập ngừng. Dùng độc lập VAD với ngưỡng im lặng cố định gây hai loại lỗi đối nghịch: cắt ngang người đang suy nghĩ (đặt ngưỡng thấp) hoặc để người dùng chờ dài (đặt ngưỡng cao).

Các giải pháp hiện có:

Giải pháp	Hỗ trợ tiếng Việt?	Ghi chú
Smart Turn v3 (Pipecat)	Có — hỗ trợ 23 ngôn ngữ, trong đó có tiếng Việt [S21] 🟢	Khoảng 8 triệu tham số; bản CPU 8 MB (lượng tử hóa int8), bản GPU 32 MB; "dưới 100 ms trên đa số máy cloud", khoảng 65 ms trên Pipecat Cloud; không công bố độ chính xác theo từng ngôn ngữ → phải tự đo
Bộ phát hiện lượt của LiveKit	Không — mô hình âm thanh v1 hỗ trợ 14 ngôn ngữ (Anh, Ả Rập, Đức, Tây Ban Nha, Pháp, Hindi, Indonesia, Ý, Nhật, Hàn, Hà Lan, Bồ Đào Nha, Thổ Nhĩ Kỳ, Trung); mô hình văn bản (đã bị khai tử) cũng không có tiếng Việt [S22] 🟢	Với LiveKit và tiếng Việt, bạn phải dùng endpointing của STT hoặc VAD + ngưỡng thời gian
Endpointing của STT (ví dụ Deepgram Flux)	Flux không hỗ trợ tiếng Việt [S14] 🟢	Nova-2 và Nova-3 có tiếng Việt nhưng không phải biến thể Flux
Ngưỡng im lặng tự chỉnh (200–400 ms cho agent, 800–1200 ms cho đọc chính tả) [S9]	Mọi ngôn ngữ	Đơn giản nhất; điều chỉnh theo từng bước hội thoại
Mô hình ngữ nghĩa tự huấn luyện	Tự làm	Dùng LLM nhỏ đánh giá "câu đã hoàn chỉnh chưa" trên bản ghi chữ một phần
Có hai đặc thù tiếng Việt nên lưu ý (suy luận từ cách người Việt nói, ⚪): thứ nhất, trợ từ cuối câu như "ạ, nhé, nhỉ, à" và các tiếng đệm "ờ, ừm, thì…" khiến một câu dễ bị cắt non hoặc kéo dài; thứ hai, khi người dùng đọc số điện thoại, số CCCD, số tài khoản, họ dừng giữa các nhóm số — ngưỡng im lặng cho bước "thu thập chuỗi số" phải dài hơn bước hội thoại thường. Nên cấu hình ngưỡng theo ngữ cảnh của bước hội thoại, không dùng một giá trị toàn cục. Dự án demo của bạn dùng 3 giây im lặng — phù hợp để thử nghiệm, nhưng quá chậm cho hội thoại thật.

Một kỹ thuật giảm độ trễ cảm nhận đáng biết: suy luận gối đầu (speculative prefill) — bắt đầu chạy LLM ngay khi có tín hiệu hết lượt sớm thay vì chờ chốt hẳn [S25] 🟡.

2.5 Các model "phát hiện giọng nói" khác thường cần trong giai đoạn này ⚪
Phần này là kiến thức chung, chưa đối chiếu nguồn trong lần nghiên cứu này, nhưng đây là những khối bạn sẽ gặp khi làm sản phẩm thật:

Nhu cầu	Mô hình/kỹ thuật thường dùng	Khi nào cần
Từ khóa đánh thức (wake word)	openWakeWord, Picovoice Porcupine	Thiết bị luôn bật, loa thông minh, kiosk
Nhận dạng/xác thực người nói	Mô hình nhúng người nói (ECAPA-TDNN, TitaNet)	Xác thực bằng giọng nói trong ngân hàng
Phân tách người nói (diarization)	pyannote, NeMo	Cuộc gọi có nhiều người, ghi âm kiểm soát chất lượng
Nhận dạng ngôn ngữ / trộn mã	Language-ID của Whisper, của nhà cung cấp STT	Khách trộn tiếng Anh–Việt (rất phổ biến trong ngân hàng, công nghệ)
Chống giả mạo giọng nói (anti-spoofing/deepfake)	Mô hình phát hiện voice cloning và replay	Bắt buộc cân nhắc nếu cho phép thao tác tài chính bằng giọng nói
Phát hiện hộp thư thoại/DTMF	Answering-machine detection	Cuộc gọi chủ động ra ngoài (outbound)
Phân tích cảm xúc/giọng điệu	Mô hình cảm xúc giọng nói	Chuyển người thật khi khách bực bội
3. Truyền dữ liệu về backend: âm thanh hay chữ, và công nghệ streaming nào
3.1 Truyền chữ hay truyền âm thanh?
Câu trả lời ngắn: mặc định nên truyền âm thanh (đã nén) lên máy chủ, và chạy STT ở máy chủ. Truyền chữ chỉ hợp lý khi bạn chấp nhận chạy STT trên thiết bị.

Phương án	Ưu điểm	Nhược điểm	Phù hợp
A. STT trên thiết bị, chỉ gửi chữ	Băng thông gần bằng không; âm thanh không rời thiết bị (riêng tư); chạy được ngoại tuyến	Phải dùng model nhỏ (độ chính xác thấp hơn); phải cập nhật model trên hàng nghìn thiết bị; tốn pin/CPU; không có âm thanh gốc để kiểm tra chất lượng và chạy lại STT khi model tốt hơn ra đời; khó xử lý điện thoại	Ứng dụng yêu cầu riêng tư cao hoặc ngoại tuyến; thiết bị đủ mạnh
B. Gửi âm thanh lên máy chủ (khuyến nghị mặc định)	Dùng được model lớn nhất; nâng cấp model không cần đụng thiết bị; giữ được âm thanh để QA, đánh giá và huấn luyện lại (nếu pháp lý cho phép); hợp với điện thoại	Tốn băng thông; phụ thuộc mạng; phải xử lý dữ liệu giọng nói (là dữ liệu cá nhân)	Hầu hết voice bot dịch vụ khách hàng
C. Lai: lọc ở thiết bị, STT ở máy chủ	Thiết bị làm AEC, khử nhiễu, VAD, wake word; chỉ gửi khi có tiếng nói → tiết kiệm băng thông và chi phí STT	Phức tạp hơn một chút	Ứng dụng di động, kiosk
D. Speech-to-speech	Không cần ghép STT/LLM/TTS	Xem mục 6	Khi chấp nhận phụ thuộc nhà cung cấp
Về chọn model on-device cho tiếng Việt: các model nhỏ có số liệu tiếng Việt gồm Zipformer-30M-RNNT-6000h (30 triệu tham số, N-WER trung bình 6,77 — thấp nhất trong 7 model của bảng) và Qwen3-ASR-0.6B [S1] 🟢. Cả hai chưa được xác minh chạy trên điện thoại trong lần nghiên cứu này.

3.2 Giao thức streaming âm thanh
Giao thức	Cơ chế	Điểm mạnh	Điểm yếu	Dùng ở đâu
WebRTC	UDP/RTP, bộ đệm jitter thích ứng, điều khiển nghẽn theo độ trễ (GCC), Opus, xuyên NAT (ICE/STUN/TURN), AEC/NS/AGC có sẵn	Chịu mất gói tốt; "mất một khung 20 ms gần như không nghe thấy"; chuẩn của trình duyệt	Phức tạp hơn để vận hành (cần SFU hoặc media server)	Trình duyệt và app di động ↔ backend
WebSocket	TCP, hai chiều	Đơn giản, mọi ngôn ngữ đều có; ElevenLabs khuyến nghị cho pipeline phía máy chủ [S9]	Chặn đầu hàng đợi của TCP: một gói mất làm cả luồng chờ "có khi hàng trăm mili-giây", âm thanh đến sau phát dồn một cục; tự xây bộ đệm jitter; không có AEC/NS	Server ↔ server, dịch vụ STT/TTS; chấp nhận được cho mạng tốt
gRPC streaming hai chiều	HTTP/2 trên TCP	Hợp đồng kiểu mạnh (protobuf), hiệu quả nội bộ	Cùng hạn chế TCP; không dành cho trình duyệt trực tiếp	Service nội bộ
SIP + RTP	Chuẩn viễn thông	Kết nối nhà mạng, tổng đài, đầu số	G.711 8 kHz; cần SBC/media server	Cuộc gọi điện thoại
Các luận điểm về TCP chặn đầu hàng đợi, bộ đệm jitter và quy tắc "khung rơi tốt hơn khung trễ" lấy từ bài của LiveKit [S23] — đây là nhà bán nền tảng WebRTC nên có lợi ích; tuy vậy các cơ chế kỹ thuật là chuẩn, và OpenAI cũng chọn WebRTC làm giao thức phía client cho dịch vụ giọng nói quy mô hàng trăm triệu người dùng hằng tuần, với kiến trúc "transceiver" thay vì SFU vì phần lớn phiên là 1–1 [S24] 🟢. Tài liệu Realtime của OpenAI nói rõ phiên kết nối "qua WebRTC trong trình duyệt hoặc WebSocket trên máy chủ" [S28] 🟢.

Quy tắc chọn: âm thanh sống (người nói chuyện thật) đi qua WebRTC hoặc SIP; mọi thứ còn lại (giữa các dịch vụ trong datacenter) có thể dùng WebSocket/gRPC. Chỉ dùng WebSocket cho client nếu mạng ổn định (mạng văn phòng, LAN) hoặc khi chấp nhận kém chất lượng.

3.3 Kiến trúc triển khai tham chiếu
 Trình duyệt / App ──WebRTC──►┐
                               ├─► SFU / Media gateway ─► Agent worker (Pipecat | LiveKit Agents)
 Điện thoại ──SIP trunk──────►┘     (LiveKit, Daily,              │  gọi STT / LLM / TTS qua WebSocket,
                                     Pion tự dựng)                │  gRPC hoặc SDK; giữ trạng thái phiên
                                                                  ▼
                                      Redis (trạng thái phiên, định tuyến "dính" một cuộc gọi → một worker)
                                                                  │
                                                                  └─► sự kiện ──► Kafka/Redpanda/NATS JetStream

Ba nguyên tắc rút ra từ kinh nghiệm công bố của OpenAI [S24] 🟢: (1) giao thức có trạng thái (ICE, DTLS) đòi hỏi một cuộc gọi luôn được định tuyến cố định về cùng một tiến trình; (2) không dùng một cổng UDP cho mỗi phiên — khó mở, bảo mật và tự mở rộng trên Kubernetes; (3) giữ độ phức tạp ở lớp định tuyến mỏng, không rải khắp các dịch vụ backend.

Hai khung phần mềm phổ biến để điều phối agent (Python):

Pipecat	LiveKit Agents
Truyền tải	Daily, LiveKit, Twilio, local, SmallWebRTC (chất lượng hỗ trợ không đồng đều)	LiveKit, tích hợp chặt
Điện thoại/SIP	Qua cầu nối Daily; số điện thoại từ bên thứ ba	SIP có sẵn
Phát hiện hết lượt	Smart Turn v3 (có tiếng Việt)	Mô hình riêng (không có tiếng Việt)
Phù hợp	Muốn kiểm soát chi tiết, "chỉ là code Python", flow editor	Muốn ít code hơn, có sẵn quan sát/kiểm thử
Nguồn so sánh [S26] do LiveKit xuất bản nên thiên về LiveKit 🟡. Dữ kiện hỗ trợ tiếng Việt của hai bộ phát hiện hết lượt ở trên [S21, S22] là điểm khác biệt có thể quyết định lựa chọn của bạn.

3.4 Kafka có phải là câu trả lời không?
Không cho luồng âm thanh, nhưng rất có chỗ đứng trong hệ thống. Cần phân biệt rõ, vì lý do không phải "Kafka chậm":

Độ trễ không phải vấn đề chính. Confluent công bố p99 độ trễ đầu–cuối 5 ms ở tải 200 MB/s trên cụm ba broker được tinh chỉnh (batch tối đa 1 ms) [S27] 🟡. Kafka đủ nhanh cho khung 20 ms về mặt trung bình trong cụm khỏe.
Kafka không phải giao thức đầu cuối. Trình duyệt và điện thoại không nói chuyện trực tiếp với Kafka; bạn vẫn cần WebRTC/WebSocket để vào hệ thống. Thêm Kafka giữa gateway và STT chỉ thêm một chặng.
Triết lý đối nghịch. Kafka đảm bảo bền vững và đúng thứ tự — khung âm thanh đến muộn thì vô giá trị (khung rơi tốt hơn khung trễ [S23]), không cần ghi đĩa và phát lại. Kafka không có bộ đệm jitter, đồng hồ media, hay che mất gói.
Nhịp tải và độ trễ đuôi. Cân bằng lại consumer group, hot partition (khi dùng khóa là mã cuộc gọi) và broker chậm gây đỉnh trễ trăm mili-giây đến giây ⚪ — trái với yêu cầu p95 của hội thoại.
Cuộc gọi là luồng tạm thời. Mỗi cuộc gọi vài phút rồi biến mất; tạo hàng nghìn topic/phân vùng cho luồng ngắn hạn là cách dùng Kafka sai mục đích ⚪.
Ngắt lời đòi hỏi hủy tức thì. Khi người dùng nói chen vào, bạn phải xóa ngay hàng đợi âm thanh chưa phát; hàng đợi bền vững như Kafka làm việc này phức tạp hơn.
Kafka (hoặc Redpanda, Pulsar, NATS JetStream) nên đặt ở đâu — mặt phẳng sự kiện:

Sự kiện	Vì sao dùng hàng đợi
Bản ghi chữ từng lượt (partial/final)	Phân tích, tìm kiếm, kiểm soát chất lượng
Kết quả gọi công cụ nghiệp vụ, thay đổi trạng thái hội thoại	Kiểm toán, tái hiện lỗi
Số đo từng lượt: STT/LLM/TTS latency, tỷ lệ ngắt lời	Bảng điều khiển và cảnh báo (mục 9)
Siêu dữ liệu bản ghi âm (đường dẫn lưu trữ)	Truyền sang quy trình lưu trữ, xóa theo thời hạn
Sự kiện sau cuộc gọi: tóm tắt, cập nhật CRM, chấm điểm bằng LLM	Xử lý bất đồng bộ, thử lại an toàn
Tóm lại: âm thanh ở mặt phẳng media, sự kiện ở mặt phẳng message. Nếu bạn chỉ cần quy mô vài chục cuộc gọi đồng thời, có thể bắt đầu không cần Kafka — một Postgres cộng Redis và log có cấu trúc là đủ — rồi thêm hàng đợi khi bắt đầu cần phân tích khối lượng lớn.

4. Nhận dạng giọng nói (STT) cho tiếng Việt: top 5 và số liệu benchmark
4.1 Trước khi xem bảng: STT cho voice bot khác STT để chép băng
Một model chép băng (offline) được phép đọc cả đoạn rồi mới quyết định. Voice bot cần streaming: trả bản chữ tạm (partial) khi người dùng còn đang nói, rồi bản chữ chốt (final) khi họ nói xong. Chọn model chỉ theo WER là chưa đủ; bạn cần đồng thời nhìn bốn trục:

Độ chính xác trên giọng nói tiếng Việt thật (giọng vùng miền, tự nhiên, có nhiễu, có thể là điện thoại 8 kHz).
Khả năng streaming và độ trễ (thời gian đến chữ tạm đầu tiên, thời gian chốt cuối lượt).
Xử lý thực thể (số điện thoại, số tài khoản, tên riêng, địa chỉ, số tiền) — nguồn lỗi nghiêm trọng nhất trong hội thoại thật. Khung đánh giá EVA của ServiceNow, thử 20 hệ thống, kết luận rằng phiên âm thực thể là một trong những kiểu hỏng nổi trội vì "một ký tự nghe sai là đủ làm hỏng bước xác thực" [S43].
Chi phí/triển khai: tự host (cần GPU?) hay gọi API; giấy phép; nơi dữ liệu âm thanh được xử lý.
Hai đặc thù tiếng Việt làm bài toán khó hơn tiếng Anh:

Thanh điệu (6 thanh) và dấu. Một chữ sai dấu là một chữ khác nghĩa ("ma, má, mả, mã, mà, mạ"). Vì tiếng Việt viết tách từng âm tiết bằng dấu cách, WER thực tế được tính trên âm tiết chứ không phải từ ghép ⚪ — nên một câu 10 âm tiết sai 1 dấu là WER 10%.
Giọng vùng miền, trộn tiếng Anh, âm thanh điện thoại. Bài báo bộ dữ liệu VietSuperSpeech lập luận rằng model huấn luyện trên giọng đọc/chuẩn dễ suy giảm với lời nói đời thường (ngập ngừng, khẩu ngữ, trộn phương ngữ) nhưng không đưa số đo [S18]. Nghĩa là: chưa có số liệu công khai tin cậy cho hội thoại call-center tiếng Việt; bạn sẽ phải tự đo (mục 9).
4.2 Bảng 1 — Top 5 STT mã nguồn mở, theo bảng xếp hạng tiếng Việt có thể kiểm chứng 🟢
Nguồn: Vietnamese Open ASR Leaderboard [S1] (chỉ 7 model được xếp hạng; bảng không ghi ngày cập nhật, không ghi RTF, không định nghĩa cột O-WER và N-WER — mình sắp theo O-WER trung bình tăng dần, đúng như trên trang). Cả hai cột đều là trung bình trên 7 tập dữ liệu "tự nhiên" (viVoice, vlsp, ViMD, LSVSC, FOSD, BUD500, GigaSpeech).

Hạng	Model	O-WER TB (%)	N-WER TB (%)	Tham số	Streaming?	Ghi chú đã xác minh
1	vinai/PhoWhisper-medium	16,73	7,61	769 M [S3]	Không (họ Whisper, xử lý theo đoạn)	Tinh chỉnh từ Whisper trên 844 giờ tiếng Việt nhiều giọng; giấy phép BSD-3-Clause; trên 4 tập thử trong bài báo gốc: CMV-Vi 8,27 / VIVOS 4,97 / VLSP Task-1 14,12 / VLSP Task-2 26,85 [S2]
2	hynt/Zipformer-30M-RNNT-6000h	17,51	6,77	30 M	Chưa xác nhận	Chỉ 30 M tham số — nhỏ nhất bảng, N-WER tốt nhất bảng. Trang model không đọc được (bị chặn); cũng là "teacher" gán nhãn giả cho VietSuperSpeech [S18]
3	vinai/PhoWhisper-small	17,73	8,77	244 M	Không	Chênh lệch nhỏ so với medium, nhẹ hơn 3 lần
4	Qwen/Qwen3-ASR-1.7B	18,76	9,84	1,7 B	Có (gộp streaming/offline trong một model, đoạn 2 giây)	TTFT trung bình 102 ms (p95 113 ms) ở concurrency 1, nhưng 3 392 ms (p95 6 227 ms) ở concurrency 128 [S5]
5	nvidia/parakeet-ctc-0.6b-Vietnamese	20,78	12,81	600 M	Không xác nhận	Huấn luyện > 2 000 giờ từ 10 bộ dữ liệu, giấy phép NVIDIA Open Model [S6]
(Hạng 6–7: Qwen3-ASR-0.6B với 20,92 / 12,44; wav2vec2-base-vietnamese-250h với 24,26 / 14,89.)

Cách đọc bảng này cho đúng:

Hai cột chênh nhau rất lớn (PhoWhisper-medium: 16,73 vs 7,61, hơn gấp đôi; Zipformer-30M: 17,51 vs 6,77). Trang không định nghĩa hai cột; mình đoán cột "N" áp dụng chuẩn hóa văn bản nhẹ tay hơn nhưng đó chỉ là suy đoán ⚪. Đừng lấy cột đẹp nhất để so với con số của hãng khác.
Thứ hạng đổi theo từng tập dữ liệu. Theo O-WER: parakeet tốt nhất trên VIVOS (11,74), Common Voice (15,05) và LSVSC (14,12) nhưng tệ nhất bảng trên viVoice (27,82) và BUD500 (30,90); Zipformer-30M tốt nhất trên BUD500 (3,29) nhưng trên VIVOS chỉ 21,85 [S1]. Một model "đứng đầu trung bình" vẫn có thể thua rõ trên đúng loại âm thanh của bạn. Đây là lý do mục 9 yêu cầu tập đánh giá riêng.
Con số FLEURS 6,86 của parakeet là "in-domain". Model card ghi FLEURS nằm trong 10 bộ dữ liệu huấn luyện, và số này nằm ở nhóm test "in-domain". Phép đo công bằng hơn là nhóm "blind": trung bình 9,30 (Gigaspeech2 11,23; VLSP2021-Task2 8,99; ViMD 11,02; VIVOS 5,96) [S6].
Bảng không có Whisper large-v3, nhưng các nguồn khác đo nó: FLEURS tiếng Việt 8,59 [S4], 9,1 [S10], 10,7 [S8]; GigaSpeech 2 là 17,94 và Common Voice 13,74 [S4]. PhoWhisper-large (1,55 B) đạt 8,14 / 4,67 / 13,75 / 26,68 trên cùng bốn tập của PhoWhisper-medium (8,27 / 4,97 / 14,12 / 26,85) — tức là tăng gấp đôi kích thước gần như không cải thiện [S2]. (Một nhóm khác chạy PhoWhisper-large trên FLEURS lại báo 19,0% — ví dụ cho thấy kết quả phụ thuộc rất nhiều vào cách chạy [S10].)
Chênh lệch giữa các tập rất lớn. Cùng PhoWhisper-medium: 4,97% trên VIVOS nhưng 26,85% trên VLSP Task-2 [S2]. Con số "WER của model X" không có nghĩa nếu không nói rõ đo trên cái gì.
Ba lựa chọn đáng nhắc nhưng không nằm trong bảng:

Lựa chọn	Số liệu	Lưu ý
VietASR (SJTU + Tencent AI Lab)	68 M tham số; GigaSpeech2 7,68 / CommonVoice 11,46 / FLEURS 10,96; trung bình 8,31 (Whisper large-v3: 16,44; Google USM 13,03; Azure 11,78). Biến thể causal streaming: trung bình 10,13 [S4]	Pretrain trên 73 000 giờ YouTube, tinh chỉnh chỉ 50 giờ có nhãn. Bài báo nói "sẽ mã nguồn mở" — hãy coi đây là ý định, chưa phải model bạn tải được. Thua Whisper trên FLEURS, thua Azure trên Common Voice
Whisper large-v3 / turbo	Xem trên	Mặc định phổ biến nhất; một phép đo của chính nhà phát triển ứng dụng Whisper Notes: Qwen3-ASR-1.7B 4,91% so với Whisper large-v3-turbo 8,10% trên FLEURS tiếng Việt, đây là số tự chạy, không độc lập [S7]
Qwen3-ASR-0.6B	FLEURS vi 8,52 (1.7B: 5,55); TTFT 92 ms ở concurrency 1 [S5]	Bản nhỏ, phù hợp khi cần streaming nhẹ; kém bản 1,7B rõ rệt trên bảng tiếng Việt (20,92 vs 18,76)
4.3 Bảng 2 — Top 5 STT thương mại theo số do nhà cung cấp tự công bố 🟡
Không có bảng xếp hạng trung lập mà mình đọc được cho STT thương mại tiếng Việt. Bảng sau xếp theo FLEURS tiếng Việt do chính nhà cung cấp hoặc tác giả công bố; mỗi dòng có thể dùng cách chuẩn hóa khác nhau nên không so sánh trực tiếp với Bảng 1.

Hạng (theo số công bố)	Dịch vụ / model	FLEURS vi WER (%)	Nguồn / mức tin cậy	Streaming và ghi chú
1	ElevenLabs Scribe v1	3,5 (trang cũng ghi 3,1 ở phần mở đầu, không giải thích)	Trang của chính ElevenLabs [S8] 🟡	Có biến thể realtime (Scribe v2 Realtime: chữ tạm khoảng 150 ms "độ trễ model", bài viết không nêu tiếng Việt) [S9]
2	Qwen3-ASR-Flash (API Alibaba)	3,64	Bài báo Qwen3-ASR, ghi là bản API "chỉ để tham khảo" [S5] 🟡	Nhà cung cấp chính là tác giả model
3	Gemini Flash 2	4,2	Bảng trên trang ElevenLabs [S8] 🟡	Số đo do đối thủ công bố
4	AssemblyAI Universal-2	9,75	Tài liệu AssemblyAI, tháng 1/2026; bản Universal-3.5 Pro để trống tiếng Việt [S11] 🟡	Chính AssemblyAI cảnh báo benchmark công khai dễ gây hiểu lầm
5	Deepgram Nova-2	11,6	Bảng trên trang ElevenLabs [S8] 🟡	Nova-3 cũng hỗ trợ vi nhưng không có WER công bố; Flux không hỗ trợ tiếng Việt [S14]
Ba điều cần nói rõ về Bảng 2:

Đây là các phiên bản đã cũ (Scribe v1, Gemini 2.0 Flash, Nova-2). Các phiên bản mới (Scribe v2, Gemini 3.x, Nova-3) không có số tiếng Việt công khai mà mình tìm được.
FLEURS là giọng đọc, không phải hội thoại; trang Whisper Notes tự nhận bảng của họ "chỉ cho biết model có khả thi với ngôn ngữ đó, không phải độ chính xác bạn sẽ có" [S7].
Thứ tự này sẽ thay đổi khi có nhiễu và điện thoại. WildASR (đo EN/ZH/JA/KO, không có tiếng Việt) cho thấy xếp hạng đảo chiều theo điều kiện: Gemini 3 Pro đạt 3,8% WER trên FLEURS sạch nhưng 61,2% trên bộ có đoạn nhiễu MagicData, và lỗi trên câu ngắn dao động 38,7%–73,9% [S17]. Hãy coi đó là cảnh báo phương pháp, không phải số của tiếng Việt.
Các dịch vụ hỗ trợ tiếng Việt nhưng không có WER công khai:

Google Chirp 3: vi-VN ở trạng thái GA cho cả StreamingRecognize, Recognize và BatchRecognize; có chế độ chốt cuối lượt SUPERSHORT (độ trễ thấp nhất) [S12]. Trang không đưa WER tiếng Việt. Google USM (thế hệ trước) có WER trung bình 13,03 trong bảng của VietASR (FLEURS 11,75) [S4].
Microsoft Azure Speech: vi-VN có fast transcription, tùy biến bằng phrase list; WER trung bình 11,78 (FLEURS 11,88) trong bảng VietASR [S4, S13]. Phần đọc được không xác nhận hỗ trợ real-time cho vi-VN.
Deepgram Nova-3: có vi [S14].
Nhà cung cấp trong nước (FPT.AI, Viettel AI, VAIS, Zalo, Vbee): mình không tìm thấy bảng WER công khai mới nào. Có một bài so sánh học thuật cũ (VAIS, Viettel, Zalo, FPT, Google) mà mình chỉ thấy tiêu đề, chưa đọc nội dung. Nếu dữ liệu của bạn là điện thoại trong nước, hãy đưa các nhà cung cấp này vào vòng đo thử.
4.4 Shortlist STT cho voice bot realtime (nhận định của người viết ⚪)
Từ các dữ kiện trên, đây là năm ứng viên mình sẽ đưa vào vòng đo thử. Đây là danh sách để đo, không phải bảng xếp hạng chất lượng, vì không có phép đo trung lập nào xếp được cả năm ứng viên trên cùng tập tiếng Việt.

#	Ứng viên	Lý do đưa vào	Rủi ro/điều phải kiểm chứng
1	Qwen3-ASR-1.7B (tự host)	Open, streaming trong một model, tiếng Việt có trong bảng xếp hạng, TTFT ~100 ms	Độ trễ tăng mạnh khi nhiều luồng đồng thời (3,4 s ở 128 luồng) — phải đo trên GPU của bạn; chưa có số streaming tiếng Việt (bảng streaming trong bài báo chỉ có LibriSpeech/Fleurs-en/zh) [S5]
2	Google Chirp 3	vi-VN streaming GA, endpointing cấu hình được, dịch vụ được quản lý	Không có WER công bố; dữ liệu âm thanh rời khỏi hạ tầng của bạn
3	ElevenLabs Scribe (v1/v2 Realtime)	Số FLEURS tiếng Việt tốt nhất công bố; WebSocket, chữ tạm/chữ chốt, phân đoạn theo VAD	Số là của nhà bán; bài viết về bản realtime không nêu rõ tiếng Việt — kiểm tra trước khi mua
4	PhoWhisper-medium/small (tự host)	Dẫn đầu bảng open tiếng Việt; BSD-3; chạy được trên GPU/CPU vừa	Không streaming gốc — phải tự cắt đoạn (VAD) và xử lý phần chồng; repo chỉ hướng dẫn transformers, việc chuyển sang faster-whisper/CTranslate2 chưa được xác minh [S3]
5	Deepgram Nova-3 hoặc Azure Speech	Streaming trưởng thành, SLA, phrase list/keyterm (nếu cần từ vựng nghiệp vụ)	Không có WER tiếng Việt công khai; Flux (có turn detection tích hợp) không dùng được
Dự phòng nên cân nhắc: Zipformer-30M-RNNT hoặc VietASR streaming nếu cần mô hình rất nhỏ chạy trên CPU/thiết bị biên — nhưng cần xác nhận khả năng streaming và giấy phép trước.

4.5 Các kỹ thuật STT khác nên có
Từ vựng nghiệp vụ (contextual biasing / keyterm / phrase list): Azure có phrase list cho vi-VN [S13]; với model open, dùng hotword/ngôn ngữ-model phụ ⚪. Rất đáng làm khi có tên sản phẩm, tên chi nhánh, tên người.
Hậu xử lý bản chữ: chuẩn hóa số nói thành chữ số ("không chín ba hai…" → số điện thoại), viết hoa, dấu câu. Parakeet-Vietnamese dùng Qwen3 để sinh dấu câu/viết hoa cho phần lớn dữ liệu huấn luyện [S6].
Hỏi lại và xác nhận: với chuỗi quan trọng (số tài khoản, mã), cho bot đọc lại và xin xác nhận — đây là biện pháp ở tầng hội thoại, rẻ hơn nhiều so với cố đẩy WER xuống mức không thể.
Chống ảo giác (hallucination): khi audio rỗng hoặc rất nhiễu, một số model "bịa" chữ. WildASR đo riêng Hallucination Error Rate vì WER không bắt được loại lỗi này, và nêu ví dụ "delete" so với "delete all" như thế nào thì hành động nguy hiểm [S17]. Dự án demo của bạn bật vad_filter của Whisper là một lớp phòng thủ hợp lý.
5. Backend: xử lý lượt hội thoại, LLM và điều phối
5.1 Việc backend phải làm giữa "chữ vào" và "chữ ra"
Sau khi nhận được bản chữ chốt, backend chạy một vòng lặp:

Quyết định bắt đầu xử lý: có thể bắt đầu sớm trên bản chữ tạm (speculative prefill) và hủy nếu người dùng nói tiếp [S25]; đa số hệ thống chờ tín hiệu hết lượt.
Dựng ngữ cảnh: lịch sử hội thoại, thông tin khách, tài liệu truy xuất (RAG), kết quả công cụ nghiệp vụ (tra đơn, đặt lịch…).
Gọi LLM ở chế độ streaming (token ra từng phần) kèm gọi công cụ nếu cần.
Đẩy token qua bộ cắt câu sang TTS (mục 7), đồng thời gửi bản chữ về giao diện nếu có hiển thị phụ đề.
Giữ trạng thái và xử lý ngắt lời: khi người dùng nói chen vào, hủy luồng LLM + TTS, và cắt ngữ cảnh về đúng phần khách đã nghe.
5.2 Thiết kế prompt riêng cho giọng nói
Đây là các nguyên tắc ⚪ (kinh nghiệm chung, không từ một nguồn đơn lẻ), nhưng chúng ảnh hưởng chất lượng nhiều hơn việc chọn model:

Câu trả lời ngắn, mỗi câu một ý. Văn bản dài làm tăng thời gian nói và khiến người nghe mất kiên nhẫn; EVA đo "Conciseness" như một tiêu chí riêng cho độ phù hợp với giọng nói [S43].
Cấm định dạng thị giác: không markdown, danh sách gạch đầu dòng, emoji, URL dài, ký hiệu — TTS sẽ đọc thành lời hoặc im bất thường.
Viết ra dạng "để đọc": yêu cầu LLM xuất số, ngày, tiền ở dạng đọc được hoặc đi qua bộ chuẩn hóa văn bản (mục 7.3).
Chốt xưng hô nhất quán: ví dụ bot xưng "em", gọi khách "anh/chị", dùng "dạ, vâng, ạ" đúng chỗ. Đưa vào prompt hệ thống và kiểm bằng rubric (mục 9).
Câu đệm khi gọi công cụ chậm: nói ngay "Dạ, em kiểm tra giúp mình một chút ạ" trong lúc chờ API nghiệp vụ, để tránh khoảng lặng dài.
Ràng buộc an toàn trước khi hành động: với thao tác có hậu quả (chuyển tiền, hủy đơn) luôn xác nhận bằng bản chữ đã nhận dạng — đây là nơi ASR tường minh phát huy giá trị kiểm toán [S17].
5.3 Top 5 LLM cho tiếng Việt — bằng chứng đang có
Câu hỏi "LLM nào tốt nhất cho voice bot tiếng Việt" bị chặn bởi hai giới hạn của dữ liệu:

SEA-HELM (bảng xếp hạng của AI Singapore) có cột tiếng Việt, cập nhật 18/09/2026, đánh giá các năng lực như hội thoại nhiều lượt, làm theo chỉ dẫn, kiến thức, an toàn, văn hóa. Danh sách mình đọc được chỉ gồm model trọng số mở (bộ lọc "Base models"); mình không thấy GPT, Gemini, Claude hay các model đóng khác trên trang đã đọc [S30]. Vì vậy bảng dưới là top 5 model mở, không phải top 5 mọi model.
VMLU đo kiến thức đa lĩnh vực dạng trắc nghiệm, zero-shot; đây không phải đo chất lượng hội thoại hay độ trễ [S31].
Bảng 3 — Top 5 model mở theo SEA-HELM tiếng Việt (18/09/2026) 🟢 — điểm trung bình VI và các cột liên quan đến voice bot:

Hạng	Model	VI	Văn hóa	Làm theo chỉ dẫn	Hội thoại nhiều lượt	An toàn
1	Gemma 4 (31B)	77,09	75,32	96,21	87,12	72,42
2	Qwen 3.5 (122B-A10B, MoE)	76,39	75,21	94,92	87,06	67,66
3	Qwen 3.6 (27B)	76,12	75,69	94,40	84,71	64,91
4	SEA-LION v4.5 (Qwen) 27B	73,85	74,95	94,92	79,32	62,77
5	Muse Glimmer (30B)	73,52	61,37	95,88	87,09	67,62
Đọc bảng này thận trọng:

Trang báo khoảng tin cậy 95% theo bootstrap 2000 lần ("Average of 2000 bootstraps") mà mình không ghi lại. Khoảng cách 77,09 – 76,39 – 76,12 giữa ba model đầu rất nhỏ, nhiều khả năng không phân biệt được về thống kê; đừng chọn model chỉ vì hơn nhau dưới một điểm.
Điểm "Hội thoại nhiều lượt" và "Làm theo chỉ dẫn" liên quan trực tiếp hơn tới voice bot so với điểm trung bình; Muse Glimmer có điểm văn hóa thấp rõ rệt (61,37) dù hội thoại nhiều lượt cao.
Cột an toàn chênh lệch lớn (từ 62,77 đến 72,42) — quan trọng nếu bot nói chuyện với công chúng.
Model MoE (122B-A10B) chỉ kích hoạt khoảng 10 tỷ tham số mỗi token, nên tốc độ suy luận có thể gần với model nhỏ hơn nhiều so với tên gọi; nhưng cần nhiều bộ nhớ GPU để chứa trọng số ⚪.
Bảng không cho biết độ trễ. Với voice bot, TTFT thường quyết định trải nghiệm hơn là chênh vài điểm chất lượng.
VMLU — khối "model tiếng Việt trong nước" (track fine-tuned): VAI-LLM-v1.6+ (VAI) 90,04; V-LLM v2.1 (VinSmart Future) 89,85; VAI-LLM-v1.6 87,63; V-LLM v1.2 87,2; axis-sovereign (International AXIS) 85,75; trong top 10 còn có Vi-Qwen3.6 (Zoom AI Lab) và MISA-AI-1.0 [S31]. Điều này cho thấy các đơn vị trong nước có model rất mạnh về kiến thức tiếng Việt, nhưng mình không xác minh được các model này có API công khai, giấy phép, hay độ trễ. Track "from-scratch" do QwQ-32B (76,13) dẫn đầu [S31].

Khuyến nghị thực dụng ⚪:

Chọn một LLM nhanh, đủ tốt làm mặc định cho hội thoại thường (ưu tiên TTFT thấp, chất lượng hội thoại và an toàn), và một LLM mạnh hơn cho các lượt khó/phức tạp (định tuyến theo độ khó).
Đo trên chính kịch bản của bạn: xưng hô, số/địa chỉ, quy trình nghiệp vụ. Bảng chung không thay được điều này.
Với dữ liệu nhạy cảm hoặc yêu cầu nơi đặt máy chủ: Gemma 4 / Qwen 3.x tự host là điểm khởi đầu hợp lý theo bảng trên (giấy phép từng model cần kiểm tra).
Với model đóng (GPT, Gemini, Claude…): thiếu bảng tiếng Việt trung lập — tự chạy cùng bộ kịch bản với các model mở để so cùng thước.
5.4 Các thành phần backend khác
Thành phần	Vai trò	Lựa chọn tham khảo
Khung điều phối voice agent	Gắn STT–LLM–TTS, VAD, ngắt lời, transport	Pipecat, LiveKit Agents (mục 3.3)
Quản lý hội thoại	Quy trình nhiều bước, form-filling	Pipecat Flows, state machine tự viết ⚪
RAG	Tra cứu tài liệu nội bộ	Vector DB bất kỳ; giữ ngân sách thời gian truy xuất thật chặt ⚪
Công cụ nghiệp vụ	Gọi API CRM/đặt lịch	Hợp đồng timeout rõ ràng; câu đệm khi chậm
Bộ nhớ phiên/Định tuyến	Dính một cuộc gọi với một worker	Redis ⚪, mẫu như OpenAI mô tả [S24]
Cổng an toàn	Lọc prompt injection bằng giọng nói, kiểm duyệt đầu ra	Lớp riêng trước khi đưa vào LLM và sau khi ra TTS ⚪
6. Lựa chọn thay thế: model giọng nói-sang-giọng nói (speech-to-speech)
Thay vì ghép ba dịch vụ, bạn có thể dùng một API nhận audio và trả audio.

Ứng viên	Tiếng Việt?	Dữ kiện đã xác minh
Gemini Live API (Google)	Có — vi nằm trong bảng 99 ngôn ngữ [S29] 🟢	Nhận PCM 16-bit 16 kHz, trả 24 kHz; VAD tự động bật mặc định, khi phát hiện ngắt lời thì hủy và bỏ phần đang sinh; phiên chỉ-audio giới hạn 15 phút (có kỹ thuật gia hạn); cửa sổ ngữ cảnh 128k token cho model audio gốc
OpenAI Realtime (gpt-realtime-2.1)	Không thấy nêu — tài liệu realtime không có danh sách ngôn ngữ; một bài tổng hợp ghi "không công bố riêng cho dòng realtime" [S28, S32]	Kết nối bằng WebRTC (trình duyệt) hoặc WebSocket (server); tài liệu nêu rõ barge-in và lượt hội thoại. Phải tự kiểm tra chất lượng tiếng Việt bằng thử nghiệm
Qwen Omni (Alibaba)	Có trong tập FLEURS 60 ngôn ngữ do chính Qwen công bố [S32] 🟡	Độ trễ tới gói audio đầu tiên khoảng 978 ms cho đầu vào 6 giây (số của nhà cung cấp) [S32]
Ba điều đáng nói:

So sánh độ trễ giữa các nhà cung cấp không đáng tin. Bài tổng hợp [S32] dẫn một bài đo của bên thứ ba: Gemini 180–220 ms, OpenAI 200–250 ms đến audio đầu tiên, nhưng cũng dẫn nguồn khác nói cùng một model có thể ra từ 0,6 s đến hơn 3 s tùy điểm đo và relay. Mình coi những số này là định hướng.
Chỉ có một ứng viên với bằng chứng tiếng Việt chính thức (Gemini Live) trong ba ứng viên, nên không thể lập "top 5" có ý nghĩa. Ngoài ra, mình chưa tìm thấy đánh giá tiếng Việt độc lập nào cho bất kỳ model speech-to-speech nào.
Phù hợp khi: ưu tiên tốc độ ra thị trường, hội thoại chung, không cần kiểm toán từng bước. Không phù hợp khi: cần giọng thương hiệu riêng, bảng chữ đối chiếu cho pháp lý, dữ liệu phải ở trong nước, hay cần thay riêng STT/TTS để tối ưu cho tiếng Việt.
Một cách làm phổ biến là lai: dùng speech-to-speech cho hội thoại chung nhưng vẫn chạy STT song song để có bản chữ phục vụ kiểm toán và đánh giá ⚪.

7. Chiều về: từ chữ của bot đến giọng nói ở thiết bị
7.1 Luồng chạy ngược, nhìn từ đầu ra
LLM token stream ──► Bộ cắt câu (chunker) ──► Chuẩn hóa văn bản ──► TTS streaming
        │                    │                                          │
        │                    └──► (kênh chữ) phụ đề / UI ◄── mốc thời gian từng từ (nếu TTS trả)
        ▼                                                               ▼
  ghi nhận "đã nói đến đâu"                             khung audio 20–40 ms ──► Opus/PCM
        ▲                                                               │
        └──────── ngắt lời (barge-in): hủy LLM + TTS + xả hàng đợi ◄────┘
                                          WebRTC / WebSocket ──► jitter buffer ──► loa

Thành phần quyết định trải nghiệm không phải TTS mà là khâu ghép: cắt chữ khi nào, phát khi nào, dừng thế nào khi bị ngắt. Mỗi khâu có một nguồn trễ riêng; mục tiêu tổng cộng "từ lúc người dùng ngừng nói đến lúc nghe âm đầu tiên" là dưới 1,5 s ở P50 (mục 1.3).

7.2 "Nhả chữ": hai luồng chữ khác nhau
Mình hiểu "nhả chữ" là cách hệ thống phát ra văn bản từng phần khi LLM đang sinh. Thực tế có hai luồng cần phân biệt:

(a) Luồng chữ sang TTS — mục tiêu là tiếng nói liền mạch với độ trễ thấp nhất.

Không đưa từng token vào TTS: giọng bị đứt, ngữ điệu sai. Cũng không chờ LLM xong cả câu trả lời: trễ quá nhiều.
Dùng bộ cắt câu (chunker): chunk đầu cắt sớm theo mệnh đề (dấu phẩy, chấm phẩy) khi đủ vài âm tiết để giảm độ trễ đến âm đầu tiên; các chunk sau cắt theo câu để giữ ngữ điệu; có trần độ dài để không chờ vô hạn khi LLM không ngắt câu.
Tiếng Việt viết tách âm tiết, nên một token trắng ≈ một âm tiết; đếm số âm tiết bằng cách tách theo dấu cách là đủ cho ngưỡng ⚪.
Cần tránh cắt nhầm ở số thập phân/phân cách nghìn ("1.500.000đ") và chữ viết tắt có dấu chấm ("TP. Hồ Chí Minh"). Đoạn mã dưới đây yêu cầu có khoảng trắng sau dấu câu nên không cắt trong "1.500.000đ", nhưng vẫn cắt nhầm ở "TP. " — danh sách viết tắt cần bổ sung.
import re

SENT_END = re.compile(r"(?<=[.!?…])\s+")   # sentence boundary (needs trailing space)
CLAUSE_END = re.compile(r"(?<=[,;:])\s+")   # clause boundary


class SpeechChunker:
    """Turn a stream of LLM tokens into TTS-sized text chunks.

    - First chunk is flushed early (clause level) to cut time-to-first-audio.
    - Later chunks are flushed at sentence level for better prosody.
    - A hard cap bounds latency when the model never emits punctuation.
    Vietnamese: one whitespace-separated token ~= one syllable.
    """

    def __init__(self, first_min=5, min_len=10, max_len=45):
        self.buf = ""
        self.first = True
        self.first_min, self.min_len, self.max_len = first_min, min_len, max_len

    @staticmethod
    def _n(text: str) -> int:
        return len(text.split())

    def _emit(self, out, head, tail):
        out.append(head.strip())
        self.buf = tail
        self.first = False

    def push(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while True:
            need = self.first_min if self.first else self.min_len
            head, *rest = SENT_END.split(self.buf, maxsplit=1)
            if rest and self._n(head) >= need:
                self._emit(out, head, rest[0]); continue
            if self.first:
                head, *rest = CLAUSE_END.split(self.buf, maxsplit=1)
                if rest and self._n(head) >= self.first_min:
                    self._emit(out, head, rest[0]); continue
            if self._n(self.buf) >= self.max_len:
                head, _, tail = self.buf.rpartition(" ")
                if head:
                    self._emit(out, head, tail); continue
            return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []

(Mình đã chạy đoạn mã này trên câu "Dạ, em đã kiểm tra đơn hàng của anh rồi ạ. Đơn sẽ được giao vào ngày mai, tổng cộng 1.500.000đ. Anh cần em hỗ trợ thêm gì không ạ?" và nó tách đúng thành ba câu, không cắt trong số tiền. Đây là phác thảo để minh họa, chưa phải thư viện đã kiểm thử đầy đủ.)

Hai mô hình TTS streaming: (1) đầu vào streaming — TTS nhận chữ từng phần và bắt đầu sinh âm khi mới có vài từ (các TTS đám mây thường có WebSocket cho việc này ⚪; VieNeu-TTS Turbo sinh âm thanh theo từng frame [S33]); (2) đầu vào theo câu — bạn tự cắt câu như trên rồi gọi TTS nhiều lần, mỗi lần nhận âm thanh streaming. Mô hình (2) đơn giản hơn và hoạt động với mọi TTS, kể cả loại chỉ có API theo câu.
Câu mở đầu đóng gói sẵn: các cụm hay dùng ("Dạ, em nghe ạ", "Dạ, em kiểm tra ngay ạ") có thể sinh trước và lưu cache để phát tức thì, độ trễ gần bằng không ⚪.
(b) Luồng chữ lên giao diện (phụ đề, hiển thị trên app) — mục tiêu là chữ khớp với những gì người dùng đang nghe.

LLM sinh chữ nhanh hơn nhiều so với tốc độ nói. Nếu đẩy thẳng token ra màn hình, chữ hiện ra trước tiếng nói vài giây — gây khó chịu và sai khi bị ngắt.
Cách đúng: hiển thị chữ theo mốc thời gian của audio đã phát. Nếu TTS trả mốc thời gian từng từ thì dùng chúng; nếu không, ước lượng theo thời lượng từng chunk và tốc độ phát ⚪.
Khi bị ngắt lời, cần biết phần chữ nào thực sự đã được phát để cập nhật ngữ cảnh LLM. Hướng dẫn của Deepgram nêu đúng vấn đề này: phải đối chiếu âm thanh đã phát với chữ khách chưa nghe, và một số TTS API mới trả về cả phần đã đọc lẫn phần còn lại để việc này dễ hơn [S25].
Chữ từ STT hiển thị cho người dùng (bản "tạm") nên đánh dấu là tạm thời và cho phép sửa; chữ tạm có thể đổi khi model nghe thêm ngữ cảnh [S9].
7.3 Chuẩn hóa văn bản tiếng Việt trước khi đọc
TTS đọc đúng những gì nó nhận được. Văn bản thật chứa nhiều thứ không đọc được thẳng: số, ngày giờ, tiền, đơn vị, viết tắt, tiếng Anh. Phần này là nguồn lỗi phổ biến trong sản phẩm thật và dễ kiểm thử.

Loại	Ví dụ đầu vào	Việc cần làm
Số tiền	1.500.000đ	"một triệu năm trăm nghìn đồng" (dấu chấm là phân cách nghìn trong tiếng Việt, khác tiếng Anh)
Số điện thoại, mã	0912 345 678, OTP 482913	đọc từng chữ số; nhóm hợp lý; tránh đọc thành số lớn
Ngày giờ	14/10/2026, 9h30	"ngày mười bốn tháng mười năm hai nghìn không trăm hai mươi sáu"; "chín giờ ba mươi"
Phần trăm, đơn vị	15%, 20 km/h	"mười lăm phần trăm", "hai mươi ki-lô-mét trên giờ"
Viết tắt, địa danh	TP.HCM, UBND, CCCD	từ điển viết tắt: đọc cả cụm hay đánh vần từng chữ cái
Từ tiếng Anh/tên riêng	email, iPhone, Facebook	từ điển phiên âm hoặc để TTS đọc theo cách đã cấu hình
Ký hiệu, emoji, URL	https://..., →	loại bỏ hoặc thay bằng lời
Công cụ: VietNormalizer là thư viện Python mã nguồn mở (MIT), dựa trên luật, không cần GPU hay API, bao gồm bảy nhóm (số, ngày, giờ, tiền VND/USD, phần trăm, viết tắt, từ mượn); bài báo không đưa số đo độ chính xác, chỉ so sánh định tính với các công cụ khác [S47]. VieNeu-TTS dùng bộ chuyển chữ-sang-âm vị sea-g2p [S33]. Dù dùng công cụ nào, nên có bộ kiểm thử đơn vị cho bảng trên (đầu vào → dạng đọc mong đợi) và chạy mỗi lần nâng cấp.

Một lựa chọn thay thế ⚪ là yêu cầu LLM xuất sẵn "dạng để đọc" trong prompt, rồi vẫn đi qua bộ chuẩn hóa như lớp bảo hiểm. Không nên chỉ phụ thuộc LLM vì lỗi hiếm (đọc sai một chữ số) có hậu quả lớn.

7.4 Phát về thiết bị và xử lý ngắt lời (barge-in)
Định dạng đầu ra: Gemini Live trả PCM 24 kHz [S29]; VieNeu-TTS Turbo xuất 48 kHz [S33]; Piper vais1000 xuất 22,05 kHz (dự án demo của bạn). Chuẩn hóa về một tần số ở gateway; qua WebRTC thì mã hóa Opus.
Bộ đệm phát: giữ vài trăm mili-giây để chống giật nhưng đừng quá lớn, vì bộ đệm chính là độ trễ khi ngắt lời.
Quy trình khi người dùng nói chen vào (có thể xử lý ở client để phản ứng tức thì):
VAD phía client phát hiện tiếng nói mới trong khi bot đang phát (AEC phải hoạt động tốt, nếu không bot tự ngắt vì nghe chính mình, mục 2.2).
Dừng phát và xả hàng đợi audio ngay lập tức.
Gửi tín hiệu hủy lên backend: hủy luồng TTS, hủy sinh LLM (Gemini Live làm đúng việc này: "phần sinh đang chạy bị hủy và bỏ đi" khi phát hiện ngắt lời [S29]).
Cắt ngữ cảnh về phần bot đã thực sự nói.
Xử lý lượt nói mới.
Tiếng đệm của người dùng ("ừ", "vâng") không nên ngắt bot: cần ngưỡng thời lượng tối thiểu hoặc phân loại "backchannel" ⚪. Full-Duplex-Bench liệt kê "backchanneling" và "interruption management" thành các hành vi được đo riêng [S44].
7.5 Top 5 TTS cho tiếng Việt — bằng chứng đang có
Giới hạn của dữ liệu: mình không đọc được bảng xếp hạng TTS tiếng Việt trung lập nào. Voice Arena có tiếng Việt trong mô tả phạm vi nhưng nội dung bảng không tải được [S46]; Artificial Analysis có "Multilingual Text to Speech Arena" (bài thông báo trên X bị chặn truy cập) [S45]; bảng Speech Arena chính của Artificial Analysis mà mình đọc được không hiển thị bộ lọc ngôn ngữ và độ trễ [S40]. Vì vậy dưới đây là năm lựa chọn có bằng chứng hỗ trợ tiếng Việt, kèm bằng chứng chất lượng nếu có — không phải bảng xếp hạng.

Bức tranh chung (chủ yếu tiếng Anh): Speech Arena của Artificial Analysis (bình chọn mù của người nghe), 10 model đầu theo Elo: Eleven v4 Turbo 1330; Eleven v4 1328; Qwen-Audio-3.1-TTS-Plus 1300; Sonic 3.6 1286; Gemini 3.8 Flash TTS 1274; Qwen-Audio-3.0-TTS-Plus 1268; Realtime TTS-2 (Inworld) 1260; rồi Gemini 3.8 Flash-Lite TTS 1248, Simba 3.2 1245, Fish Audio Drama 3 Preview 1243 [S40]. Trang không nói bảng này đo ngôn ngữ nào; không dùng nó để chọn TTS tiếng Việt.

#	Lựa chọn	Triển khai	Bằng chứng tiếng Việt đã xác minh	Hiệu năng/độ trễ	Rủi ro và điều phải kiểm chứng
1	VieNeu-TTS (v3 Turbo)	Tự host (Apache-2.0); CPU hoặc GPU	Mô hình chuyên tiếng Việt, huấn luyện trên "10 000+ giờ song ngữ"; nhân bản giọng từ mẫu 3–8 s; tinh chỉnh LoRA cho một giọng cần 10–30 phút âm thanh sạch [S33] 🟡	Streaming từng frame. RTF CPU int8 0,35–0,37 (fp32 0,55–0,62); GPU một câu 0,10, gộp batch 0,011–0,02. Thời gian đến audio đầu: GPU 1 luồng ~115 ms, 16 luồng 185 ms (trung vị), 32 luồng ~450 ms; CPU int8 140–195 ms. Đầu ra 48 kHz	Không công bố MOS hay WER; số trên do tác giả đo (RTX 3060, Core i5 thế hệ 12); int8 cần CPU có VNNI; mặc định chỉ 1 luồng trên CPU. Phiên bản v4 là độc quyền (qua API). Repo ghi v3 Turbo do một tác giả thiết kế và huấn luyện; độ trưởng thành của cộng đồng cần tự đánh giá
2	Cartesia Sonic	API đám mây	Có trang tiếng Việt với giọng Minh, Lien; hỗ trợ streaming [S36] 🟡	Trang nêu "dưới 90 ms" (không rõ model, điều kiện). Voice Arena đo trung vị: Sonic-3.5 250 ms, Sonic-3.6 341 ms (không riêng tiếng Việt) [S41]	Huy hiệu "Ranked #1 in Speech Arena" không có ngày/tiêu chí; trên bảng AA mình đọc, Sonic 3.6 xếp hạng 4 [S36, S40]. Giá khoảng 49 USD/1 triệu ký tự (tháng 9/2026) [S41]
3	ElevenLabs (Eleven v3/v4)	API đám mây	Có trang TTS tiếng Việt; trang không nêu model nào hỗ trợ tiếng Việt hay số giọng [S37] 🟡	Eleven v4 / v4 Turbo đứng đầu Elo trên bảng AA [S40]; hãng nói Flash ~75 ms, v4 Turbo ~100 ms suy luận / ~150 ms đến tiếng đầu tiên (số hãng). Voice Arena đo trung vị Eleven v3: 588 ms [S41]	Phải tự xác minh model nào đọc tiếng Việt tốt; giá cao (Eleven v4: 80 USD/1 triệu ký tự niêm yết; mức giảm giá ra mắt hết hạn 12/10/2026, tức là đã kết thúc tính đến hôm nay) [S41]
4	Google Chirp 3 HD (và Gemini TTS)	API đám mây	vi-VN có trong bảng ngôn ngữ [S38] 🟢; Gemini 3.8 Flash TTS nêu 130 ngôn ngữ (số do hãng, tiếng Việt không được nêu tên) [S41]	Chirp 3 HD có API streaming nhưng trang không xác nhận riêng cho vi-VN; trang không liệt kê tên giọng tiếng Việt. Gemini 3.8 Flash TTS: Elo 1274 (hạng 5 bảng AA), 16,49 USD/1 triệu ký tự [S40, S41]	Chất lượng tiếng Việt chưa có số đo; đo trực tiếp
5	Nhà cung cấp trong nước: Vbee AIVoice, FPT.AI, Viettel AI	API (Vbee, FPT.AI xác nhận có API; Viettel chưa rõ)	Bài so sánh do chính Vbee đăng, tự xếp mình "xuất sắc"; FPT.AI được mô tả "giống người thật 98%" — chưa kiểm chứng [S39]	Chưa có số về streaming hoặc độ trễ	Không có đánh giá độc lập. Ưu điểm tiềm năng: hợp đồng trong nước, đặt dữ liệu trong nước (⚪ cần xác minh)
Các lựa chọn khác cần biết:

Piper vi_VN-vais1000-medium (đang dùng trong dự án demo của bạn): VITS 22,05 kHz, chạy nhanh trên CPU; dữ liệu huấn luyện VAIS-1000 (CC BY 4.0). Trong log của dự án, TTS trung vị 0,16 s cho 17–100 ký tự (có một ngoại lệ 3,79 s) — nhưng đó là thời gian tổng hợp cả câu rồi mới trả file, không phải streaming. Chất lượng giọng chưa có số đo (dự án chưa đánh giá MOS).
XTTS-v2 và biến thể tinh chỉnh tiếng Việt (nghiên cứu): bài báo "Zero-Shot Text-to-Speech for Vietnamese" tạo bộ dữ liệu PhoAudiobook (941 giờ, 735 người nói) và tinh chỉnh XTTS-v2 → MOS 4,20 / WER 4,16 trên tập cùng phân phối (giọng đọc tham chiếu thật: MOS 4,61), nhưng trên VIVOS (ngoài phân phối) MOS chỉ 2,79 và WER 37,81; viXTTS đạt 4,05 / 4,23 trên cùng tập đầu [S34]. Bài học: TTS clone giọng có thể rất giòn ngoài miền huấn luyện. Bộ dữ liệu chỉ phát hành phi thương mại, và bài báo không đưa liên kết trọng số — không phải lựa chọn production.
Qwen3-TTS (mã nguồn mở): hỗ trợ 10 ngôn ngữ, không có tiếng Việt [S35] — loại khỏi danh sách dù độ trễ 97 ms hấp dẫn.
Azure Neural TTS vi-VN: theo hiểu biết của mình có các giọng tiếng Việt nhưng trang tài liệu đã đọc không hiển thị bảng giọng [S13] ⚪ — kiểm tra trực tiếp.
Độ trễ đo thực tế (trung vị đến âm thanh đầu tiên, Voice Arena, chủ yếu tiếng Anh) [S41]: Simba 3.2 123 ms; Inworld Realtime TTS 2 (preview) 169 ms; Gradium 236 ms; Cartesia Sonic-3.5 250 ms; Fish Audio S2 Pro 267 ms; Sonic-3.6 341 ms; Azure Dragon HD Omni 485 ms; ElevenLabs v3 588 ms; gpt-4o-mini-tts 812 ms; Gemini 3.1 Flash TTS 890 ms. Bảng này cho bậc độ lớn, không thay phép đo trên chính tiếng Việt của bạn.

Khuyến nghị cho lộ trình ⚪: bắt đầu với VieNeu-TTS tự host làm đường chính (rẻ, streaming, kiểm soát được), một TTS đám mây làm đường dự phòng/so sánh chất lượng, và dành thời gian xây bộ kiểm thử nghe và đọc thực thể (mục 9.6) — vì chính điều đó, chứ không phải bảng xếp hạng, sẽ quyết định lựa chọn cuối cùng.

8. Triển khai tại Việt Nam: vị trí máy chủ, điện thoại, dữ liệu và chi phí
8.1 Vị trí máy chủ và độ trễ mạng
Ngân sách độ trễ (mục 1.3) rất sát, nên mỗi khâu trong chuỗi phải đặt gần người dùng. Bài của LiveKit nêu ví dụ Singapore ↔ US-East: RTT thực tế khoảng 230–280 ms trước khi tính xử lý [S23]. Với người dùng ở Việt Nam, điều đó nghĩa là:

Không gọi STT/LLM/TTS ở vùng Mỹ cho cuộc gọi thời gian thực nếu có lựa chọn gần hơn.
Ưu tiên vùng Singapore / Hồng Kông / Đông Nam Á của nhà cung cấp, hoặc trung tâm dữ liệu trong nước nếu cần.
Kiểm tra từng nhà cung cấp có điểm đặt máy chủ nào gần Việt Nam không và có cho cố định vùng xử lý không ⚪ (mình chưa tra thông tin vùng của từng nhà cung cấp).
Đo RTT thật từ mạng của khách hàng (4G/5G/Wi-Fi ở các thành phố khác nhau), không chỉ từ văn phòng.
8.2 Điện thoại
Cuộc gọi qua tổng đài thường là G.711 8 kHz (hoặc một số codec khác ⚪); STT phải chạy được ở băng hẹp. Kết quả mục 4 hầu hết đo trên âm thanh băng rộng, nên phải đo riêng trên bản ghi điện thoại (mục 9).
Tích hợp qua SIP trunk với nhà mạng/nhà cung cấp tổng đài; LiveKit Agents hỗ trợ SIP gốc, Pipecat thường qua cầu nối [S26].
Cần xử lý DTMF, hộp thư thoại cho cuộc gọi ra, và chuyển cuộc gọi cho nhân viên khi bot không giải quyết được ⚪.
8.3 Dữ liệu giọng nói và pháp lý (cần xác nhận với luật sư)
Phần này là hiểu biết chung của người viết ⚪; mình chưa tra cứu lại văn bản pháp luật trong lần nghiên cứu này, nên coi đây là danh sách câu hỏi cần xử lý chứ không phải kết luận pháp lý.

Giọng nói và bản ghi chữ là dữ liệu cá nhân. Khung pháp lý hiện hành ở Việt Nam gồm Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân và Luật Bảo vệ dữ liệu cá nhân (theo hiểu biết của mình có hiệu lực từ 1/1/2026) — cần xác nhận phạm vi áp dụng, nghĩa vụ thông báo/đồng ý, thời hạn lưu trữ, đánh giá tác động, và quy định chuyển dữ liệu ra nước ngoài khi gọi dịch vụ đám mây ở nước khác.
Thông báo ghi âm và bản chất AI ngay đầu cuộc gọi.
Dấu giọng (voiceprint) dùng để xác thực có thể được coi là dữ liệu sinh trắc/nhạy cảm — hỏi chuyên gia pháp lý.
Hợp đồng với nhà cung cấp: điều khoản không dùng dữ liệu của bạn để huấn luyện model, vùng lưu trữ, thời hạn xóa.
Biện pháp kỹ thuật nên áp dụng dù pháp lý ra sao: mã hóa khi truyền và khi lưu; thời hạn lưu cố định, xóa tự động; che thông tin định danh trong log và bản chữ; kiểm soát truy cập bản ghi; tách môi trường thử nghiệm khỏi dữ liệu thật. Dự án demo hiện chưa có các lớp này (bản ghi, transcript và log JSONL lưu không mã hóa, không tự xóa — xem tài liệu kỹ thuật 1-technote.md mục 9.1), điều ổn cho thử nghiệm cục bộ nhưng cần thay đổi trước khi chạm vào dữ liệu khách hàng thật.

Một vấn đề khác: giả mạo giọng nói. Công nghệ nhân bản giọng (VieNeu-TTS nhân bản từ mẫu 3–8 s [S33]) cũng được kẻ gian dùng; nếu bot cho phép thao tác tài chính, đừng xác thực chỉ bằng giọng — kết hợp OTP, thiết bị tin cậy và mô hình chống giả mạo (mục 2.5) ⚪.

8.4 Chi phí — mô hình tính nhanh
Chi phí mỗi phút cuộc gọi ≈ STT (tính theo giờ audio hoặc GPU tự host) + LLM (token vào/ra) + TTS (ký tự) + hạ tầng media. Các mốc giá tham khảo (tháng 9/2026, không riêng tiếng Việt) [S41]:

TTS	USD / 1 triệu ký tự
Simba 3.2 (Speechify)	6,58
Gemini 3.8 Flash-Lite TTS	11,03
Gemini 3.8 Flash TTS	16,49
Cartesia Sonic 3.6	49,00
ElevenLabs Eleven v4	80 (giá niêm yết)
Nếu một phút lời nói bot tương đương khoảng 1 000–1 500 ký tự tiếng Việt (ước lượng của người viết ⚪, hãy tự đo), TTS tính theo mức trên rơi vào khoảng 0,7–1 xu USD mỗi phút (Simba) cho đến khoảng 8–12 xu USD mỗi phút (Eleven v4). Điểm quan trọng hơn là tỷ lệ: giữa TTS rẻ nhất và đắt nhất trong bảng có khoảng 12 lần; LLM và STT cũng có chênh lệch tương tự. Tự host (VieNeu-TTS, Qwen3-ASR, PhoWhisper) đổi chi phí biến đổi lấy chi phí GPU cố định và công vận hành — chỉ có lợi khi lưu lượng đủ lớn ⚪.

8.5 Ba giai đoạn triển khai
Giai đoạn	Mục tiêu	Cấu hình gợi ý ⚪
Prototype (1–2 tuần)	Chứng minh vòng hội thoại, đo thử model	Một máy, WebSocket, Silero VAD, STT (PhoWhisper hoặc Qwen3-ASR hoặc Chirp 3), LLM qua API, VieNeu-TTS hoặc Piper, log JSONL + bộ đo mục 9
Pilot (vài tuần)	Người dùng thật, giới hạn quy mô	Pipecat hoặc LiveKit Agents, WebRTC cho web/app và SIP cho điện thoại, vùng Singapore/trong nước, bản ghi âm có đồng ý, bảng điều khiển độ trễ P50/P95
Production	Ổn định, mở rộng, tuân thủ	Nhóm GPU tự co giãn, nhiều nhà cung cấp dự phòng cho STT/LLM/TTS, Redis cho định tuyến phiên, hàng đợi sự kiện (Kafka/Redpanda/NATS) cho phân tích và QA, cổng hồi quy tự động, giám sát cuộc gọi tổng hợp 24/7
9. Đánh giá hệ thống: làm sao biết voice bot "ổn" và "có chất lượng"
9.1 Nguyên tắc nền
"Hệ thống ổn" và "hệ thống có chất lượng" là hai câu hỏi khác nhau và cần hai nhóm bằng chứng:

Chất lượng (quality): bot nghe đúng không, hiểu đúng không, trả lời đúng và dễ nghe không, hoàn thành việc cho khách không.
Ổn định (reliability): bot có giữ được chất lượng đó liên tục, dưới tải, khi mạng xấu, khi một dịch vụ phía sau lỗi, và sau mỗi lần nâng cấp không.
Bốn nguyên tắc cho cả phương pháp:

Đo theo tầng rồi mới đo đầu–cuối. Khi tỷ lệ hoàn thành tác vụ giảm, phải biết lỗi nằm ở STT, turn detection, LLM, công cụ hay TTS. Hamming chia khung đánh giá thành ASR / LLM / TTS / hệ thống / kinh doanh [S42]; EVA tách độ chính xác (task completion, faithfulness, speech fidelity) khỏi trải nghiệm (conciseness, conversation progression, turn-taking) [S43].
Đo cả hai thế giới: trước khi ra mắt (bộ kiểm thử cố định, mô phỏng, tải) và sau khi ra mắt (giám sát, chấm điểm cuộc gọi thật). Cái thứ nhất chặn lỗi hồi quy, cái thứ hai bắt những thứ bộ kiểm thử không có.
Báo cáo phân phối và lát cắt, không chỉ trung bình. WildASR cho thấy tổng hợp WER che giấu việc lỗi dồn vào điều kiện cụ thể, nên họ báo thêm P90 WER và mức suy giảm theo từng điều kiện [S17]. Một độ trễ trung bình 1,2 s có thể đi kèm 10% lượt trễ trên 4 s.
Có bằng chứng thống kê. Chênh lệch nhỏ trên tập nhỏ là nhiễu (mục 9.10).
Một khung tổng thể:

                    ┌──────────────────── TRƯỚC KHI RA MẮT ────────────────────┐
 Tầng 1 Thành phần  │ STT · VAD/turn · LLM · TTS · chuẩn hóa văn bản          │  bộ kiểm thử cố định (golden set)
 Tầng 2 Hội thoại   │ mô phỏng bot-to-bot, kịch bản + nhiễu + ngắt lời        │  tự động, chạy lặp k lần
 Tầng 3 Hệ thống    │ độ trễ phân vị · tải · soak · chaos                     │  hồ sơ mạng thật, nhiều vùng
 Tầng 4 Cổng        │ so với baseline: chặn triển khai nếu hồi quy            │
                    └────────────────────────────┬──────────────────────────────┘
                                                 ▼ canary / shadow
                    ┌──────────────────── SAU KHI RA MẮT ─────────────────────┐
                    │ cuộc gọi tổng hợp 5–15 phút/lần · chấm điểm cuộc gọi thật│
                    │ cảnh báo · duyệt mẫu bởi người · đưa lỗi vào golden set  │
                    └──────────────────────────────────────────────────────────┘

9.2 Xây bộ dữ liệu đánh giá cho tiếng Việt
Chất lượng đánh giá không vượt quá chất lượng bộ dữ liệu. Với tiếng Việt, mình khuyên:

Nguồn dữ liệu:

Nguồn	Ưu	Nhược
Bản ghi cuộc gọi/phiên thật (có đồng ý, đã che thông tin cá nhân)	Đúng phân phối thực	Cần gán nhãn tay; vấn đề pháp lý (mục 8.3)
Kịch bản đọc theo thiết kế (người thật đọc)	Kiểm soát nội dung, có nhãn sẵn	Kém tự nhiên
Âm thanh tổng hợp (TTS + nhiễu)	Rẻ, phủ nhiều tình huống	WildASR chỉ ra khoảng cách giữa giọng tổng hợp và giọng thật: ví dụ Whisper large-v3 trên giọng trẻ em tiếng Anh đạt 3,7% với giọng tổng hợp nhưng 21,7% với giọng thật [S17] — đừng chỉ tin vào dữ liệu tổng hợp
Bộ dữ liệu công khai (VIVOS, Common Voice vi, FLEURS, VLSP, ViMD…)	Để so sánh nhanh các model	Phần lớn là giọng đọc; model có thể đã huấn luyện trên đó (xem lưu ý parakeet ở mục 4.2) [S1, S6]
Kích thước và cơ cấu: Hamming gợi ý ít nhất 50 kịch bản, với 200+ cho production, chia 40% đường đi bình thường / 30% tình huống biên / 15% xử lý lỗi / 10% đối kháng / 5% biến thiên âm học [S42] 🟡. Đây là điểm khởi đầu hợp lý, không phải chuẩn.

Các lát cắt riêng cho tiếng Việt (mỗi lát cắt báo cáo riêng, không gộp):

Lát cắt	Lý do
Giọng Bắc / Trung / Nam, và giọng địa phương	Đổi rõ chất lượng STT; chọn tỷ lệ theo khách hàng thật
Giới tính, tuổi (trẻ em, người lớn tuổi)	WildASR chứng minh lỗi nhảy vọt với các nhóm này (ở tiếng Anh/Trung) [S17]
Nhiễu: SNR 20 / 10 / 5 / 0 dB	Hamming đề xuất các mức này; ngưỡng tăng WER ≤ 5% ở 10 dB và ≤ 10% ở 5 dB [S42] 🟡
Kênh: băng rộng 16 kHz, điện thoại 8 kHz (G.711), loa ngoài, xe cộ	Codec điện thoại là một trong các điều kiện làm lệch kết quả [S17]
Câu rất ngắn ("dạ", "vâng", "không", "có")	Trong WildASR, câu ngắn đạt 38,7–73,9% lỗi ở mọi model thử [S17]; trong hội thoại thật phần lớn lượt trả lời của khách rất ngắn
Chuỗi số và thực thể: số điện thoại, CCCD, số tài khoản, ngày sinh, số tiền, địa chỉ, tên riêng	Nguồn lỗi có hậu quả lớn nhất [S43]
Trộn tiếng Anh–Việt (email, tên sản phẩm, "ok", "app")	WildASR dùng MER riêng cho code-switching [S17]
Ngập ngừng, nói đè, sửa lời giữa chừng ("à nhầm, ý em là…")	Phần lớn nguồn tiếng Việt là giọng đọc; VietSuperSpeech lưu ý khẩu ngữ khiến WER tăng mạnh nhưng không có số [S18]
Âm thanh rỗng, tiếng ồn thuần, tiếng bot vọng lại	Thử khả năng chống ảo giác và AEC
9.3 Đo STT
Chỉ số cốt lõi:

WER = (thay + xóa + chèn) / tổng số từ chuẩn [S42]. Với tiếng Việt, đơn vị "từ" thực chất là âm tiết (tách theo dấu cách) nên mỗi lỗi dấu là một lỗi từ.
CER để bổ sung, nhất là khi so sánh chuỗi số và tên riêng.
WER tách theo dấu thanh: tính hai lần — có dấu thanh và bỏ dấu thanh (giữ nguyên ă â ê ô ơ ư). Hiệu số cho biết bao nhiêu lỗi chỉ do thanh điệu, rất hữu ích khi so sánh model hoặc khi nghi ngờ vấn đề là âm thanh điện thoại ⚪. Đoạn mã dưới đây đã chạy trên ví dụ nhỏ (WER 21,4% khi tính dấu thanh, 0% khi bỏ qua dấu thanh):
import random
import re
import unicodedata

TONE_MARKS = {"̀", "́", "̉", "̃", "̣"}  # huyen sac hoi nga nang


def norm(text: str, strip_tones: bool = False) -> list[str]:
    """Fixed, versioned normalization. Keep it identical for every system you compare."""
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s]", " ", t)            # drop punctuation
    if strip_tones:                            # remove ONLY the 5 tone marks (keep a, o, u, e vowel shapes)
        t = unicodedata.normalize("NFD", t)
        t = "".join(c for c in t if c not in TONE_MARKS)
        t = unicodedata.normalize("NFC", t)
    return t.split()                           # whitespace tokens ~ syllables


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def corpus_wer(pairs, strip_tones=False) -> float:
    err = n = 0
    for ref, hyp in pairs:
        r, h = norm(ref, strip_tones), norm(hyp, strip_tones)
        err += edit_distance(r, h)
        n += len(r)
    return err / max(n, 1)


def bootstrap_ci(pairs, strip_tones=False, b=1000, seed=0):
    """95% CI by resampling utterances with replacement."""
    rng = random.Random(seed)
    stats = sorted(
        corpus_wer([rng.choice(pairs) for _ in pairs], strip_tones) for _ in range(b)
    )
    return stats[int(0.025 * b)], stats[int(0.975 * b)]

(Bộ ví dụ ba cặp câu chỉ để kiểm tra mã, khoảng tin cậy 95% của nó rộng — từ 7,7% đến 50% — đúng như kỳ vọng với tập quá nhỏ.)

Chuẩn hóa cố định và có phiên bản. Cùng hệ thống có thể ra WER chênh hai lần chỉ vì chuẩn hóa khác (PhoWhisper-medium: 16,73 và 7,61 trên cùng bảng [S1]; Whisper large-v3 FLEURS vi: 8,59 / 9,1 / 10,7 ở ba nguồn [S4, S10, S8]). Quy tắc: một bộ chuẩn hóa, ghi vào mã, dùng cho mọi hệ thống, kể cả quy ước đọc số ("25" so với "hai mươi lăm") — và áp dụng cho cả tham chiếu lẫn giả thuyết.
Độ chính xác thực thể (entity accuracy): khớp tuyệt đối trên số điện thoại, mã, số tiền, ngày, tên. Đây thường là số liệu quan trọng hơn WER; EVA nhận định phiên âm thực thể là kiểu hỏng nổi trội [S43].
Tỷ lệ ảo giác (HER) và P90 WER: HER bắt các trường hợp model "bịa" nội dung mà WER bỏ sót; ví dụ trong WildASR, Nova 2 trên tiếng Trung có code-switching đạt MER 33,7 nhưng HER 68,4 [S17]. Với âm thanh rỗng/rất nhiễu, đo tỷ lệ model sinh chữ khi không có lời nói.
Độ chính xác ý định/slot đầu ra: lỗi STT không làm đổi ý nghĩa thì vô hại. Đo thêm: tỷ lệ lượt mà LLM vẫn hiểu đúng ý dù chữ có lỗi ⚪ — giúp tránh tối ưu WER cho chính nó.
Chỉ số streaming:

Chỉ số	Ý nghĩa	Cách đo ⚪
Thời gian đến chữ tạm đầu tiên	Cảm nhận phản hồi, kích hoạt prefill	Từ lúc phát âm đầu tiên (nhãn tay/tổng hợp) đến partial đầu
Độ trễ chốt (finalization)	Thời gian từ hết lời đến final	Từ ground-truth cuối câu đến sự kiện final
Độ ổn định chữ tạm	Số chữ bị đổi giữa các partial	Tỷ lệ token bị sửa
Lượt bị chốt non	Cắt khi khách còn nói	Dựa trên nhãn "hết lượt" thật
Mức mục tiêu tham khảo (Hamming 🟡): WER âm thanh sạch: xuất sắc < 5%, chấp nhận < 10%, kém > 12%; nhiễu văn phòng: < 8% / < 15% / > 18%; ngoài đường: < 12% / < 20% / > 25%; giọng nặng: < 10% / < 20% / > 25% [S42]. Lưu ý: trên bảng tiếng Việt mà mình đọc, các model mở tốt nhất có O-WER 16,7–20,8% và N-WER 6,8–12,8% (tùy cách chuẩn hóa) [S1] — nhiều ngưỡng trên nếu áp thẳng có thể là không đạt được trên dữ liệu tự nhiên tiếng Việt. Hãy tự đặt ngưỡng từ baseline đo được trên dữ liệu của bạn, rồi siết dần.

9.4 Đo VAD, phát hiện hết lượt và ngắt lời
Đây là phần voice bot khác biệt nhất so với chatbot và hay bị bỏ quên.

Lỗi	Chỉ số	Cách kiểm thử ⚪
Bắt nhầm tiếng không phải lời nói (ồn, TV, tiếng bot vọng)	Tỷ lệ kích hoạt giả/giờ	Phát tiếng ồn nền, âm thanh phát lại; kiểm AEC bằng cách bật loa lớn
Bỏ lỡ lời nói	Tỷ lệ bỏ sót đầu câu; độ trễ nhận ra bắt đầu nói	Câu có mở đầu nhỏ ("dạ…")
Cắt non lượt (người đang nghĩ/ngập ngừng)	Tỷ lệ cắt giữa ý	Câu có nghỉ 0,5–1,5 s giữa các nhóm số; đọc số tài khoản theo nhóm
Chờ quá lâu sau khi nói xong	Độ trễ phát hiện hết lượt (P50/P95)	So với nhãn hết lời tay
Không ngắt được bot	Tỷ lệ barge-in được nhận	Chèn lời nói tại điểm ngẫu nhiên trong câu trả lời của bot (Hamming có phương pháp này [S42])
Ngắt nhầm (tiếng "ừ", ho)	Tỷ lệ barge-in sai	Chèn backchannel "ừ", "vâng" khi bot đang nói
Bot không dừng kịp	Thời gian từ bắt đầu chen lời đến khi audio bot dừng hẳn	Đo ở phía client bằng đồng hồ chung
Hamming đặt mục tiêu: phục hồi sau ngắt lời > 90%, hiệu quả luân phiên > 95%, khoảng lặng ngượng (> 2 s) < 5% số lượt, chồng lấn < 3%, tỷ lệ hỏi lại < 10% [S42] 🟡. Với mô hình đối thoại song công, Full-Duplex-Bench đánh giá bốn nhóm hành vi: xử lý quãng nghỉ, phản hồi đệm (backchannel), luân phiên lượt và quản lý ngắt lời, bằng chỉ số tự động có thể tái lập [S44] (mình chưa xem định nghĩa từng chỉ số trong bài).

Với tiếng Việt, hãy có bộ kiểm thử turn-taking riêng: câu có trợ từ cuối ("ạ, nhé, nhỉ"), tiếng đệm ("ờ, ừm"), đọc số có nghỉ giữa nhóm, trả lời cực ngắn. Smart Turn v3 hỗ trợ tiếng Việt nhưng không công bố độ chính xác theo ngôn ngữ [S21] — mọi tuyên bố về tiếng Việt đều phải do bạn tự kiểm chứng.

9.5 Đo LLM và hội thoại
Tiêu chí	Cách đo	Ghi chú
Hoàn thành tác vụ	So sánh trạng thái cơ sở dữ liệu cuối với trạng thái mong đợi (xác định, không cần LLM chấm)	EVA làm đúng cách này; hạn chế: nhị phân, không tính một phần [S43]
Trung thành với nguồn (faithfulness)	LLM-as-judge kiểm xem câu trả lời có bám chính sách, lời khách, kết quả công cụ không	Cần hiệu chuẩn với nhãn người
Đúng và đủ khi gọi công cụ	Tỷ lệ gọi đúng công cụ, đúng tham số; tỷ lệ thành công	Hamming: > 99% với công cụ quan trọng, > 95% với phụ; tỷ lệ fallback < 5% [S42] 🟡
Phù hợp giọng nói	Độ ngắn gọn, không định dạng thị giác, tiến triển hội thoại	EVA đo "Conciseness" và "Conversation Progression" [S43]
An toàn và tuân thủ	Bộ câu hỏi đối kháng (jailbreak qua giọng nói, yêu cầu ngoài phạm vi, lộ thông tin)	SEA-HELM có cột "an toàn" tiếng Việt cho model [S30]; vẫn phải kiểm trên kịch bản của bạn
Chất lượng tiếng Việt (rubric ⚪)	Xưng hô nhất quán ("em – anh/chị"), dùng "dạ/vâng/ạ" đúng chỗ, từ vựng vùng miền, cách đọc số/địa chỉ tự nhiên	Dùng rubric có thang điểm, người Việt chấm mẫu để hiệu chuẩn LLM-judge
Độ nhất quán qua nhiều lần chạy — chỉ số bị bỏ quên. Model sinh ngẫu nhiên, nên cùng một kịch bản có thể thành công lần này và hỏng lần sau. EVA báo cáo pass@3 (ít nhất 1 trong 3 lần thành công) và pass^3 (cả 3 lần đều thành công) trên 20 hệ thống, và nhận thấy khoảng cách giữa hai chỉ số rất lớn: nhiều tác nhân có thể hoàn thành tác vụ nhưng không làm được nhất quán [S43]. Với sản phẩm cho khách thật, pass^k mới là chỉ số liên quan: khách chỉ gọi một lần. EVA còn báo một đánh đổi nhất quán giữa độ chính xác và trải nghiệm — không cấu hình nào dẫn đầu cả hai [S43], nên cần theo dõi cả hai nhóm chỉ số, không gộp thành một điểm.

LLM-as-judge: dùng nhưng phải kiểm định. Tác giả EVA tự ghi nhận judge có thể thiên lệch [S43]. Quy trình ⚪: (1) người Việt chấm một mẫu 100–200 lượt theo rubric; (2) so khớp với điểm judge (độ đồng thuận, ví dụ hệ số Cohen's κ); (3) chỉ dùng judge khi đồng thuận đủ; (4) lặp lại mỗi khi đổi model judge hoặc prompt.

9.6 Đo TTS
Khía cạnh	Chỉ số	Cách đo
Dễ hiểu	WER vòng tròn: chữ → TTS → STT cố định → so với chữ gốc	Bài báo TTS tiếng Việt cũng dùng WER (cùng MOS, SMOS, MCD, RMSEF0) [S34]. Dùng cùng một STT và chuẩn hóa cho mọi TTS; chú ý lỗi của STT cũng lọt vào
Tự nhiên	MOS (1–5) bởi người nghe Việt	Bài báo trên dùng 10 người nghe cho tập trong phân phối và 20 cho tập ngoài phân phối; khoảng tin cậy khoảng ±0,2–0,4 (có ô tới ±0,44) [S34]. Cần đủ người và đủ câu
Giống người nói (nếu clone giọng)	SMOS	Như trên
Ưa thích tương đối	So sánh mù từng cặp (A/B), tính Elo	Cách Artificial Analysis và Voice Arena làm: bình chọn mù của người nghe [S40, S46]
Đúng thực thể	Đọc đúng số điện thoại, tiền, ngày; tên riêng	EVA có "Speech Fidelity" (LALM-as-judge) kiểm xem âm thanh có khớp chữ dự định, nhất là mã xác nhận và số tiền [S43]
Chuẩn hóa văn bản	Bộ kiểm thử đơn vị: đầu vào → dạng đọc	Bảng ở mục 7.3; chạy tự động mỗi lần nâng cấp
Độ trễ	Thời gian đến audio đầu (TTFA), RTF, phân vị theo tải	VieNeu-TTS: TTFA ~115 ms (1 luồng GPU) nhưng ~450 ms ở 32 luồng [S33] — đo ở mức tải thật
Ổn định	Tỷ lệ audio hỏng: cụt, rè, lặp, im lặng, đọc sai giữa chừng khi chuỗi dài	Chạy hàng nghìn câu ngẫu nhiên và câu dài; phát hiện bằng STT + độ dài mong đợi
Riêng tiếng Việt ⚪: kiểm tra TTS đọc đúng thanh điệu (dùng WER vòng tròn tính theo âm tiết có dấu, và so với WER bỏ dấu thanh để biết lỗi nằm ở thanh), đọc đúng tên người/địa danh Việt, từ mượn tiếng Anh, và lựa chọn giọng vùng miền phù hợp với khách hàng.

9.7 Đo độ trễ đầu–cuối
Định nghĩa trước khi đo. Mình đề xuất: độ trễ lượt = từ thời điểm người dùng thực sự ngừng nói (nhãn thật, không phải thời điểm VAD kết luận) đến byte audio đầu tiên của bot đến được client. Ngoài ra đo thời gian đến từ đầu tiên (TTFW, Hamming đặt mục tiêu < 400 ms [S42]) và độ trễ từng thành phần.

Cách đo:

Gắn dấu thời gian ở client và ở từng bước backend; đồng bộ đồng hồ hoặc dùng độ lệch đã hiệu chỉnh. Nếu chỉ đo ở server, bạn bỏ qua mạng và jitter buffer.
Báo cáo P50, P90, P95, P99 theo từng lát cắt (vùng, loại mạng, độ dài lượt, có gọi công cụ hay không), không báo trung bình.
Tách theo thành phần (STT chốt, TTFT của LLM, TTFA của TTS, mạng) để biết điều chỉnh ở đâu. Bài của Deepgram lưu ý chốt cuối lượt của STT và TTFT thường chiếm phần lớn ngân sách [S25].
Đo trong các hồ sơ mạng: Wi-Fi tốt, 4G yếu, mất gói 1–5%, jitter cao, chuyển mạng giữa chừng. WebRTC và WebSocket phản ứng rất khác nhau với mất gói (mục 3.2) [S23].
Ngưỡng tham khảo (Hamming 🟡, đã gồm điện thoại): lượt P50 < 1,5 s (nghiêm trọng > 1,7 s), P90 < 2,5 s (> 3,0 s), P95 < 3,5 s (> 5,0 s) [S42]. Chính trang này mâu thuẫn ở vài chỗ (một chỗ nói P95 quanh 5 s là điển hình cho pipeline nối tầng) nên hãy dùng như khung tham khảo để tự đặt SLO. Đặt SLO có hai phần: mục tiêu (ví dụ "P95 lượt ≤ 3 s trong 99% số phút") và ngưỡng cảnh báo.

9.8 Mô phỏng hội thoại tự động (bot đấu bot)
Kiểm thử tay không mở rộng được. Phương pháp của EVA là mẫu đáng tham khảo [S43]:

Trình giả lập người dùng đóng một khách có mục tiêu và tính cách cụ thể, nói bằng TTS chất lượng cao, đi theo cây quyết định chính xác;
Tác nhân cần kiểm thử (chạy đúng stack thật — EVA dùng Pipecat);
Bộ thực thi công cụ trả kết quả xác định từ cơ sở dữ liệu theo từng kịch bản;
Bộ kiểm tra hợp lệ đảm bảo cuộc hội thoại hoàn chỉnh và giả lập hành xử đúng; hội thoại hỏng thì tạo lại, không cần gán nhãn tay;
Bộ chỉ số (mục 9.5, 9.6), chạy mỗi kịch bản nhiều lần (k = 3 trong EVA).
Hamming thêm: tiêm nhiễu ở 20/10/5/0 dB, chèn ngắt lời ngẫu nhiên, và so sánh với baseline [S42].

Hạn chế phải kể đến khi áp dụng cho tiếng Việt: bản phát hành EVA chỉ có 50 kịch bản hãng bay tiếng Anh; judge có thể thiên lệch; giả lập dùng một nhà cung cấp thương mại và có thể không phản ánh khách thật [S43]. Với tiếng Việt bạn phải tự viết kịch bản (ngân hàng, bảo hiểm, giao hàng… theo nghiệp vụ của bạn) và tạo nhiều giọng người dùng giả lập (Bắc/Trung/Nam, nam/nữ, nhiều tốc độ), nếu không bộ kiểm thử sẽ chỉ phản ánh một kiểu khách.

9.9 Độ ổn định: tải, soak, hỗn loạn
Loại kiểm thử	Câu hỏi	Cách làm ⚪
Tải	Chất lượng và độ trễ giữ thế nào khi số cuộc gọi đồng thời tăng?	Tăng dần số luồng giả lập đến vài lần dự kiến đỉnh; ghi P95. Dữ kiện cảnh báo: Qwen3-ASR-1.7B có TTFT 102 ms ở concurrency 1 nhưng 3 392 ms (p95 6 227 ms) ở concurrency 128 [S5]; VieNeu-TTS TTFA 115 ms ở 1 luồng, ~450 ms ở 32 luồng, mặc định giới hạn 16 luồng GPU và 1 luồng CPU [S33]
Soak	Có rò rỉ bộ nhớ/kết nối khi chạy nhiều giờ không?	Chạy tải ổn định 4–24 giờ, theo dõi bộ nhớ, số kết nối, độ trễ trôi dần
Giới hạn phiên	Hệ thống xử lý thế nào khi nhà cung cấp cắt phiên?	Ví dụ phiên chỉ-audio của Gemini Live giới hạn 15 phút [S29]; kiểm tra gia hạn/nối phiên
Hỗn loạn (chaos)	Một dịch vụ chết thì bot làm gì?	Tắt/làm chậm STT, LLM, TTS; ngắt WebSocket; trả lỗi 5xx; xem bot có xin lỗi, thử lại, chuyển nhà cung cấp dự phòng hoặc chuyển nhân viên không. Deepgram nhắc kiểm thử kết nối lại, ngắt lời, hành vi lỗi một phần và giới hạn phiên [S25]
Hạ tầng media	Mất gói, jitter, NAT/TURN, đổi mạng	Bộ mô phỏng mạng; kiểm thử ở nhiều nhà mạng
Dự phòng	Chuyển nhà cung cấp mất bao lâu, chất lượng sau chuyển ra sao?	Chạy golden set trên cả đường chính và dự phòng
Từ dự án demo của bạn có một ví dụ nhỏ nhưng điển hình: log cho thấy 40 trong 69 lượt ghi âm lỗi dồn ở hai sự cố (mỗi sự cố 20 lần lỗi liên tiếp trong 10–18 giây) vì nhánh lỗi dùng continue không sleep — vòng lặp tự đập vào lỗi ngay lập tức; stderr của recorder cũng không được ghi lại nên không chẩn đoán được từ log (1-technote.md mục 8.3, 9.3). Đó chính là loại lỗi mà thử nghiệm hỗn loạn và nguyên tắc backoff/timeout/circuit-breaker nhằm ngăn: lỗi nhỏ không được phép biến thành bão thử lại.

Nguyên tắc thiết kế để dễ đạt độ ổn định ⚪: đặt timeout cho từng khâu; thử lại có giới hạn với backoff; circuit-breaker khi một nhà cung cấp lỗi liên tục; hàng đợi nhỏ có giới hạn để không tích lũy trễ; thông điệp lùi bước ("Em xin lỗi, đường truyền đang không ổn, em kết nối anh với nhân viên nhé") thay vì im lặng; chuyển người thật khi lỗi lặp lại; kiểm tra sức khỏe từng thành phần.

9.10 Thống kê: để không tự lừa mình
Khoảng tin cậy. Báo cáo mọi chỉ số cùng khoảng tin cậy; bootstrap theo câu/cuộc gọi (như mã ở 9.3) là cách đơn giản. SEA-HELM cũng dùng bootstrap 2 000 lần và báo khoảng tin cậy 95% [S30] — đáng noi theo.
So sánh có cặp. So hai hệ thống trên cùng các câu và kiểm định chênh lệch theo từng câu; không so hai trung bình độc lập.
Kích thước mẫu. Bộ có vài chục câu chỉ phát hiện được chênh lệch rất lớn. Mỗi lát cắt cũng cần đủ mẫu — báo cáo kích thước mẫu cùng số liệu.
Báo cáo theo lát cắt. Một cải thiện trung bình có thể che một lát cắt tệ đi (ví dụ giọng Trung hoặc điện thoại 8 kHz).
pass^k cho hành vi ngẫu nhiên (mục 9.5).
Đừng tối ưu quá mức cho bộ kiểm thử. AssemblyAI tự cảnh báo rằng benchmark công khai dễ gây hiểu lầm do overfitting và "gaming" và khuyên chạy đánh giá riêng trên âm thanh của bạn [S11]. Giữ một bộ giữ kín (held-out) không dùng khi chỉnh model/prompt.
CSAT/khảo sát chỉ là tín hiệu phụ: Hamming ghi tỷ lệ phản hồi khảo sát chỉ 5–10% [S42].
9.11 Cổng hồi quy và triển khai an toàn
Cổng hồi quy tự động trước khi triển khai: so golden set mới với baseline và chặn nếu vượt dung sai. Hamming gợi ý ±10% cho độ trễ, ±3% cho hoàn thành tác vụ, ±2% cho WER [S42] 🟡 — hãy chỉnh theo khoảng tin cậy thực tế của bạn.
Ghim phiên bản model. Dịch vụ đám mây có thể đổi hành vi khi cập nhật ⚪; ghim phiên bản (hoặc ít nhất chạy lại golden set hằng đêm) để phát hiện thay đổi âm thầm.
Shadow rồi canary. Chạy hệ thống mới song song trên lưu lượng thật (không phục vụ khách) để so sánh; sau đó cho một phần nhỏ khách thật qua đường mới với khả năng quay lại tức thì ⚪.
A/B thí nghiệm cho thay đổi ảnh hưởng trải nghiệm (đổi TTS, đổi ngưỡng turn detection), với chỉ số chính xác định trước.
9.12 Giám sát sau khi ra mắt
Hamming gợi ý: cuộc gọi tổng hợp liên tục mỗi 5–15 phút, chấm điểm cuộc gọi thật, bảng điều khiển làm mới mỗi 5 phút; cảnh báo ví dụ — P95 độ trễ cảnh báo > 5 s và nghiêm trọng > 7 s; hoàn thành tác vụ cảnh báo < 80%, nghiêm trọng < 70%; WER cảnh báo > 12%, nghiêm trọng > 18%; tỷ lệ lỗi cảnh báo > 5%, nghiêm trọng > 10% [S42] 🟡. Hãy coi đó là mẫu và hiệu chỉnh theo baseline của bạn.

Những thứ nên có ⚪:

Nhóm	Gồm
Sức khỏe hạ tầng	tỷ lệ lỗi theo thành phần, mất gói/jitter, số phiên đồng thời, thời gian chờ hàng đợi
Chất lượng thời gian thực	phân vị độ trễ theo thành phần, tỷ lệ barge-in, tỷ lệ im lặng ngượng, tỷ lệ hỏi lại, tỷ lệ chuyển người
Chất lượng nội dung	điểm LLM-judge trên mẫu cuộc gọi thật, tỷ lệ hoàn thành tác vụ, tỷ lệ fallback
Trôi dữ liệu	phân bố giọng/vùng/kênh/nhiễu thay đổi theo thời gian; từ vựng mới (sản phẩm mới)
Kinh doanh	chứa trong bot (containment), giải quyết ngay lần đầu, leo thang; Hamming nêu các mốc như containment > 70%, FCR > 75% (hỗ trợ), leo thang < 25% [S42] 🟡
Vòng phản hồi (quan trọng nhất): mọi cuộc gọi thất bại (khách bực, chuyển người, lỗi công cụ) được đưa vào golden set sau khi gán nhãn → lỗi đó không bao giờ lặp lại âm thầm. Kèm theo: duyệt ngẫu nhiên một tỷ lệ cuộc gọi bởi người (và duyệt toàn bộ cuộc gọi bị chuyển người/khiếu nại) ⚪; che thông tin cá nhân trong bản chữ trước khi người xem; tuân thủ thời hạn lưu trữ.

9.13 Bảng tổng hợp: chỉ số ↔ cách đo ↔ mức khởi điểm
Các mức dưới đây là điểm khởi đầu để chỉnh (nguồn: Hamming 🟡 trừ khi ghi khác), không phải chuẩn ngành.

Tầng	Chỉ số	Cách đo	Mức khởi điểm tham khảo
STT	WER (có/không dấu thanh), CER, entity accuracy, HER, P90 WER	Golden set theo lát cắt, chuẩn hóa cố định	Tự đặt từ baseline; xem lưu ý ở 9.3
STT	TTFP, độ trễ chốt, ổn định partial	Dấu thời gian streaming	ASR < 300 ms, nghiêm trọng > 500 ms
VAD/Turn	Kích hoạt giả/h, bỏ sót, cắt non, trễ hết lượt	Kịch bản turn-taking tiếng Việt	Tự đặt; turn-taking > 95%, overlap < 3%
Ngắt lời	Phục hồi sau barge-in, barge-in sai	Chèn ngắt lời ngẫu nhiên	> 90%
LLM	Hoàn thành tác vụ, faithfulness, tool-call success, pass^k	Mô phỏng bot-to-bot, nhiều lần chạy	Tool-call > 99% (quan trọng); task success tùy nghiệp vụ (hỗ trợ > 75%)
LLM	TTFT	Đo ở server	< 400 ms, nghiêm trọng > 600 ms
TTS	Round-trip WER, MOS/SMOS, đúng thực thể, đúng chuẩn hóa	Nghe + STT cố định + bộ unit test	Tự đặt; entity 100% trên bộ kiểm thử chuẩn hóa
TTS	TTFA, RTF ở mức tải thật	Dấu thời gian	< 200 ms, nghiêm trọng > 400 ms
Hệ thống	Độ trễ lượt P50/P90/P95/P99, TTFW	Đo ở client, theo lát cắt mạng	P50 < 1,5 s; P90 < 2,5 s; P95 < 3,5 s; TTFW < 400 ms
Hệ thống	Tỷ lệ lỗi, khả dụng, hành vi khi lỗi	Tải, soak, chaos	Lỗi hệ thống < 1%
Kinh doanh	Containment, FCR, leo thang	Từ log và CRM	> 70% / > 75% / < 25% (tùy nghiệp vụ)
10. Áp dụng vào dự án demo của bạn: lộ trình từ "echo" đến voice bot
Dự án 03_voice_stt_tts hiện là một pipeline echo (Mic → STT → TTS → Loa) gồm: faster-whisper base int8 trên CPU, Piper vi_VN-vais1000-medium, WebRTC VAD, ghi âm xong mới gửi HTTP (không streaming), dừng ghi sau 3 s im lặng, không có LLM (xem 1-technote.md). Log 11 phiên cho thấy: 17/27 lượt STT trả chuỗi rỗng; 40/69 lượt ghi âm lỗi (dồn ở hai sự cố); 4/10 lượt phát âm thanh lỗi thiết bị; STT trung vị 0,80 s cho đoạn có lời (5–19,8 s); TTS trung vị 0,16 s; chưa có số WER hay đánh giá chất lượng giọng nào. Đó là nền tốt cho một prototype; dưới đây là thứ tự việc mình đề xuất, xếp theo "giá trị trên công sức".

Bước	Việc	Vì sao đặt ở đây	Đo bằng (mục 9)
0	Sửa quan sát được: ghi stderr của recorder, ghi RMS/độ dài lời nói mỗi bản ghi, thêm sleep/backoff khi lỗi	Hiện không phân biệt được "mic im" với "Whisper không nhận ra" và các lỗi dồn thành bão thử lại (1-technote.md mục 8.3, 10)	Tỷ lệ bản ghi rỗng, tỷ lệ lỗi theo thiết bị
1	Xây bộ đánh giá trước khi đổi model: 100–200 câu tiếng Việt có nhãn (đọc + vài hội thoại), chia lát cắt giọng/nhiễu/số; dùng mã WER ở mục 9.3	Không có thước đo thì mọi nâng cấp sau đều là cảm tính. Đây là hạng mục rẻ nhất và có lợi nhất	WER (có/không dấu thanh), entity accuracy, CI bootstrap
2	Đổi VAD và rút thời gian chờ: Silero VAD thay WebRTC VAD; giảm 3 s xuống mức dưới 1 s (cấu hình theo bước hội thoại); thử Smart Turn v3	Riêng 3 s chờ đã đủ làm độ trễ cảm nhận ≥ 3 s + xử lý, vượt xa mục tiêu P50 1,5 s (mục 1.3, 2.4)	Cắt non, trễ hết lượt, độ trễ lượt
3	Nâng STT theo số đo bước 1: PhoWhisper-small/medium, Qwen3-ASR (streaming), hoặc Chirp 3	Whisper base (74 M, tổng quát) rất có thể chưa phải lựa chọn tốt cho tiếng Việt (xem bên dưới)	WER, TTFP, độ trễ chốt
4	Thêm LLM streaming + bộ cắt câu (mã ở mục 7.2) — biến "echo" thành voice bot	Đây mới là bước tạo ra giá trị sản phẩm; từ đây có thể đo task completion	TTFT, hoàn thành tác vụ, pass^k
5	TTS streaming + chuẩn hóa văn bản: thử VieNeu-TTS; bộ kiểm thử đơn vị cho số/ngày/tiền; so với Piper bằng nghe mù và round-trip WER	Piper tổng hợp cả câu rồi mới trả file; cần streaming để rút TTFA	TTFA, MOS, entity đúng
6	Truyền streaming và ngắt lời: WebSocket hai chiều cho thử nghiệm mạng nội bộ; WebRTC (Pipecat hoặc LiveKit Agents) khi ra thiết bị thật; AEC bật	Barge-in chỉ làm tốt khi có AEC và hủy luồng hai đầu (mục 7.4)	Barge-in recovery, kích hoạt giả
7	Ổn định và an toàn: API key, giới hạn upload, mã hóa/xóa tự động bản ghi, thêm output/mic_record/ và output/pipeline_logs/ vào .gitignore, test tự động, Docker	Các rủi ro đã nêu ở 1-technote.md mục 9; cần trước khi chạm dữ liệu khách thật	Tải, soak, chaos (mục 9.9)
Một nhận xét về Whisper base cho tiếng Việt (suy luận có số liệu, không phải đo trực tiếp): bài báo PhoWhisper cho thấy kích thước model ảnh hưởng mạnh ở dải nhỏ. Cùng bốn tập thử, PhoWhisper-base (74 M) đạt CMV-Vi 16,19 / VIVOS 8,46 / VLSP Task-1 19,70 / Task-2 43,01, còn PhoWhisper-small (244 M) đạt 11,08 / 6,33 / 15,93 / 32,96 — giảm lỗi khoảng 19–32% [S2]. Dự án của bạn dùng Whisper base gốc (không phải PhoWhisper-base), và mình không có số WER của Whisper base gốc trên tiếng Việt từ các nguồn đã đọc. Vì vậy kết luận đúng là: nhiều khả năng còn nhiều dư địa cải thiện bằng cách tăng cỡ hoặc dùng model tiếng Việt, nhưng hãy đo ở bước 1 trước khi quyết định. Cũng lưu ý bản PhoWhisper trong repo chỉ có hướng dẫn transformers; việc chuyển sang định dạng CTranslate2 để dùng với faster-whisper chưa được xác minh [S3].

Khung quyết định ngắn gọn cho lựa chọn kiến trúc của bạn:

Nếu ưu tiên…	Chọn…
Làm nhanh một sản phẩm chạy được, chấp nhận phụ thuộc nhà cung cấp	Speech-to-speech (Gemini Live có tiếng Việt chính thức) hoặc ghép STT/TTS đám mây
Kiểm soát chất lượng tiếng Việt, dữ liệu trong nước, kiểm toán	Cascaded tự host: STT mở (PhoWhisper/Qwen3-ASR) + LLM mở (Gemma/Qwen) + VieNeu-TTS
Điện thoại tại Việt Nam	Cascaded + SIP; đo riêng trên âm thanh 8 kHz; chuẩn bị chuyển người
Chạy trên thiết bị biên/riêng tư cao	STT nhỏ (Zipformer-30M, Qwen3-ASR-0.6B) + TTS nhỏ — cần xác minh streaming và giấy phép
11. Kết luận và những điều chưa kiểm chứng
11.1 Mười điều rút ra
Cascaded (STT → LLM → TTS) là mặc định hợp lý cho tiếng Việt vì cho phép chọn model tốt nhất từng khâu, tự host và kiểm toán; speech-to-speech là lựa chọn nhanh nhưng bằng chứng tiếng Việt còn mỏng (chỉ Gemini Live nêu tiếng Việt chính thức trong ba ứng viên mình xem).
Truyền âm thanh lên máy chủ, không truyền chữ (trừ khi cần chạy STT trên thiết bị vì riêng tư). Dùng WebRTC cho trình duyệt/app và SIP cho điện thoại; WebSocket/gRPC chỉ cho liên lạc giữa các dịch vụ hoặc mạng tốt.
Kafka không phải kênh truyền âm thanh. Nó thuộc mặt phẳng sự kiện (bản chữ, số đo, kiểm toán, QA), nơi độ bền quan trọng hơn độ trễ từng mili-giây.
VAD + phát hiện hết lượt quyết định cảm giác "tự nhiên" nhiều hơn model STT. Silero VAD là điểm khởi đầu; Smart Turn v3 hỗ trợ tiếng Việt, còn bộ phát hiện lượt của LiveKit và Deepgram Flux thì không.
Không có bảng xếp hạng STT tiếng Việt trung lập nào bao trùm cả model mở lẫn thương mại. Trong các model mở có số đo: PhoWhisper-medium, Zipformer-30M, PhoWhisper-small, Qwen3-ASR-1.7B, parakeet-ctc-0.6b-Vietnamese. Số của STT thương mại là do nhà cung cấp tự công bố.
Về LLM: top model mở theo SEA-HELM tiếng Việt là Gemma 4 31B, Qwen 3.5 122B-A10B, Qwen 3.6 27B, SEA-LION v4.5 27B, Muse Glimmer 30B — chênh lệch nhỏ giữa các hạng đầu; model đóng chưa có bảng tiếng Việt trung lập.
TTS tiếng Việt: VieNeu-TTS là lựa chọn mở có streaming nhưng chưa có MOS/WER công bố; các TTS đám mây (Cartesia, ElevenLabs, Google) có trang tiếng Việt; nhà cung cấp trong nước chưa có đánh giá độc lập. Bộ cắt câu và chuẩn hóa văn bản quyết định chất lượng cảm nhận không kém model.
"Nhả chữ" phải chia hai luồng: chữ cho TTS (cắt mệnh đề rồi cắt câu) và chữ cho giao diện (bám theo audio đã phát); khi bị ngắt phải biết bot thực sự đã nói tới đâu.
Đánh giá theo tầng, theo lát cắt, có khoảng tin cậy: WER tách dấu thanh, độ chính xác thực thể, kiểm thử turn-taking, độ trễ phân vị đo ở client, mô phỏng bot-to-bot với pass^k, kiểm thử tải/soak/chaos, cổng hồi quy, giám sát sản xuất và vòng phản hồi đưa cuộc gọi lỗi vào bộ kiểm thử.
Việc đầu tiên cần làm không phải đổi model mà là xây thước đo. Bộ kiểm thử 100–200 câu tiếng Việt của chính bạn sẽ đáng giá hơn mọi bảng xếp hạng trong bài này.
11.2 Những gì mình không kiểm chứng được hoặc có thể sai
Mục	Tình trạng
Bảng STT tiếng Việt của Voice Arena	Trang tải bằng JavaScript, mình chỉ đọc được phần mô tả (có tiếng Việt); không có số liệu. Có thể là nguồn tốt nhất — hãy mở trực tiếp [S46]
Multilingual TTS Arena của Artificial Analysis	Bài thông báo trên X bị chặn truy cập; mình không biết bảng có xếp hạng tiếng Việt hay không [S45]. Bảng Speech Arena chính không có bộ lọc ngôn ngữ ở phần mình đọc [S40]
Trang model Zipformer-30M-RNNT-6000h	Không đọc được (bị chặn); khả năng streaming và giấy phép chưa xác nhận
Tài liệu API TTS WebSocket của ElevenLabs	Không đọc được; phần mô tả "đầu vào streaming" ở mục 7.2 là hiểu biết chung ⚪
Tài liệu Qwen-Audio-3.0-ASR (arXiv 2609.07549)	PDF rỗng, không đọc được; không đưa vào bài
Số STT thương mại	Là số của nhà cung cấp, trên model cũ (Scribe v1, Gemini 2.0 Flash, Nova-2); model mới không có số tiếng Việt công khai mà mình tìm được
Model đóng (GPT, Gemini, Claude…) về LLM tiếng Việt	Không có bảng tiếng Việt trung lập đã đọc
Nhà cung cấp trong nước (FPT.AI, Viettel AI, VAIS, Zalo, Vbee)	Không có đánh giá độc lập mới; bài so sánh học thuật cũ chỉ thấy tiêu đề
Pháp lý dữ liệu cá nhân (Nghị định 13/2023, Luật Bảo vệ dữ liệu cá nhân)	Hiểu biết chung ⚪, chưa tra cứu văn bản; cần luật sư
Chuyển PhoWhisper sang faster-whisper/CTranslate2	Chưa xác minh khả thi
Giấy phép Silero VAD	Trang ghi MIT nhưng có huy hiệu CC BY-NC 4.0 mâu thuẫn — kiểm tra trước khi dùng thương mại [S20]
Các mục đánh dấu ⚪	Kinh nghiệm chung của người viết, chưa đối chiếu nguồn trong lần nghiên cứu này
Dữ liệu trong lĩnh vực này đổi theo từng tuần (nhiều model trong bài ra mắt trong vài tháng gần đây). Mọi số liệu ở đây là ảnh chụp tại thời điểm viết (tháng 10/2026); hãy chạy lại bước 1 của mục 10 trước các quyết định lớn.

Nguồn
Nhãn: 🟢 nguồn có thể kiểm chứng/không phải nhà bán; 🟡 do nhà cung cấp hoặc tác giả model tự công bố; ⚪ chỉ thấy tiêu đề hoặc không đọc được nội dung.

STT và đánh giá STT

[S1] Vietnamese Open ASR Leaderboard 🟢
[S2] PhoWhisper: Automatic Speech Recognition for Vietnamese (arXiv 2406.02555) 🟢
[S3] VinAIResearch/PhoWhisper (GitHub) 🟢
[S4] VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining (arXiv 2505.21527) 🟢
[S5] Qwen3-ASR Technical Report (arXiv 2601.21337) 🟡
[S6] nvidia/parakeet-ctc-0.6b-Vietnamese (Hugging Face) 🟡
[S7] Qwen3-ASR 1.7B vs Whisper Large V3 Turbo — Whisper Notes 🟡 (tự chạy)
[S8] ElevenLabs — Vietnamese Speech to Text 🟡
[S9] ElevenLabs — Real-time Speech to Text under 200 ms 🟡
[S10] Better Vietnamese: find a fine-tune that beats large-v3 (babelscribe issue #1) 🟡
[S11] AssemblyAI — Pre-recorded audio benchmarks 🟡
[S12] Google Cloud — Chirp 3 Transcription 🟢
[S13] Microsoft — Speech service language support 🟢
[S14] Deepgram — Models and languages overview 🟢
[S15] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation (arXiv 2510.06961) 🟢
[S16] Artificial Analysis — Speech to Text leaderboard 🟢
[S17] Back to Basics: Revisiting ASR in the Age of Voice Agents — WildASR (arXiv 2603.25727) 🟢
[S18] VietSuperSpeech: A Large-Scale Vietnamese Conversational Speech Dataset (arXiv 2603.01894) 🟢
VAD, phát hiện hết lượt, truyền tải, khung điều phối

[S19] TEN VAD (GitHub) 🟡
[S20] Silero VAD (GitHub) 🟢
[S21] Pipecat Smart Turn (GitHub) 🟢
[S22] LiveKit — Turn detector 🟢
[S23] LiveKit — Why WebRTC beats WebSockets for voice AI agents 🟡
[S24] OpenAI — Delivering low-latency voice AI at scale 🟢
[S25] Deepgram — The voice agent architecture guide (bản tóm tắt trên Plushcap) 🟡
[S26] LiveKit — LiveKit vs Pipecat 🟡
[S27] Confluent — Benchmark testing and results for Apache Kafka's performance 🟡
[S28] OpenAI — Realtime API guide 🟢
[S29] Google — Live API capabilities guide 🟢
[S32] OpenAI vs Gemini vs Qwen Realtime Voice API 2026 — Tech Insider 🟡 (nguồn thứ cấp)
LLM tiếng Việt

[S30] SEA-HELM — Vietnamese leaderboard (18/09/2026) 🟢
[S31] VMLU Leaderboard 🟢
TTS và chuẩn hóa văn bản

[S33] VieNeu-TTS (GitHub) 🟡
[S34] Zero-Shot Text-to-Speech for Vietnamese (arXiv 2506.01322) 🟢
[S35] QwenLM/Qwen3-TTS (GitHub) 🟡
[S36] Cartesia — Native Vietnamese Text to Speech 🟡
[S37] ElevenLabs — Vietnamese Text to Speech 🟡
[S38] Google Cloud — Chirp 3: HD voices 🟢
[S39] Vbee — Top phần mềm chuyển văn bản thành giọng nói 🟡 (do một nhà cung cấp được xếp hạng đăng)
[S40] Artificial Analysis — Speech Arena (Text to Speech) leaderboard 🟢
[S41] Digital Applied — Best text-to-speech models, September 2026 🟡 (nguồn thứ cấp)
[S45] Artificial Analysis — Multilingual Text to Speech Arena (bài thông báo trên X) ⚪ không đọc được
[S46] Voice Arena — Vietnamese STT leaderboard ⚪ chỉ đọc được phần mô tả
[S47] VietNormalizer: An Open-Source, Dependency-Free Python Library for Vietnamese Text Normalization (arXiv 2603.04145) 🟢
Đánh giá voice bot

[S42] Hamming — How to Evaluate Voice Agents 🟡 (nhà bán công cụ kiểm thử; số liệu từ dữ liệu của họ)
[S43] ServiceNow-AI — A New Framework for Evaluating Voice Agents (EVA) 🟢
[S44] Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities (arXiv 2503.04721) 🟢 (chỉ đọc phần tóm tắt)
