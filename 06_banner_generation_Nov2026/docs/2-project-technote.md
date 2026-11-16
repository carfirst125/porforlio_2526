# 2. Project Technote — DCO Dynamic Creative Optimization (`version_1`)

## 1. Kiến trúc tổng thể

```
request.json ──► [LangGraph pipeline] ──► shared prompt ──► per (model, version) ──► crop ──► output/*.png
                   1 load_request              (built once,     generate_image()        │
                   2 resolve_assets              cached &        via app/providers/       │
                   3 load_guideline               reused)                                  │
                   4 prepare_images                                                          ▼
                   5 build_prompt (─ optional LLM refine, app/llm/) ─────────────► output/<req>/prompt/*.txt (audit log)

Streamlit webapp (webapp/) ──HTTP──► FastAPI (app/api/routes.py) ──► app/core/pipeline.py ──► app/providers/* (image) + app/llm/* (text)
```

`app/core/pipeline.py` runs the LangGraph graph (steps 1-5, REQUEST.md [1])
exactly ONCE per `/generate` call to build 1 shared prompt, then loops
`model × version` OUTSIDE the graph calling `app/providers/registry.py` per
pair — this is what lets 1 failed (model, version) stay isolated from the
rest (REQUEST.md's "dump exception, never crash the whole run") and lets the
SAME prompt text be reused across every model in the run (REQUEST.md [3d]).

**Why `app/llm/` is separate from `app/providers/`:** `app/llm/` is the
generic, swappable TEXT-generation abstraction from the
`nhantn-mlai-project-structure` standard (`BaseLLMProvider` + `LLMFactory`,
`generate(prompt: str) -> str`) — used ONLY for the optional prompt-refinement
step. `app/providers/` is this app's own domain-specific IMAGE-generation
layer: its interface takes `(prompt, attachments, width, height)` and returns
image bytes, which doesn't fit the generic text-LLM shape, and is core
business logic specific to this app rather than a generic swappable utility —
so it stays a sibling of `app/core/`, not a specialization of `app/llm/`.

## 2. Cây thư mục đầy đủ

```
version_1/
├── 1-project_description.md      what/why/scope (this doc's sibling)
├── 2-project-technote.md          you are here
├── 3-project-evaluation.md          test methodology + pass/fail thresholds
├── 4-project-evaluation-results.md    change log + REAL measured results
├── README.md                            short pointer + fastest-path run commands
├── .env.example                          template, safe to commit
├── .gitignore
├── requirements.txt                       runtime deps (backend + webapp)
├── requirements-dev.txt                    + pytest (app/evaluation/ only)
├── pytest.ini
├── Dockerfile                               multi-stage, backend image only
├── .dockerignore
├── docker-compose.yml                        backend + webapp (+ optional ollama profile)
├── run_cli.py                                 REQUEST.md [4] --mode= CLI runner
├── app/
│   ├── settings.py             pydantic-settings Settings + get_settings() -- THE config source
│   ├── logging_conf.py           console + output/_logs/app.log logging setup
│   ├── main.py                     FastAPI app (uvicorn entrypoint)
│   ├── models/
│   │   └── schemas.py                 request.json pydantic models (CreativeRequest, ...)
│   ├── api/
│   │   └── routes.py                    every endpoint webapp/utils/api_client.py calls
│   ├── core/                              business logic (REQUEST.md steps [1]-[6])
│   │   ├── guidelines.py                    guidelines.json lookup + general_guide recursion
│   │   ├── asset_resolver.py                  buckets/reference -> real files under ASSET_LIBRARY_DIR
│   │   ├── image_prep.py                        downsample attachments (REQUEST.md [1] step [4])
│   │   ├── prompt_builder.py                      assemble + cache the final prompt (REQUEST.md [3]/[5])
│   │   ├── image_post.py                            crop generation.safe_area out of the generated canvas
│   │   ├── output_manager.py                          ALL output/prompt/release path + naming conventions
│   │   ├── model_usage.py                               config/model_usage.json -> resolved model list per mode
│   │   └── pipeline.py                                    the LangGraph workflow + generate/clone execution loops
│   ├── llm/                                 swappable TEXT-LLM layer (nhantn-mlai-project-structure standard)
│   │   ├── base.py                            BaseLLMProvider abstract interface
│   │   ├── factory.py                           LLMFactory (config -> provider instance)
│   │   └── providers/
│   │       ├── openai_provider.py                 OpenAI-compatible (real: Azure AI Foundry gpt-5-mini)
│   │       ├── claude_provider.py                   Anthropic Messages API
│   │       ├── llama_provider.py                      hosted-cloud Llama (Groq/Together/Bedrock-style)
│   │       ├── deepseek_provider.py                     DeepSeek chat API
│   │       └── ollama_provider.py                         self-host (no API key needed)
│   ├── providers/                          this app's own IMAGE-generation layer (see section 1 above)
│   │   ├── registry.py                       model name -> provider client + Settings wiring
│   │   ├── gemini_image.py                     Vertex AI Gemini image models
│   │   ├── azure_openai_image.py                 gpt-image-2 (Azure OpenAI images/generations + /edits)
│   │   ├── azure_generic_image.py                  MAI-Image-2.5/-Flash/-Pro, FLUX.2-Pro (shared client)
│   │   ├── base.py, _http.py                         shared types + retrying HTTP helper
│   └── evaluation/                           pytest suite (see 3-project-evaluation.md)
│       ├── conftest.py                          shared fixtures (isolated_output, fake_generate_image)
│       ├── unit/                                  no network/filesystem-outside-tmp, pure logic
│       ├── smoke/                                   full API round-trip, real fixture data, mocked provider
│       └── stress/                                    concurrency/load (excluded from default `pytest` run)
├── script/                              one-time/prep utilities, NOT runtime app logic
│   ├── check_env.py                        validate .env / config completeness (+ optional --ping)
│   └── validate_requests.py                  validate every input/*.json before a big batch run
├── webapp/                             Streamlit frontend (pre-existing; only `views/generate.py`'s
│                                         `_expected_display_size()` was touched, to stay in sync with
│                                         the backend's request.json schema -- see 4-project-evaluation-results.md)
├── config/model_usage.json             mode -> model list (REQUEST.md [4])
├── input/*.json                        request.json files
├── lib/                                 asset library (buckets + guidelines source images)
├── output/                               generated images + prompts + release (git-ignored)
└── prompts/design_rules.md               fixed MUST/NOT-MUST rules embedded in every prompt
```

## 3. `Settings` — bảng tham số

Mọi field dưới đây đọc qua `app.settings.get_settings()`; KHÔNG module nào
khác đọc thẳng `os.environ`. Field không có trong `.env` dùng default liệt kê.

| Field | Kiểu | Default | Ý nghĩa |
|---|---|---|---|
| `app_name` | str | `dco-dynamic-creative-opt` | Tên app (hiện ở `/health`, FastAPI title) |
| `environment` | str | `development` | development / staging / production |
| `log_level` | str | `INFO` | Mức log console + file |
| `asset_library_dir` | Path | `./lib` | Thư mục gốc chứa bucket asset + guideline images |
| `input_dir` | Path | `./input` | Thư mục chứa `request.json` |
| `output_dir` | Path | `./output` | Thư mục ghi ảnh/prompt/release |
| `script_output_dir` | Path | `./script/output` | Output riêng cho `script/*.py` (nếu cần) |
| `prompt_rules_file` | Path | `./prompts/design_rules.md` | File MUST/NOT-MUST cố định |
| `api_host` / `api_port` | str / int | `0.0.0.0` / `8000` | uvicorn bind |
| `backend_base_url` | str | `http://localhost:8000` | Webapp dùng để gọi API |
| `llm_provider` | str | `openai` | `openai\|claude\|llama\|deepseek\|ollama` (+ alias `azure_gpt5mini`) |
| `llm_model_name` | str | `gpt-5-mini` | Model name gửi cho provider |
| `llm_temperature` / `llm_max_tokens` | float / int | `0.2` / `2048` | Tham số sinh text |
| `llm_base_url` | str\|None | `None` | Override endpoint (self-host/custom) |
| `use_llm_prompt_refinement` | bool | `true` | Bật/tắt bước gpt-5-mini tinh gọn prompt |
| `openai_api_key` / `anthropic_api_key` / `deepseek_api_key` / `llama_api_key` | str\|None | `None` | Credential từng LLM provider |
| `ollama_base_url` | str | `http://localhost:11434` | Ollama instance |
| `az_gpt5mini_endpoint` / `az_gpt5mini_key` | str | `""` | Azure AI Foundry gpt-5-mini (fallback cho `openai` provider) |
| `gcp_gemini_31_flash_image_endpoint` / `gcp_gemini_3_pro_image_endpoint` / `gcp_vertex_api_key` | str | `""` | Gemini image models |
| `az_mai_image_25*_endpoint/key/deployment` | str | `""` / model name | MAI-Image-2.5/-Flash/-Pro |
| `az_flux_2_pro_*` | str | `""` / `flux-2-pro` | FLUX.2-Pro |
| `az_gpt_image_2_endpoint` / `az_gpt_image_2_key` | str | `""` | GPT-Image-2 |
| `http_timeout_seconds` / `http_max_retries` | int | `600` / `3` | Mọi HTTP call ra ngoài (app/providers, app/llm) |
| `attach_image_max_dim` / `attach_image_jpeg_quality` | int | `1024` / `85` | Downsample ảnh đính kèm |
| `guidelines_file` / `model_usage_file` | Path (computed) | — | Luôn cố định `lib/guidelines/guidelines.json` / `config/model_usage.json`, không cấu hình qua `.env` |

## 4. Hướng dẫn end-to-end

```bash
cd version_1
cp .env.example .env        # rồi điền giá trị thật (hoặc dùng .env chung ở thư mục cha)
pip install -r requirements.txt

# kiểm tra config trước khi chạy thật (không tốn API)
python script/check_env.py
python script/validate_requests.py

# Terminal 1 -- backend
uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000

# Terminal 2 -- frontend
streamlit run webapp/Home.py
```

**QUAN TRỌNG -- `--reload-dir app` (2026-08-20, user-reported bug: FastAPI
tự dưng đóng kết nối giữa chừng, phía frontend nhận
`ConnectionResetError`/"connection aborted" khi gọi `GET
/requests/<request_name>`):** mặc định, `uvicorn --reload` (KHÔNG có
`--reload-dir`) theo dõi thay đổi trên TOÀN BỘ cây thư mục hiện tại
(`version_1/`) để tự restart server -- bao gồm cả `input/`, `output/`,
`lib/`, những thư mục mà CHÍNH app này liên tục ghi vào trong lúc chạy
(prompt log, ảnh cache đã downsample, ảnh safe_area_guide cache, và đặc
biệt là chính file log của app tại `output/_logs/app.log`). Kết quả: app
tự ghi log -> uvicorn thấy file thay đổi -> tự restart -> restart lại ghi
thêm log -> lại restart tiếp, cứ thế lặp lại; hoặc đơn giản là user thêm
1 file `request.json` mới vào `input/` để test cũng đủ trigger 1 lần
restart giữa lúc frontend đang gọi API, khiến kết nối TCP bị đóng đột
ngột đúng lúc request đang chạy dở. Bằng chứng thấy trong `app.log`: 2
dòng "starting" liên tiếp cách nhau vài giây, không có dòng "shutting
down" hợp lệ ở giữa -- dấu hiệu điển hình của vòng lặp tự-restart.

Thêm `--reload-dir app` giới hạn uvicorn chỉ theo dõi thay đổi trong
`app/` (mã nguồn thực sự), không còn tự restart khi có file mới trong
`input/`/`output/`/`lib/` nữa. Nếu server đang chạy mà không có
`--reload-dir app`, cần dừng và khởi động lại đúng câu lệnh trên.

Chạy CLI trực tiếp không cần webapp (REQUEST.md [4] `--mode=` syntax):

```bash
python run_cli.py DB_NOTI_AUTOPAY_NOEL --mode=default
python run_cli.py DB_NOTI_AUTOPAY_NOEL --mode=quality
python run_cli.py DB_NOTI_AUTOPAY_NOEL --mode=custom,gemini-3-pro-image,gpt-image-2
python run_cli.py DB_NOTI_AUTOPAY_NOEL --mode=all
python run_cli.py DB_NOTI_AUTOPAY_NOEL --mode=clone,output/DB_NOTI_AUTOPAY_NOEL/release/<file>.png,gpt-image-2
```

Test 1 model image thật riêng lẻ, không cần chạy cả app:

```bash
python -m app.providers.registry --model gpt-image-2 --prompt "a red apple" --width 512 --height 512 --out test.png
python -m app.llm.factory --prompt "Xin chào"   # test LLM provider hiện đang chọn theo .env
```

Chạy evaluation suite (xem `3-project-evaluation.md`):

```bash
pip install -r requirements-dev.txt
pytest              # unit + smoke (mặc định, ~6s, không tốn API)
pytest -m stress -v # + stress (thêm ~5s)
```

## 5. Docker build & push

```bash
# Build image (đặt tên theo <docker-hub-username>/<app-name>:<tag>)
docker build -t nhanngo/dco-dynamic-creative-opt:1.0.0 .

# Chạy thử local, truyền biến môi trường qua --env-file
docker run --rm -p 8000:8000 --env-file .env nhanngo/dco-dynamic-creative-opt:1.0.0

# Hoặc chạy cả backend + webapp cùng lúc
docker compose up

# Đăng nhập Docker Hub (chạy 1 lần, hoặc dùng access token thay password)
docker login

# Gắn thêm tag latest (tuỳ chọn) rồi push cả 2 tag lên Docker Hub
docker tag nhanngo/dco-dynamic-creative-opt:1.0.0 nhanngo/dco-dynamic-creative-opt:latest
docker push nhanngo/dco-dynamic-creative-opt:1.0.0
docker push nhanngo/dco-dynamic-creative-opt:latest
```

Quy ước tag: semantic version (`MAJOR.MINOR.PATCH`), bump `PATCH` cho fix,
`MINOR` cho thêm model/tính năng không phá tương thích. Không có build
riêng cho GPU/CPU (app không chạy inference cục bộ, chỉ gọi API ngoài).

Đánh đổi có chủ đích: `Dockerfile` cài `requirements.txt` ĐẦY ĐỦ (gồm cả
`streamlit`) dù backend image không copy `webapp/` vào — để
`docker-compose.yml`'s `webapp` service dùng LẠI đúng image này (chỉ mount
thêm `webapp/` + đổi `command`) chạy được ngay, không cần build riêng.
Ưu tiên "chạy được ngay, ít Dockerfile cần bảo trì" hơn là tối ưu vài trăm
KB kích thước image ở quy mô project cá nhân này — tách riêng
`requirements-backend.txt` (không streamlit) + Dockerfile/target riêng cho
webapp là việc có thể làm sau nếu kích thước image thực sự thành vấn đề
(chưa quan sát thấy vấn đề gì ở quy mô hiện tại).
**Không bao giờ** push image có chứa `.env`/secret thật — `.dockerignore`
đã loại `.env` khỏi build context, nhưng luôn `docker history <image>` kiểm
tra lại nếu nghi ngờ trước khi push.

## 6. Những chỗ cần double-check với tài liệu provider thật

`gemini-3.1-flash-image` / `gemini-3-pro-image` / `mai-image-2.5*` /
`flux.2-pro` là model rất mới — request/response JSON shape chính xác chưa
xác nhận được với tài liệu chính thức tại thời điểm viết code.
`app/providers/gemini_image.py` và `app/providers/azure_generic_image.py`
implement theo format phổ biến nhất cho lớp API đó (`generateContent`
multimodal cho Gemini; `prompt`/`size`/`n`/`b64_json` kiểu OpenAI-images cho
nhóm Azure). `gpt-image-2` theo đúng tài liệu Azure OpenAI Images API đã công
bố. Nếu response thực tế khác, chỗ cần sửa CHỈ nằm trong file `generate()`
tương ứng của provider đó — phần còn lại (pipeline, prompt building, output
management, hợp đồng FastAPI/webapp) độc lập với chi tiết này. Mọi lỗi đều
được log đầy đủ và trả về dạng `{"ok": false, "error": ...}` theo từng
model thay vì crash hay treo — sai field name sẽ hiện rõ ngay trong log/UI.
