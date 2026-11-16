# Backend App Architecture — DCO Dynamic Creative Opt (`version_2h`)

Tài liệu này mô tả luồng xử lý **thật** của backend (`app/`) trong `version_2h`,
từ lúc đọc `request.json` cho tới lúc ảnh cuối cùng được lưu ra đĩa: đi qua
node/hàm nào, mỗi bước xử lý gì, dùng kỹ thuật/thư viện gì, có gọi Image
GenAI model hay LLM hay không, và có ghi log/file ra đĩa hay không. Nội dung
bám sát code thật đang có trong repo tại thời điểm viết tài liệu (không phải
bản thiết kế lý thuyết) — mọi đường dẫn file, tên hàm, tên node đều là tên
thật trong `app/`.

`version_2h` đã đi khá xa so với bản `version_1` gốc: có thêm bước tự động
lên kế hoạch canvas theo tỷ lệ native của model (`canvas_planner.py`), có
thêm 1 vòng lặp validate + sửa lỗi có mục tiêu SAU khi ảnh đã sinh ra
(`output_validator.py`), có thêm đường sinh text bằng LLM và ghép text bằng
code thay vì để AI vẽ chữ (`text_gen.py`, `content_writer.py`,
`text_compositor.py`), có bộ "kỹ thuật thiết kế" (design techniques) và suy
màu tint/shade nạp từ config (`design_techniques.py`, `color_utils.py`), và
đã **bỏ hẳn** ảnh SAFE AREA GUIDE đính kèm gửi cho model (3 module cũ
`safe_area_guide.py`, `guideline_rewriter.py`, `prompt_bleed_guard.py` giờ
là dead code — xem mục 9).

---

## 1. Tổng quan các lớp

```
Streamlit webapp (webapp/)          -- giao diện demo, KHÔNG chứa business logic
        |  HTTP (JSON) qua webapp/utils/api_client.py
        v
FastAPI backend (app/api/routes.py) -- các route mỏng, chỉ gọi xuống app/core/*
        |
        v
app/core/pipeline.py                -- ĐIỀU PHỐI: LangGraph (bước 1-5, mục 3)
                                        + vòng lặp model x version, SINH ẢNH +
                                        VALIDATE + SỬA LỖI CÓ MỤC TIÊU + GHÉP
                                        TEXT BẰNG CODE (bước 6, ngoài graph,
                                        mục 4-7)
        |          |            |             |
        v          v            v             v
app/core/*.py   app/providers/*.py   app/llm/*.py        app/core/text_gen.py
(guideline,     (Image GenAI: Gemini,  (LLM cho prompt      app/core/content_writer.py
 prompt_builder,  gpt-image-2,          refine + vision      (copy LLM, opt-in)
 design_techniques, MAI-Image, FLUX)    validate + copy
 color_utils,                            LLM -- Azure
 canvas_planner,                         GPT-5-mini)
 image_*, output_*,
 text_compositor,
 text_style_profiles)
```

`webapp/` (Streamlit) không bao giờ tự đọc file trên đĩa của backend hay tự
build prompt — nó chỉ gọi các endpoint REST của `app/api/routes.py` qua
`webapp/utils/api_client.py` (`webapp/views/generate.py`,
`webapp/views/release.py`, `webapp/views/home.py` là 3 view gọi client này).
Toàn bộ "trí tuệ" của app nằm trong `app/core/pipeline.py`: **LangGraph**
điều phối 5 bước đầu (đọc request → resolve assets → tra guideline →
downsample ảnh → build prompt), còn phần sinh ảnh thật + hậu xử lý (crop,
validate, sửa lỗi có mục tiêu, ghép text bằng code, lưu file) nằm **ngoài
graph**, chạy dạng vòng lặp Python thường cho từng `(model, version)`.

---

## 2. Sơ đồ luồng xử lý 1 lần chạy `POST /generate`

```
POST /generate {request_name, mode, custom_models}
        |
        v
app/api/routes.py::generate()
        |  model_usage.resolve_models(mode, custom_models)  -- KHÔNG gọi LLM/model,
        |                                                       chỉ đọc config/model_usage.json
        v
app/core/pipeline.py::run_generate(request_name, models, num_version_override)
        |
        |============ PHẦN 1: LangGraph -- chạy 1 LẦN DUY NHẤT cho cả run ============
        v
   ┌──────────────────┐
   │ load_request      │  đọc input/<request_name>.json, validate Pydantic (CreativeRequest),
   │                    │  rồi TỰ ĐỘNG lên lại kế hoạch generation.canvas/safe_area/output.crop
   │                    │  theo tỷ lệ native của (các) model đã chọn (canvas_planner.py, mục 3a)
   └────────┬───────────┘
            v
   ┌──────────────────┐
   │ resolve_assets    │  buckets.theme/context/product + reference -> Path thật dưới lib/,
   │                    │  hoặc tách thành free-text description nếu không phải path ảnh
   └────────┬───────────┘
            v
   ┌──────────────────┐
   │ load_guideline    │  tra lib/guidelines/guidelines.json theo format_type (+ general_guide đệ quy)
   └────────┬───────────┘
            v
   ┌──────────────────┐
   │ prepare_images    │  downsample toàn bộ ảnh đính kèm (Pillow) -- KHÔNG còn vẽ/đính kèm ảnh
   │                    │  SAFE AREA GUIDE nữa (xem mục 9)
   └────────┬───────────┘
            v
   ┌──────────────────┐
   │ build_prompt      │  ghép prompt text (guideline + design rules + design techniques + màu
   │                    │  tint/shade + …), cache theo hash, TÙY CHỌN gọi LLM (GPT-5-mini) refine
   └────────┬───────────┘
            v
        (END graph -- trả về `built_prompt` DÙNG CHUNG cho MỌI model/version bên dưới,
         trừ khi có sinh text theo từng version -- xem mục 4b/5)
        |
        |============ PHẦN 2: vòng lặp model x version -- NGOÀI graph (mục 4) ============
        v
   [1 lần/run] clear_text_gen_cache(request_name)
   [1 lần/run] nếu generate_text=true: text_gen.generate_creative_texts() sinh headline/
               message/cta cho TỪNG version, build lại built_prompt riêng cho từng version
   [1 lần/run] ngược lại, nếu text_by_code=true HOẶC auto_generate_text=true: content_writer.
               draft_missing_text() điền các field text còn trống, rebuild built_prompt 1 lần

   for model in models:
       for version in 1..num_version:
           write_prompt_log(...)                         -- ghi prompt/prompt-<model>-version-<n>.txt
           for attempt in 1..2 (_MAX_GENERATE_ATTEMPTS):
               raw_bytes = generate_image(model, ...)     -- GỌI IMAGE GENAI MODEL THẬT
               cropped_bytes = crop_canvas_to_output(...) -- Pillow crop generation.canvas -> output.size
               problems = validate_output(cropped_bytes)  -- vision-LLM check trên ẢNH CUỐI
               nếu problems rỗng -> break; nếu còn attempt -> build targeted-repair
               prompt (đính kèm chính ẢNH VỪA SINH LỖI) rồi lặp lại
           final_bytes = cropped_bytes (giữ attempt cuối dù còn lỗi)
           nếu text_by_code=true: final_bytes = render_text_layer(...)  -- PIL vẽ đè headline/
               message/CTA lên ẢNH CUỐI, KHÔNG để model tự vẽ chữ
           lưu <display_size>/<model>/<file>.png              (ảnh cuối, đã crop + có thể có text)
           lưu <display_size>/<model>/ORG/<file>.png          (ảnh gốc model trả về ở LẦN GỌI CUỐI,
                                                                 CHƯA crop, CHƯA vẽ text)
```

Vì sao bước [6] KHÔNG phải là 1 node LangGraph: `built_prompt` chỉ cần build
**1 lần** rồi tái sử dụng cho mọi (model, version) (trừ khi có sinh text
theo version — khi đó build lại 1 lần/version, vẫn không phụ thuộc model).
Nếu để "gọi model + crop + validate + lưu" thành node trong graph, graph sẽ
phải chạy lại từ đầu cho từng model/version, tốn kém và không cô lập được
lỗi. Để nó ở vòng lặp Python thường, mỗi cặp (model, version) được bọc
`try/except` RIÊNG — 1 model lỗi (hết quota, sai key, network timeout...)
không làm hỏng kết quả của các model khác trong cùng lần chạy (đúng nguyên
tắc REQUEST.md: "dump exception, không bao giờ crash cả run").

---

## 3. Chi tiết từng node LangGraph

Graph được dựng trong `app/core/pipeline.py::build_graph()` bằng
`langgraph.graph.StateGraph`, state kiểu `PipelineState` (một `TypedDict`).
Mỗi node là 1 hàm `_node_xxx(state) -> dict`, trả về các key sẽ được merge
vào state chung. Graph chỉ có 1 nhánh tuyến tính (không rẽ nhánh/không vòng
lặp bên trong graph).

### Node 1 — `load_request`

| | |
|---|---|
| Hàm | `_node_load_request()` → `load_request_json()` + `app/core/canvas_planner.py::plan_request_geometry()` |
| Input | `state["request_name"]`, `state["models"]` |
| Output | `state["req"]` (kiểu `CreativeRequest`), và `state["canvas_plan"]` nếu có lên kế hoạch lại |
| Kỹ thuật | Đọc file JSON (`json.loads`) từ `input/<request_name>.json`, validate + parse bằng **Pydantic v2** (`CreativeRequest.model_validate`). Sau đó gọi `plan_request_geometry(req, models)`: nếu MỌI model trong `models` cùng có 1 bảng tỷ lệ khung hình đã biết (hiện chỉ `gemini-3.1-flash-image`, 14 tỷ lệ trong `GEMINI_31_FLASH_IMAGE_RATIOS`), tự động chọn tỷ lệ native gần nhất với `output.size`, dựng canvas nhỏ nhất chứa trọn `output.size` theo tỷ lệ đó, và ghi đè `generation.canvas`/`generation.safe_area`/`output.crop` trong bộ nhớ (KHÔNG sửa file JSON gốc trên đĩa). Nếu model không nằm trong bảng đã biết hoặc các model trộn lẫn nhiều bảng khác nhau, giữ nguyên hình học đã khai báo trong request.json. |
| Gọi Image Model? | Không |
| Gọi LLM? | Không |
| Ghi log/file? | Không ghi file mới; log INFO `"[1/5] Reading request.json for ..."` + 1 dòng log INFO cho biết đã auto-plan hay giữ nguyên hình học |

### Node 2 — `resolve_assets`

| | |
|---|---|
| Hàm | `_node_resolve_assets()` → `app/core/asset_resolver.py::resolve_assets()` |
| Input | `state["req"]` |
| Output | `state["resolved_assets"]` (kiểu `ResolvedBucketAssets`: list Path cho theme/context/product đã resolve + list free-text description riêng cho từng bucket, Path reference nếu có, Path guideline_extra nếu có) |
| Kỹ thuật | Với mỗi giá trị trong `buckets.theme/context/product`: nếu đuôi file là ảnh (`.png/.jpg/.jpeg/.webp/.gif/.bmp/.tif/.tiff`) thì ghép với `ASSET_LIBRARY_DIR` (`lib/`) và kiểm tra tồn tại thật trên đĩa; nếu KHÔNG phải đuôi ảnh, coi đó là **mô tả bằng chữ** (free-text description) — không resolve thành file, không coi là lỗi "file not found", thay vào đó được đưa thẳng vào `prompt_builder.py` như 1 chỉ dẫn sáng tạo (xem mục 2026-09-04 note trong `asset_resolver.py`). File ảnh không tồn tại thì log warning và bỏ qua (không crash). |
| Gọi Image Model? | Không |
| Gọi LLM? | Không |
| Ghi log/file? | Không ghi file mới; chỉ log warning cho asset ảnh không tìm thấy |

### Node 3 — `load_guideline`

| | |
|---|---|
| Hàm | `_node_load_guideline()` → `app/core/guidelines.py::resolve_guideline()` |
| Input | `state["req"].format_type` |
| Output | `state["guideline"]` (kiểu `ResolvedGuideline`: text đã gộp + list Path ảnh guideline) |
| Kỹ thuật | Đọc `lib/guidelines/guidelines.json`, tra theo `format_type`, đệ quy gộp thêm mọi entry trong `general_guide` (BFS, có `seen` set chống lặp vô hạn) |
| Gọi Image Model? | Không |
| Gọi LLM? | Không |
| Ghi log/file? | Không ghi file mới; log INFO liệt kê các entry đã resolve |

### Node 4 — `prepare_images`

| | |
|---|---|
| Hàm | `_node_prepare_images()` → `app/core/image_prep.py::prepare_attachments()` |
| Input | `state["req"]`, `state["resolved_assets"]`, `state["guideline"]` |
| Output | `state["prepared_images"]` (list `PreparedImage`: label + source_path + resized_path + mime_type) |
| Kỹ thuật | **Pillow**: resize mọi ảnh đính kèm (theme/context/product/reference) xuống tối đa `attach_image_max_dim` px (mặc định 1024px), giữ alpha thì xuất PNG, không thì xuất JPEG (`attach_image_jpeg_quality`, mặc định 85). Có cache theo `tên_file__mtime`. **KHÔNG còn** gửi ảnh guideline (screenshot guideline) hay ảnh SAFE AREA GUIDE cho model nữa — cả hai đều bị bỏ có chủ đích (xem chú thích trong `_node_prepare_images()` và mục 9 bên dưới): screenshot guideline từng bị model chép lại border/ruler vào ảnh, còn SAFE AREA GUIDE từng bị model vẽ lại khung viền/caption của chính nó vào ảnh sinh ra. |
| Gọi Image Model? | Không |
| Gọi LLM? | Không |
| Ghi log/file? | **Có** — ghi ảnh đã downsample vào `output/<request_name>/_resized_refs/` |

### Node 5 — `build_prompt`

| | |
|---|---|
| Hàm | `_node_build_prompt()` → `app/core/prompt_builder.py::build_prompt()` |
| Input | `state["req"]`, `state["guideline"]`, `state["prepared_images"]` (chỉ dùng `label` theo đúng thứ tự đính kèm) |
| Output | `state["built_prompt"]` (kiểu `BuiltPrompt`: text cuối cùng + cache_key + thứ tự nhãn ảnh + cờ `llm_refined`) |
| Kỹ thuật | Ghép prompt từ ~20 "section" text theo thứ tự cố định (xem danh sách đầy đủ ở mục 3a bên dưới). Cache theo **SHA-256 hash** của mọi input có thể ảnh hưởng nội dung prompt (design_brief, buckets, text, guideline, design_options, generation/output geometry, thứ tự ảnh đính kèm, nội dung `design_techniques.json` đã resolve, và `_PROMPT_TEMPLATE_VERSION` — hằng số này phải tăng lên mỗi khi sửa code assembly, để 1 thay đổi code thuần túy cũng làm mất hiệu lực cache cũ dù `request.json` không đổi). |
| Gọi Image Model? | Không |
| **Gọi LLM?** | **CÓ, TÙY CHỌN** — chỉ khi request KHÔNG có crop (`has_crop()=False`) và `Settings.use_llm_prompt_refinement=True` (mặc định BẬT): gọi `LLMFactory.create(settings).generate(raw_prompt, system=...)` để viết lại prompt cho cô đọng, giữ nguyên mọi ràng buộc. Với request CÓ crop, bước refine này bị **bỏ qua có chủ đích** (log INFO) vì từng làm prompt bị paraphrase lại và tái xuất hiện từ ngữ kiểu "khung an toàn" khiến model vẽ hộp inset giả. Provider mặc định `llm_provider="openai"` thực chất trỏ tới Azure AI Foundry GPT-5-mini (`az_gpt5mini_endpoint`/`az_gpt5mini_key`). Lỗi gọi LLM được bắt và fallback về prompt chưa refine. |
| Ghi log/file? | **Có** — ghi `output/<request_name>/_prompt_cache/built_prompt.json` (cache_key + text đã refine + text thô + cờ llm_refined) |

### 3a. Thứ tự các section trong prompt (`build_prompt()`)

`app/core/prompt_builder.py::build_prompt()` ghép prompt theo đúng thứ tự
sau (mỗi mục là 1 "section" nối bằng `\n\n`):

1. **TASK** — câu lệnh gốc: sinh 1 banner marketing hoàn chỉnh cho OCB.
2. **DESIGN BRIEF** — nguyên văn `req.design_brief`.
3. **REQUEST CONTROL FLAGS** — nhắc lại `generate_person`/`has_logo` và số
   lượng ảnh/mô tả chữ đã populate cho từng bucket, nhấn mạnh đây là hợp
   đồng cứng, không được tự suy diễn thêm.
4. **CANVAS AND FINAL TRIM** (nếu có crop) hoặc **CANVAS** (nếu không) —
   mô tả kích thước canvas thật + vùng sẽ bị crop đi sau khi sinh.
5. **FULL-BLEED BACKGROUND** — cấm tuyệt đối việc vẽ khung/viền/vignette
   bao quanh toàn bộ canvas (thêm sau sự cố 1 banner bị model tự vẽ viền
   vàng kim loại quanh toàn khung, 2026-09-03).
6. **Bucket section** (`_bucket_section()`) — liệt kê thứ tự ảnh đính kèm
   thật sự gửi cho model kèm mô tả vai trò từng ảnh.
7. **TEXT** — nếu `text_by_code=true`: yêu cầu KHÔNG vẽ chữ gì cả + để
   trống 1 cột bên phải canvas (theo `text_style_profiles.py`) cho code
   ghép chữ sau; ngược lại: **REQUIRED TEXT CONTENT** với headline/
   message/CTA nguyên văn + quy tắc không thêm ký tự trang trí/metadata.
8. **LOGO** (`_logo_instruction()`).
9. **Theme/background** (`_theme_instruction()`) — có thể là ảnh theme
   khóa cứng, mô tả chữ, hoặc gradient theo `dominant_color` (xem mục 8).
10. **Person/context** (`_person_instruction()`) — 4 nhánh: ảnh + mô tả
    chữ đi kèm (edit instruction), chỉ ảnh (khóa cứng danh tính), chỉ mô tả
    chữ (tự sinh theo mô tả), hoặc không có gì (`generate_person` quyết
    định có tự bịa người hay không).
11. **Product** (`_product_instruction()`) — tương tự: ảnh khóa cứng (được
    phép nghiêng góc camera nhưng không được bịa hình học mới), mô tả chữ,
    hoặc không có sản phẩm nào.
12. **MULTI-ELEMENT COMPOSITION** — quy tắc bố cục khi design brief liệt kê
    nhiều ý tưởng hình ảnh (chọn 1 ý chính + 2-4 ý phụ, gộp thành 1 cảnh
    liền mạch thay vì collage rời rạc).
13. **REQUEST-SPECIFIC ART DIRECTION** (nếu `generation.instruction` có
    nội dung).
14. **STYLE AND ASSET COHERENCE OVERRIDE**.
15. **FORMAT DESIGN GUIDELINE** — text guideline đã qua
    `_prepare_guideline_for_generation()` (lọc bỏ các đoạn nói về hình học/
    kích thước tuyệt đối, giữ lại phần mô tả phong cách/nội dung).
16. **DESIGN RULES** — nguyên văn `prompts/design_rules.md`.
17. **DECORATIVE DETAIL LIMIT** (nếu `num_decorative_elements` được set).
18. **DESIGN TECHNIQUES** (nếu `design_options.techniques` không rỗng) —
    xem mục 8 và tài liệu `6-design_techniques.md`.
19. **OVERRIDE ON THE DESIGN RULES ABOVE** (chỉ khi `text_by_code=true`) —
    gỡ bỏ ràng buộc "giữ nguyên text" của DESIGN RULES cho riêng request
    này.
20. **BACKGROUND COVERAGE**.
21. **CREATIVE VARIATION** (nếu `num_version > 1`).
22. **FINAL LOCKED-ASSET REMINDER** (nếu có context/product asset).
23. **FINAL COMPOSITION CHECK** (nếu có crop) — nhắc lại ràng buộc an toàn
    lần cuối ở cuối prompt (lặp lại chủ đích ở cả đầu và cuối vì đây là lỗi
    hay gặp nhất).

---

## 4. Bước [6] — vòng lặp model × version (ngoài graph)

Nằm trong `app/core/pipeline.py::run_generate()`, SAU khi graph đã chạy
xong.

### 4a. Chuẩn bị trước vòng lặp

1. `output_manager.clear_text_gen_cache(request_name)` — xoá cache text đã
   sinh ở lần chạy trước (mỗi lần `/generate` được coi là 1 lần sinh text
   mới nếu `generate_text=true`).
2. `_load_attachment_bytes()` — đọc bytes thật của mọi `PreparedImage` để
   gửi kèm request tới model.
3. **Sinh text (3 chế độ loại trừ lẫn nhau, xem mục 5):**
   - `generate_text=true` → gọi `text_gen.generate_creative_texts()` 1 lần
     cho cả `num_version`, build lại `built_prompt` RIÊNG cho từng version
     (vì mỗi version có thể có headline/message/cta khác nhau).
   - ngược lại, nếu `text_by_code=true` HOẶC `auto_generate_text=true` →
     gọi `content_writer.draft_missing_text()` 1 lần để điền các field
     text còn trống, rồi build lại `built_prompt` DÙNG CHUNG cho mọi
     version (vì chỉ điền field trống 1 lần, không sinh riêng theo từng
     version).
   - mặc định (`generate_text=false` và không có 2 cờ trên) → dùng nguyên
     `built_prompt` do graph đã build, `text.*` tĩnh từ JSON.

### 4b. Vòng lặp chính

```python
for model in models:
    for version in range(1, num_version + 1):
        write_prompt_log(...)                              # (a)
        for attempt in range(1, _MAX_GENERATE_ATTEMPTS+1):  # tối đa 2 lần (b)-(d)
            raw_bytes = generate_image(model, ...)          # (b)  <-- GỌI IMAGE GENAI MODEL THẬT
            cropped_bytes = crop_canvas_to_output(...)      # (c)
            problems = validate_output(cropped_bytes, req)  # (d)  <-- vision-LLM check trên ảnh CUỐI
            if not problems: break
            if attempt < max: xây prompt sửa lỗi có mục tiêu, ghi log repair, lặp lại
        final_bytes = cropped_bytes (giữ lần thử cuối dù vẫn còn problems)
        if text_by_code: final_bytes = render_text_layer(final_bytes, ...)  # (e)
        lưu file .../<model>/<file>.png                     # (f)
        lưu file .../<model>/ORG/<file>.png                 # (g) -- ảnh gốc của LẦN GỌI CUỐI
```

**(a) `write_prompt_log()`** (`app/core/prompt_builder.py`) — ghi
`output/<request_name>/prompt/prompt-<model>-version-<n>.txt`: nội dung là
prompt text thật sẽ gửi cho model ở LẦN GỌI ĐẦU TIÊN, cộng thêm 1 footer
liệt kê **tên file thật** của mọi ảnh đã đính kèm
(`format_attached_images_note()`). Ghi **TRƯỚC** khi gọi model, nên dù bước
sinh ảnh lỗi thì log vẫn còn nguyên. Nếu có sửa lỗi có mục tiêu ở attempt
sau, prompt sửa lỗi được ghi vào 1 file RIÊNG (xem mục 6).

**(b) `generate_image()`** (`app/providers/registry.py`) — điểm **GỌI
IMAGE GENAI MODEL THẬT** trong pipeline (có thể được gọi tới 2 lần/​
(model, version) nếu attempt đầu có vấn đề). Route theo tên model tới đúng
client HTTP: `app/providers/gemini_image.py` (`gemini-3.1-flash-image`,
`gemini-3-pro-image`), `app/providers/azure_openai_image.py`
(`gpt-image-2`), `app/providers/azure_generic_image.py` (`mai-image-2.5`,
`mai-image-2.5-flash`, `mai-image-2.5-pro`, `flux.2-pro`). Mỗi client tự
lấy endpoint/key từ `Settings`. Trả về `bytes` ảnh thô, đúng những gì model
trả về — CHƯA qua bất kỳ xử lý nào. **Model đang thực sự dùng** theo
`config/model_usage.json` hiện tại: **chỉ `gemini-3.1-flash-image`** cho
mọi mode (`default`/`quality`/`eco`) — các model khác (gpt-image-2,
MAI-Image, FLUX) vẫn có trong registry (`available_providers()`) cho
`mode=custom` nhưng đã bị khoá khỏi mọi group theo quyết định người dùng
(vi phạm quy tắc ảnh người thật / chất lượng chưa đạt — xem comment `_note_...`
trong `model_usage.json`).

**(c) `crop_canvas_to_output()`** (`app/core/image_post.py`) — Pillow:
resize về đúng `generation.canvas` nếu model trả sai size, crop đúng
`crop_box` (từ `output.crop.area` hoặc `generation.safe_area`), rồi resize
lần cuối để khớp chính xác `output.size`.

**(d) `validate_output()`** (`app/core/output_validator.py`) — xem mục 6.

**(e) `render_text_layer()`** (`app/core/text_compositor.py`) — chỉ chạy
khi `text_by_code=true`. Xem mục 5b.

**(f)/(g)** ghi 2 file PNG ra đĩa (xem sơ đồ thư mục ở mục 10). File `ORG/`
là ảnh model trả về ở **lần gọi cuối cùng** (attempt cuối, trước crop và
trước khi vẽ text bằng code) — nếu có sửa lỗi, đây là ảnh sau khi sửa, không
phải ảnh gốc ban đầu.

---

## 5. Sinh & ghép nội dung chữ (Text generation & compositing)

`version_2h` có tới 3 con đường độc lập để có nội dung chữ cuối cùng trên
banner — tất cả đều là opt-in qua `design_options`, mặc định (`false` hết)
giữ hành vi y hệt `version_1` (text tĩnh từ JSON, do image model tự vẽ).

### 5a. Sinh copy bằng LLM theo từng version — `app/core/text_gen.py`

- Gate duy nhất: `design_options.generate_text=true`.
- Khi bật, `generate_creative_texts()` LUÔN yêu cầu LLM (model theo
  `Settings.llm_text_gen`, mặc định `gpt-5-mini`) trả về **1 bộ đầy đủ**
  headline/message/cta cho **từng version** (1..`num_version`) trong 1 lần
  gọi — không có kiểu "chỉ sinh 1 field". Prompt template lấy từ
  `prompts/<format_type>_text_gen_prompt_template.txt`, fallback
  `prompts/default_text_gen_prompt_template.txt`.
- Field nào THỰC SỰ được dùng trong thiết kế cuối lại là quyết định
  RIÊNG, theo từng field, độc lập với `generate_text`: 3 cờ
  `generate_text_headline` / `generate_text_message` /
  `generate_text_cta_label` — field nào cờ `false` thì giữ nguyên giá trị
  tĩnh `text.<field>` trong JSON dù LLM đã sinh ra nội dung khác.
- Có validate chặt: mỗi field sinh ra không được rỗng, không được chứa hex
  color token (chặn LLM lỡ in ra metadata không in được), tự động cắt các
  ký tự trang trí ở đầu/cuối (`-`, `–`, `•`, `|`, ...).
- Cache theo hash (design_brief, format_type, num_version, template, model)
  tại `output/<request_name>/_text_gen_cache/text_gen.json`, bị xoá đầu mỗi
  `run_generate()` mới (mục 4a) nên luôn sinh mới cho mỗi lần bấm Generate,
  nhưng vẫn dùng chung 1 kết quả cho mọi model trong CÙNG 1 lần chạy.
- Lỗi bất kỳ (LLM fail, JSON không hợp lệ, thiếu version...) → trả `None`,
  pipeline fallback về text tĩnh trong JSON, không bao giờ chặn cả run.

### 5b. Điền nốt field còn trống — `app/core/content_writer.py`

- Gate: `design_options.text_by_code=true` HOẶC `auto_generate_text=true`
  (và chỉ chạy khi `generate_text=false` — xem thứ tự ưu tiên ở mục 4a).
- `draft_missing_text()` chỉ điền những field `headline`/`message`/
  `cta_label` đang RỖNG trong JSON (field đã có giá trị thì giữ nguyên),
  dựa theo `design_brief` + giới hạn số dòng/số từ lấy từ
  `text_style_profiles.profile_for(output_wh)` nếu có. Dùng chung
  `LLMFactory`, best-effort — lỗi thì trả nguyên `req.text` (vẫn rỗng ở
  field lỗi), không chặn run.
- Kết quả được gộp lại vào `req.text` rồi rebuild `built_prompt` 1 LẦN
  (dùng chung cho mọi model/version — không sinh riêng theo version như
  5a).

### 5c. Ghép chữ bằng code lên ảnh cuối — `app/core/text_compositor.py`

- Gate: `design_options.text_by_code=true`.
- Chạy SAU `crop_canvas_to_output()`, TRƯỚC khi lưu file — `render_text_layer()`
  dùng Pillow vẽ trực tiếp headline/message/CTA(+nút bấm) lên ảnh cuối,
  theo đúng toạ độ/font/màu khai báo tĩnh trong
  `app/core/text_style_profiles.py` (hiện có sẵn 2 profile:
  `(1024, 440)` và `(1024, 256)` — mọi `output.size` khác không khớp thì bị
  bỏ qua bước này, log warning).
- Font chữ: Manrope (SemiBold/Medium), tìm trong
  `<asset_library_dir>/fonts/Manrope-*.ttf`; nếu chưa có file thật thì
  fallback về font Windows có sẵn (`seguisb.ttf`/`segoeui.ttf`) kèm cảnh
  báo — cần thả file Manrope thật vào `lib/fonts/` để đúng thiết kế.
- Tự động co cỡ chữ (`fit_font_and_wrap()`) để vừa số dòng tối đa quy định,
  không cắt cụt câu một cách âm thầm; tự chọn màu chữ sáng/tối theo độ sáng
  trung bình thật của vùng nền phía sau (đo trực tiếp trên ảnh đã sinh, vì
  nền có thể đổi theo `dominant_color` từng request).
- Vì đây là bước bắt buộc với các use case `text_by_code=true`, lỗi bất kỳ
  ở đây vẫn được bọc `try/except` và trả lại ảnh KHÔNG có chữ thay vì làm
  hỏng cả kết quả `(model, version)`.
- Vì text được vẽ SAU vòng lặp validate (mục 6), `validate_output()` không
  bao giờ thấy được lớp text do code vẽ — các câu hỏi vision-LLM của nó chỉ
  soi ảnh do MODEL tự vẽ (bao gồm câu hỏi riêng "model có lỡ tự vẽ chữ dù
  được yêu cầu không vẽ hay không" — mục 6).

---

## 6. Validate ảnh cuối + sửa lỗi có mục tiêu (targeted repair)

`app/core/output_validator.py::validate_output()` chạy ngay sau
`crop_canvas_to_output()` trong mỗi attempt — đóng vòng lặp mà các prompt-side
guard (đã bị gỡ, xem mục 9) chỉ có thể YÊU CẦU chứ không thể XÁC NHẬN đã xảy
ra thật hay chưa.

**2 nhóm kiểm tra:**

1. **`has_guide_color_leak()`** — quét toàn bộ pixel ảnh CUỐI (độ phân giải
   gốc, không downscale) tìm màu magenta đặc trưng `(255, 0, 220)` từng
   dùng cho SAFE AREA GUIDE cũ. Vẫn giữ lại dù ảnh guide không còn được
   đính kèm nữa, để bắt trường hợp còn sót đâu đó.
2. **`check_required_elements()`** — 1 loạt câu hỏi yes/no gửi cho
   **vision-LLM** (qua `LLMFactory.create(settings)`, cần provider có
   `generate_vision()` — provider không hỗ trợ ảnh thì bỏ qua toàn bộ nhóm
   này, không coi là lỗi):
   - Sản phẩm (`buckets.product`) có còn khớp với ảnh gốc không (so sánh
     trực tiếp với ảnh reference đã resolve, không hỏi chung chung).
   - Nhân vật/context (`buckets.context`) có còn khớp với ảnh gốc không.
   - Nếu KHÔNG có context asset/text và `generate_person=false`: có phát
     sinh người trái phép không.
   - Nếu `has_logo=true`: logo có xuất hiện không.
   - Nội dung/text bắt buộc và asset quảng cáo có bị cắt sát mép/tràn biên
     không (`_content_containment_question()`).
   - Có xuất hiện artifact kỹ thuật không mong muốn không: hình chữ nhật
     safe-area, ruler, mũi tên đo, nhãn "48px", khung/viền/vignette bao
     quanh nội dung, hex color token lộ ra ngoài dạng text...
   - Nếu `text_by_code=true`: model có lỡ tự vẽ chữ không (`_NO_UNWANTED_TEXT_QUESTION`),
     và sản phẩm/nhân vật/motif có tràn vào cột chữ dành riêng bên phải
     không (`_reserved_zone_question()`, theo `text_style_profiles`).
3. Mỗi câu hỏi lỗi gọi LLM (`None`) được coi là **bỏ qua**, không bao giờ
   biến 1 sự cố hạ tầng thành 1 lỗi validate giả.

**Sửa lỗi có mục tiêu** (`app/core/pipeline.py::_build_targeted_repair()`):
khi `validate_output()` trả về `problems` không rỗng và còn attempt (tối đa
`_MAX_GENERATE_ATTEMPTS = 2`), pipeline build 1 prompt MỚI, đính kèm CHÍNH
ẢNH VỪA SINH LỖI làm attachment #1 ("PREVIOUS FULL-CANVAS DESIGN TO REPAIR"),
yêu cầu model **giữ nguyên** ý tưởng/bố cục/màu sắc cũ, chỉ sửa đúng phần bị
liệt kê lỗi — không tạo lại từ đầu. Prompt sửa lỗi được ghi log riêng ở
`prompt-<model>-version-<n>-repair-<attempt>.txt`. Nếu sau 2 lần thử vẫn còn
lỗi, pipeline **vẫn giữ lại** kết quả lần cuối (log warning), không bao giờ
làm fail cả `(model, version)` chỉ vì validate không sạch 100%.

---

## 7. Kế hoạch hình học canvas theo model — `app/core/canvas_planner.py`

Thay vì tin tưởng tuyệt đối `generation.canvas` viết tay trong request.json,
`plan_request_geometry()` (gọi từ Node 1, mục 3) coi `output.size` mới là
nguồn sự thật, rồi tự chọn 1 tỷ lệ khung hình **native** gần nhất mà model
đã chọn hỗ trợ (`closest_supported_ratio()`, so khoảng cách log-tỷ lệ đối
xứng theo chiều), dựng canvas nhỏ nhất theo tỷ lệ đó chứa trọn `output.size`
ở giữa, rồi crop phần dư nhỏ sau khi sinh — tránh việc ép model vẽ trong 1
dải tỷ lệ quá lệch với tỷ lệ nó vốn được huấn luyện, vốn làm model vứt bỏ
phần lớn pixel đã sinh ra một cách lãng phí. Chỉ áp dụng khi **TẤT CẢ**
model trong `models` cùng chia sẻ 1 bảng tỷ lệ đã biết (`ratio_buckets_for_models()`)
— vì graph chỉ build 1 prompt dùng chung cho mọi model; nếu models không
đồng nhất hoặc không nằm trong bảng đã biết, giữ nguyên hình học request đã
khai báo. Module này cũng cấp `plan_canvas_for_model()` dùng riêng cho
`run_clone()` (mục 11, mỗi size clone tính hình học độc lập theo model được
chọn).

---

## 8. Design techniques & suy màu tint/shade — nạp từ config

Hai cơ chế bổ trợ cho `build_prompt()`, cả hai đều **tùy chọn** (opt-in qua
`design_options`, mặc định không đổi hành vi hiện có):

- **`app/core/design_techniques.py`** — đọc `config/design_techniques.json`,
  resolve `design_options.techniques` (list key) thành các
  `ResolvedTechnique{label, description, prompt_instruction}`, được
  `prompt_builder.py::_design_techniques_text()` gộp thành 1 section
  "DESIGN TECHNIQUES" chèn ngay sau DESIGN RULES/DECORATIVE LIMIT — luôn
  đóng khung là "chỉ dẫn hỗ trợ", không được ghi đè bất kỳ MUST/MUST NOT
  nào. Key lạ (typo) chỉ bị bỏ qua + log warning, không crash. Nội dung đầy
  đủ của TỪNG kỹ thuật hiện có (nhóm nền/nhóm sản phẩm/nhóm trang trí theo
  dịp lễ) được liệt kê chi tiết trong **`6-design_techniques.md`**.
- **`app/core/color_utils.py::tint_shade_from_dominant()`** — từ 1 hex màu
  `design_options.dominant_color`, suy ra cặp `TINT_HEX`/`SHADE_HEX` bằng
  cách chuyển sang không gian HSL rồi dịch **độ sáng (lightness)** lên/xuống
  1 lượng cố định nhỏ (mặc định ±0.14 trên thang 0..1), giữ nguyên hue +
  gần như giữ nguyên saturation — đảm bảo TINT/SHADE luôn đọc là "cùng 1
  màu, chỉ sáng/tối hơn chút" thay vì lệch hẳn sang màu khác (bài học rút ra
  sau khi bản mix RGB-space đầu tiên tạo ra SHADE gần như đen ở 1 màu nền
  tối, xem docstring module này). 2 giá trị này thế chỗ cho placeholder
  `{{TINT_HEX}}`/`{{SHADE_HEX}}` trong `prompts/design_rules.md`.

---

## 9. Module đã ngừng dùng (dead code trong `version_2h`)

3 module sau **không còn được import** bởi `pipeline.py` hay
`prompt_builder.py` nữa (xác nhận bằng grep — chỉ còn xuất hiện trong
docstring/comment tự tham chiếu và trong khối `if __name__ == "__main__":`
của chính chúng), cùng lý do bị gỡ như bản `version_h1` (xem
`project_version_h1_architecture` note nội bộ):

- **`app/core/safe_area_guide.py`** — từng vẽ + đính kèm 1 ảnh khung magenta
  đánh dấu safe area cho model tham chiếu. Bị gỡ khỏi `_node_prepare_images()`
  vì model liên tục copy chép lại border/caption của chính ảnh guide vào
  ảnh sinh ra (đây là lý do `validate_output()` vẫn giữ
  `has_guide_color_leak()` — phòng trường hợp code cũ hoặc cache cũ còn
  sót). Thay bằng: text padding vô hình + validate/sửa lỗi sau khi sinh
  (mục 6).
- **`app/core/guideline_rewriter.py`** — từng dùng LLM viết lại số đo trong
  guideline text cho khớp canvas thực tế. Bị gỡ vì bản viết lại vẫn để lộ
  số đo khung/safe-zone khiến model hình dung ra 1 hộp nội dung giả.
- **`app/core/prompt_bleed_guard.py`** — từng chạy 1 lượt LLM cuối cùng
  quét prompt để tìm và xoá các cụm từ "cho phép sản phẩm tràn ra ngoài".
  Bị gỡ vì việc paraphrase lại toàn prompt tái tạo ra chính các từ khoá
  "khung an toàn" mà nó đang cố xoá.

`Settings.safe_area_guide_dir` vẫn còn tồn tại trong `app/settings.py`
(không dùng tới nữa) — an toàn để giữ nguyên, không ảnh hưởng runtime.

---

## 10. Toàn bộ file/log ghi ra đĩa trong 1 lần `/generate`

```
output/
├── _logs/app.log                                  log tổng của CẢ APP (mọi request/module dùng
│                                                     chung 1 file này, xem mục 13 -- KHÔNG nằm
│                                                     trong thư mục riêng của request_name nào)
└── <request_name>/
    ├── _prompt_cache/built_prompt.json            cache prompt đã build (1 lần/run, hoặc 1/version
    │                                                 nếu generate_text=true -- mục 5a)
    ├── _text_gen_cache/text_gen.json              bộ headline/message/cta đã sinh cho từng version
    │                                                 (xoá đầu mỗi lần /generate mới -- mục 4a)
    ├── _resized_refs/
    │   └── <asset>__<mtime>.jpg|png               ảnh đính kèm đã downsample
    ├── prompt/
    │   ├── prompt-<model>-version-<n>.txt          prompt thật gửi ở LẦN GỌI ĐẦU + danh sách ảnh đính kèm
    │   └── prompt-<model>-version-<n>-repair-<a>.txt  prompt sửa lỗi có mục tiêu, nếu attempt > 1 (mục 6)
    ├── <display_size>/<image_model>/
    │   ├── <creative_id>_<size>_<model>_<ts>.png  ảnh CUỐI (đã crop, đã ghép text nếu text_by_code) -- kết quả chính
    │   └── ORG/<cùng tên file>.png                ảnh gốc model trả về ở lần gọi CUỐI (CHƯA crop, CHƯA ghép text)
    └── release/
        ├── <file đã release>.png                  bản copy của ảnh được chọn "release" trên webapp
        └── <stem>_clone_<size>_<model>.png         ảnh sinh ra từ chế độ clone
```

Ảnh SAFE AREA GUIDE và thư mục cache của nó (`lib/safe_area_guide/`,
`Settings.safe_area_guide_dir`) không còn được ghi mới nữa (mục 9) — file
cũ nếu còn trên đĩa từ trước chỉ là tàn dư, không được đọc lại.

---

## 11. Chế độ Clone (`run_clone`) — luồng riêng, KHÔNG dùng LangGraph

`app/core/pipeline.py::run_clone(request_name, source_image_path,
image_model, display_sizes)` phục vụ trang Release của webapp: lấy 1 ảnh
thiết kế đã hoàn thiện, gửi lại cho model với 1 prompt cố định ("giữ nguyên
thiết kế, đổi kích thước canvas") cho từng size trong `display_sizes`.
Không đọc lại `request.json` gốc, không cần guideline/asset resolution, nên
không đi qua graph. Với MỖI size, gọi `canvas_planner.plan_canvas_for_model()`
riêng theo `image_model` được chọn (không dùng chung 1 kế hoạch cho nhiều
model như luồng generate chính, vì đây vốn chỉ chạy 1 model/lần) để tính
canvas/crop native, rồi `generate_image()` → `crop_canvas_to_output()`.
Không chạy qua `validate_output()`/targeted-repair hay `render_text_layer()`
— đây là luồng tái tạo 1 thiết kế đã duyệt ở size khác, không phải luồng
sinh sáng tạo ban đầu. Kết quả lưu vào
`output/<request_name>/release/<source_stem>_clone_<size>_<model>.png`.

---

## 12. Logging

`app/logging_conf.py::get_logger(name)` — mọi module gọi
`logger = get_logger("tên_module")`, log ra **2 nơi cùng lúc**: stdout
(terminal chạy `uvicorn`) và file xoay vòng `output/_logs/app.log` (tối đa
5MB/file, giữ 5 file cũ). Mỗi bước trong graph + mỗi lần gọi model/crop/
validate/lưu file đều có ít nhất 1 dòng log INFO đánh số thứ tự, và mọi lỗi
bị bắt đều log bằng `logger.exception(...)` (traceback đầy đủ) trước khi
tiếp tục — đúng nguyên tắc REQUEST.md "dump exception, không bao giờ crash
cả run".

---

## 13. Bảng tổng hợp nhanh

| Bước | Có phải LangGraph node? | Gọi Image Model? | Gọi LLM? | Ghi log/file mới? |
|---|---|---|---|---|
| 1. load_request (+ auto-plan canvas) | Có | Không | Không | Không |
| 2. resolve_assets | Có | Không | Không | Không |
| 3. load_guideline | Có | Không | Không | Không |
| 4. prepare_images | Có | Không | Không | Có (`_resized_refs/`) |
| 5. build_prompt | Có | Không | **Có, tùy chọn (GPT-5-mini, chỉ khi không crop)** | Có (`_prompt_cache/`) |
| 5a. text_gen (generate_text=true) | Không (trước vòng lặp) | Không | **Có (gpt-5-mini, 1 lần/run)** | Có (`_text_gen_cache/`) |
| 5b. content_writer (text_by_code/auto_generate_text) | Không (trước vòng lặp) | Không | **Có, best-effort** | Không |
| 6a. write_prompt_log | Không (vòng lặp ngoài graph) | Không | Không | Có (`prompt/`) |
| 6b. generate_image (mỗi attempt) | Không | **Có** | Không | Không |
| 6c. crop_canvas_to_output | Không | Không | Không | Không |
| 6d. validate_output | Không | Không | **Có (vision-LLM, nhiều câu hỏi yes/no)** | Không |
| 6e. targeted repair (nếu problems) | Không | Có (lần gọi lại) | Không | Có (`prompt/*-repair-*.txt`) |
| 6f. render_text_layer (text_by_code) | Không | Không | Không | Không |
| 6g. lưu ảnh cuối + ORG | Không | Không | Không | Có (`<size>/<model>/` + `ORG/`) |

---

## 14. Endpoint FastAPI ↔ hàm backend tương ứng

| Endpoint | Hàm xử lý | Có gọi Image Model/LLM? |
|---|---|---|
| `GET /health` | trả trạng thái app | Không |
| `GET /models` | `model_usage.load_model_usage()` + `registry.available_providers()` | Không |
| `GET /requests`, `GET /requests/{name}` | đọc `input/*.json`, `resolve_assets()` | Không |
| `GET /outputs/{name}`, `.../latest`, `.../at-timestamp` | `output_manager.list_new_images/latest_run_images/images_matching_substring` | Không |
| `POST /generate` | `pipeline.run_generate()` | **Có** (mục 4-6) |
| `POST /clone`, `POST /clone-from-release` | `pipeline.run_clone()` | **Có** (mục 11) |
| `POST /release` | `output_manager.release_images()` (copy file) | Không |
| `GET /release/{name}`, `GET /release/{name}/clone-status` | đọc thư mục `release/` | Không |
| `GET /files` | `FileResponse`, giới hạn trong `output_dir`/`input_dir`/`asset_library_dir`/`guidelines_file.parent`, kiểm tra path-traversal TRƯỚC khi kiểm tra tồn tại | Không |

---

## 15. Nguyên tắc xử lý lỗi xuyên suốt

Mọi bước có khả năng lỗi (đọc file, gọi HTTP ra ngoài, ghi đĩa, gọi LLM)
đều được bọc `try/except Exception` riêng lẻ, log bằng
`logger.exception(...)` rồi **tiếp tục chạy** thay vì để lỗi lan lên làm sập
cả request — đúng nguyên tắc REQUEST.md: *"dump exception, không bao giờ
crash cả run"*. Cụ thể:

- 1 model lỗi trong `mode=quality`/`all` (nhiều model) không ảnh hưởng các
  model còn lại — mỗi (model, version) trả về 1 kết quả `ok: true/false`
  độc lập.
- Sinh text bằng LLM (`text_gen`/`content_writer`) lỗi → fallback về text
  tĩnh trong JSON, không chặn sinh ảnh.
- Vision-LLM validate lỗi hạ tầng (không phải "tìm thấy vấn đề") → coi như
  bỏ qua check đó (`None`), không tạo lỗi giả.
- Sửa lỗi có mục tiêu thất bại sau đủ số lần thử → vẫn giữ kết quả lần thử
  cuối, không làm fail cả `(model, version)`.
- Ghi text bằng code (`render_text_layer`) lỗi → trả lại ảnh không có chữ,
  không làm mất ảnh đã sinh.
- Việc ghi ảnh ORG/ hoặc ghi prompt log lỗi không làm hỏng kết quả chính.
- `GET /requests/{name}` với file không tồn tại trả `200 {"ok": false}`,
  không bao giờ trả `500`.

---

## 16. Cấu hình (Settings) liên quan trực tiếp tới luồng xử lý

Toàn bộ tham số điều khiển luồng trên đều đọc qua
`app/settings.py::get_settings()` (nguồn config duy nhất).

| Field | Ảnh hưởng tới bước nào |
|---|---|
| `input_dir`, `output_dir`, `asset_library_dir` | Node 1, 2, và mọi đường dẫn ghi file ở mục 10 |
| `attach_image_max_dim`, `attach_image_jpeg_quality` | Node 4 (downsample ảnh đính kèm) |
| `use_llm_prompt_refinement`, `llm_provider`, `llm_model_name`, `llm_text_gen`, `az_gpt5mini_endpoint/key` | Node 5 (refine prompt), mục 5a/5b (sinh text), mục 6 (vision validate) |
| `az_gpt_image_2_endpoint/key`, `gcp_gemini_*`, `az_mai_image_*`, `az_flux_2_pro_*` | Bước 6b (gọi Image GenAI model thật) |
| `design_techniques_file` (`config/design_techniques.json`) | Mục 8 (design techniques) |
| `model_usage_file` (`config/model_usage.json`) | `GET /generate`'s mode -> model list resolution |
| `safe_area_guide_dir` | Không còn dùng (mục 9) -- giữ lại cho tương thích ngược |
| `log_level` | Mục 12 (logging) |

---

*Tài liệu này mô tả code thật của `version_2h` tại thời điểm 2026-09-06,
thay thế bản mô tả cũ vốn viết cho `version_1` (không có canvas_planner,
text_gen/content_writer/text_compositor, output_validator + targeted
repair, design_techniques/color_utils). Xem `4-project-evaluation-results.md`
để biết lịch sử chi tiết từng thay đổi, và `6-design_techniques.md` cho
danh sách đầy đủ các kỹ thuật thiết kế hiện có trong
`config/design_techniques.json`.*
