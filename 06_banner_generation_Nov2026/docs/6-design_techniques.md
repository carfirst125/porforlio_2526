# Design Techniques — Danh mục kỹ thuật thiết kế đang có (`version_2h`)

Tài liệu này liệt kê **toàn bộ** kỹ thuật thiết kế đang được nạp từ
`config/design_techniques.json` và cách chúng thực sự chảy vào prompt gửi
cho Image GenAI model, để dùng làm tài liệu tham khảo khi viết
`design_options.techniques` trong `request.json`. Nội dung khớp với code
thật tại `app/core/design_techniques.py` + `app/core/prompt_builder.py::_design_techniques_text()`
— xem `5-backend_app_architeture.md` mục 8 cho vị trí kỹ thuật này trong
toàn bộ pipeline.

---

## 1. Cách khai báo & cách nó chảy vào prompt

Trong `request.json`:

```json
{
  "design_options": {
    "techniques": ["silhoutte_watermark", "numeric_ribbon_badge", "outlined_headline_typography"]
  }
}
```

- `techniques` là 1 **list các key** trỏ vào `config/design_techniques.json`.
  Rỗng (mặc định) = không thêm chỉ dẫn kỹ thuật nào, hành vi y hệt trước
  khi tính năng này tồn tại.
- Mỗi key được `app/core/design_techniques.py::resolve_techniques()` tra
  thành `{label, description, prompt_instruction}`. Key sai chính tả/không
  tồn tại chỉ bị **bỏ qua + log warning**, không làm crash request (lỗi hay
  gặp: nhầm `silhouette_watermark` với chính tả thật trong config là
  **`silhoutte_watermark`** — thiếu chữ "e", giữ nguyên theo đúng cách
  request.json gốc của user đã viết, KHÔNG tự sửa).
- Các kỹ thuật đã resolve được gộp thành 1 section duy nhất trong prompt,
  tên **"DESIGN TECHNIQUES (optional supporting art direction)"**, chèn
  ngay sau DESIGN RULES / DECORATIVE DETAIL LIMIT (thứ tự đầy đủ xem
  `5-backend_app_architeture.md` mục 3a, bước 16-18).
- Section này luôn có 1 câu mở đầu ràng buộc chung: các kỹ thuật là **chỉ
  dẫn hỗ trợ**, không phải yêu cầu nội dung bắt buộc; không được ghi đè bất
  kỳ rule cứng nào (request JSON, text bắt buộc, độ trung thực của asset đã
  khóa, quy tắc person/logo, safe area, nền full-bleed, giới hạn trang trí);
  không được vẽ lại/nhân bản/cắt/bóp méo hay bịa ra 1 góc nhìn/UI/phiên bản
  mới của sản phẩm/người/logo đã khóa chỉ để "cho đủ" 1 kỹ thuật; nếu 1 kỹ
  thuật xung đột với các ràng buộc trên hoặc không hợp với design brief thì
  **bỏ phần xung đột đó đi** thay vì cố ép vào.
- Mỗi kỹ thuật hiện ra trong prompt dưới dạng:
  `- <label>: <description>\n  Rendering guidance: <prompt_instruction>`
  (câu `description` bằng tiếng Việt trong file config chỉ để người viết
  request.json dễ hiểu — phần **thực sự gửi cho model** là
  `prompt_instruction`, luôn viết bằng tiếng Anh).
- Sửa `config/design_techniques.json` (thêm/sửa kỹ thuật) tự động làm mất
  hiệu lực cache prompt cũ (`prompt_builder.py::_cache_key()` hash luôn nội
  dung đã resolve của từng kỹ thuật, không chỉ hash key) — không cần bump
  version hay xoá cache tay.

Ngoài `techniques`, còn 2 field liên quan cùng trong `design_options` ảnh
hưởng tới "màu nền" — không phải kỹ thuật trong file JSON này nhưng nên
dùng cùng nhau:

- `dominant_color` (hex, ví dụ `"#00853E"`) + `dominant_color_tolerance_percent`
  — chọn màu chủ đạo cho nền khi KHÔNG có ảnh `theme` đính kèm; đi qua
  `app/core/color_utils.py::tint_shade_from_dominant()` để suy thêm cặp
  `TINT_HEX`/`SHADE_HEX` (biến thể sáng hơn/tối hơn, cùng hue) — dùng cho
  chỉ dẫn gradient nền trong `prompts/design_rules.md`.
- `num_decorative_elements` — giới hạn số nhóm chi tiết trang trí thuần
  tuý (không tính asset đã cấp, text bắt buộc, hay 1 phép ẩn dụ hình ảnh
  cốt lõi truyền tải thông điệp).

---

## 2. Nhóm BACKGROUND — kỹ thuật xử lý phần nền

Dùng khi cần thêm chiều sâu/không khí cho nền mà không cạnh tranh với sản
phẩm/nhân vật. Tất cả đều **subordinate** (phụ, không được lấn át) so với
chủ thể chính.

| Key | Tên | Khi nào dùng |
|---|---|---|
| `silhoutte_watermark` | Silhouette Watermark | Muốn thêm 1 câu chuyện/chủ đề ẩn vào nền (bóng người, toà nhà, cành lá...) ở độ mờ rất thấp (10-15%), không được vẽ thành 1 lớp riêng biệt sắc nét — chỉ đọc như 1 biến đổi tông màu tinh tế. |
| `textured_surface` | Textured Surface Background | Muốn nền có vân loang tự nhiên (đá cẩm thạch/vải nhung) thay vì màu phẳng/gradient hoàn hảo — cảm giác sang trọng, chiều sâu vật lý. |
| `radial_spotlight` | Radial Spotlight / Halo + Framing | Muốn có 1 vòng sáng mềm sau vùng đặt chữ/sản phẩm để dẫn hướng ánh nhìn, kết hợp khung hình học/đường cong/dải sáng trừu tượng ở góc/viền. |
| `background_defocus_falloff` | Depth-of-Field Defocus Falloff | Nền là ảnh/khung cảnh thật (không phải minh hoạ) và muốn nó "tan" dần thành bokeh mờ ra xa chủ thể, giống hiệu ứng khẩu độ lớn trong nhiếp ảnh — nhường không gian cho gradient màu chủ đạo thay vì cảnh thật lấn hết khung. |
| `theme_color_gradient_wash` | Theme Color Gradient Wash | Muốn màu chủ đạo (theme color) phủ dần từ giữa ra góc/viền, khiến nền thật (nếu có) chỉ còn rõ ở giữa, các góc ngả hẳn sang màu thương hiệu — khác `silhoutte_watermark` (ẩn hoạ tiết) hay `tech_pattern_overlay` (hoạ tiết hình học): đây là phủ MÀU THUẦN, không hoạ tiết. |
| `colored_corner_vignette` | Colored Corner Vignette | Giống vignette đen/trắng cổ điển nhưng dùng màu thương hiệu thay vì tối đen — 4 góc đậm dần màu chủ đạo, vùng giữa quanh chủ thể giữ sạch/sáng. |
| `duotone_diagonal_split` | Duotone Diagonal Split | Muốn chia nền thành 2 mảng màu tương phản theo đường chéo (ranh giới là dải ruy băng lượn sóng, không phải đường thẳng cứng) — cảm giác trẻ trung, năng động, phân vùng bố cục (ví dụ tách vùng nhân vật khỏi vùng nội dung) mà không cần khung kẻ. |
| `horizon_skyline_layer` | Horizon Skyline Layer | Muốn 1 dải bóng công trình/toà nhà xếp lớp chạy dọc chân nền, độ mờ CAO hơn `silhoutte_watermark` — tạo chiều sâu 3 lớp rõ ràng (trời — skyline — nhân vật/sản phẩm). |
| `comic_burst_rays` | Comic / Halftone Burst Rays | Muốn tia sáng cạnh cứng (đôi khi kèm chấm bi halftone) toả ra sau nhân vật/sản phẩm — phong cách Gen Z, campaign giảm giá sốc. Là biến thể đồ hoạ/góc cạnh của `radial_spotlight` (vốn là glow mềm). |
| `floating_reward_particles` | Floating Reward Particles | Muốn rải đồng xu vàng/tờ tiền bay/hộp quà nhỏ/hạt sparkle khắp khung hình (mang ý nghĩa thưởng/tài lộc/khuyến mãi) — khác `radial_spotlight`'s "organic framing elements" (cố định ở góc), đây rải khắp vùng nền mở. |
| `tech_pattern_overlay` | Tech Pattern Overlay | Muốn cảm giác số hoá/app/fintech — khối đa giác trừu tượng hoặc lưới (grid) mờ phía sau chủ thể, độ mờ thấp (8-20%), khác `textured_surface` (vân đá/nhung tự nhiên) vì đây là hoạ tiết hình học/công nghệ. |

## 3. Nhóm PRODUCT — kỹ thuật xử lý vùng sản phẩm/nhân vật/chữ

Dùng khi cần tăng độ "thật"/độ chuyên nghiệp cho vùng chính (nhân vật, sản
phẩm, số liệu, UI, headline) — nhóm này bổ sung cho quy tắc khoá cứng
product/context đã có sẵn trong `prompt_builder.py`, không thay thế.

| Key | Tên | Khi nào dùng |
|---|---|---|
| `character_cutout_ground_shadow` | Character Cutout + Ground Shadow | Nhân vật là ảnh chụp thật đã cắt nền, đặt trên nền hoàn toàn không phải ảnh thật — LUÔN cần bóng đổ mềm/quầng sáng dưới chân để "neo" nhân vật vào bối cảnh, tránh cảm giác dán đè. |
| `product_pop_out_frame` | Product Pop-Out / Break-Out From Frame | Muốn vật thể (voucher, số tiền, icon quà) bay vọt ra khỏi màn hình điện thoại/khung UI, đổ bóng lên mặt kính — hiệu ứng 3D, quà "rơi ra" trực tiếp cho người xem. |
| `numeric_ribbon_badge` | Numeric Ribbon / Ticket Badge | **Nên cân nhắc dùng gần như mặc định** cho mọi số tiền/ưu đãi — đặt số liệu trong khung ruy băng bẻ góc/ticket răng cưa/thẻ bo góc nổi khối thay vì để trôi tự do trên nền trơn; tách biệt số liệu khỏi nền bằng HÌNH KHỐI, không chỉ bằng màu sắc. |
| `callout_pointer_badge` | Callout Pointer Badge | Muốn dẫn mắt người xem đến đúng vị trí hành động trên điện thoại — icon tròn nổi bật + tay nhân vật đang chỉ vào, giống 1 chú thích UI thật, tăng tính hướng dẫn/tương tác. |
| `outlined_headline_typography` | Bold Outlined Headline Typography | **Xuất hiện trong hầu hết banner ngân hàng thực tế đã khảo sát** — headline luôn có viền contour (tối trên nền sáng / sáng-vàng trên nền tối) + đổ bóng nhẹ, đảm bảo đọc được dù nền nhiều chi tiết/độ tương phản cục bộ thấp. Cân nhắc dùng cùng lúc với `numeric_ribbon_badge` cho hầu hết mọi banner. |
| `live_app_ui_mockup` | Live App UI Mockup | Muốn màn hình điện thoại hiện ảnh chụp giao diện app THẬT (dashboard, danh sách dịch vụ, nút bấm) thay vì chỉ 1 icon/logo chung chung — tăng cảm giác "sản phẩm số thật". |
| `feature_icon_cluster` | Feature Icon Cluster | Muốn giới thiệu 3-5 tính năng/dịch vụ nhỏ (chuyển tiền, tiết kiệm...) cạnh điện thoại, xếp lỏng như 1 "chòm sao" thay vì lưới cứng — bổ sung thông tin mà không rối headline. |
| `product_lineup_strip` | Product Lineup Strip | Muốn giới thiệu NHIỀU mẫu thẻ/sản phẩm (nhiều SKU/hạng thẻ) trong cùng 1 banner — xếp chồng/xoè thành 1 dải gọn ở cuối banner như 1 "danh mục" thu nhỏ. |
| `seasonal_corner_tag` | Seasonal Corner Ribbon Tag | Muốn báo hiệu campaign theo dịp lễ/mùa vụ bằng 1 nhãn cờ đuôi nheo/ruy băng nhỏ ở góc trên — tách biệt với logo chính, không chiếm không gian bố cục. |

## 4. Nhóm DECORATION — trang trí theo dịp lễ/sự kiện

Mỗi kỹ thuật gắn với 1 ngày lễ cụ thể của Việt Nam, có field `date` riêng để
dễ tra theo lịch campaign. **Toàn bộ nhóm này đều có ràng buộc rõ ràng về
mức độ trừu tượng/trung lập chính trị** — đọc kỹ `description` trước khi
dùng, đặc biệt 2 ngày lễ có yêu cầu "không vẽ lãnh tụ/hình ảnh chính trị".

| Key | Ngày | Tên | Lưu ý quan trọng |
|---|---|---|---|
| `national_day_0209` | 02/09 | Quốc Khánh 2/9 | Ruy băng đỏ-vàng, sao vàng 5 cánh, pháo hoa. **KHÔNG** vẽ lãnh tụ/chân dung/biểu tượng chính trị cụ thể — chỉ hoạ tiết trung lập. |
| `woman_day_0803` | 08/03 | Quốc tế Phụ nữ 8/3 | Hoa hồng/tulip, ruy băng lụa, tông hồng-tím pastel — nhẹ nhàng, thanh thoát. |
| `woman_day_2010` | 20/10 | Phụ nữ Việt Nam 20/10 | Hoa sen, hoạ tiết áo dài cách điệu (KHÔNG vẽ người mặc thật), tông hồng đào-vàng ấm — trang trọng hơn 8/3. |
| `noel` | 25/12 | Noel / Christmas | Quả châu, bông tuyết, đèn lấp lánh, chuông, ông già Noel/tùng lộc kéo xe, cây thông, đỏ-xanh lá-vàng. |
| `tet_holiday` | Tết Nguyên Đán (âm lịch) | Tết Nguyên Đán | Hoa mai/đào, phong bao lì xì đỏ, đèn lồng, pháo giấy, tông đỏ-vàng chủ đạo. |
| `new_year` | 01/01 | Tết Dương lịch / New Year | Pháo hoa, kim tuyến vàng-bạc, tông vàng-bạc-navy. **KHÔNG** vẽ số năm cụ thể trừ khi nằm trong text yêu cầu. |
| `vn_unique_3004` | 30/04 | 30/4 Giải Phóng Miền Nam | Ruy băng cờ đỏ-vàng, chim bồ câu hoà bình, hoa sen. **KHÔNG** vẽ lãnh tụ/hình ảnh quân sự/vũ khí cụ thể. |
| `mid_autumn` | Rằm tháng 8 (âm lịch) | Trung Thu | Lồng đèn ông sao/cá chép, trăng tròn, múa lân, tông vàng ánh trăng-đỏ. |
| `children_day` | 01/06 | Quốc tế Thiếu nhi 1/6 | Bóng bay nhiều màu, kẹo, đồ chơi, cầu vồng — phong cách vui tươi hoạt hình nhẹ. |

---

## 5. Gợi ý phối hợp kỹ thuật (kết quả khảo sát 6 banner ngân hàng thật)

Từ khảo sát thực tế 6 banner khuyến mãi ngân hàng (LPBank, VCB, BIDV x3,
VietinBank — xem lịch sử phân tích, đã lưu tại
`version_1/7-design-techniques-extended-analysis.md`), rút ra 2 nhận xét
đáng chú ý khi chọn kỹ thuật cho 1 request mới:

1. **3 kỹ thuật nền gốc ban đầu** (`silhoutte_watermark`, `radial_spotlight`,
   `textured_surface`) thiên về phong cách nền tối giản/cao cấp (trà, thiệp
   lễ) — phần "tay nghề" phân biệt của banner chuyển đổi bán lẻ (khuyến mãi,
   thẻ, ưu đãi số tiền) lại nằm chủ yếu ở **vùng sản phẩm/nhân vật/chữ**
   (nhóm mục 3), không phải ở nền. Với 1 banner khuyến mãi/CTA rõ ràng, nên
   ưu tiên chọn ít nhất 1-2 kỹ thuật nhóm PRODUCT trước khi thêm kỹ thuật
   nhóm BACKGROUND.
2. **`outlined_headline_typography`** và **`numeric_ribbon_badge`** xuất
   hiện ở gần như MỌI banner đã khảo sát — coi đây là 2 lựa chọn "mặc định
   nên bật" cho hầu hết request có số tiền/ưu đãi + headline nổi bật, thay
   vì chỉ dùng khi có yêu cầu đặc biệt.

Ví dụ 1 tổ hợp `design_options.techniques` hợp lý cho banner khuyến mãi có
số tiền + nhân vật cắt nền:

```json
{
  "design_options": {
    "techniques": [
      "outlined_headline_typography",
      "numeric_ribbon_badge",
      "character_cutout_ground_shadow",
      "radial_spotlight"
    ]
  }
}
```

---

*Danh sách trên khớp 1-1 với `config/design_techniques.json` tại thời điểm
2026-09-06 — tổng cộng 29 key: 11 nhóm `background` (3 kỹ thuật gốc ban đầu
`silhoutte_watermark`/`textured_surface`/`radial_spotlight` + 8 kỹ thuật mở
rộng sau này), 9 nhóm `product`, và 9 nhóm `decoration` theo dịp lễ. Khi
thêm kỹ thuật mới vào file JSON, cập nhật lại bảng tương ứng ở tài liệu này
trong cùng lần sửa để 2 nguồn không lệch nhau.*
