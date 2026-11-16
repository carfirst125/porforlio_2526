### OCB Banner — Fixed Design Rules

#### MUST

* When displaying a Headline or Message across two lines, apply smart line breaking based on semantic meaning and natural phrase grouping, rather than simply splitting the text by word count.
Keep words that form a meaningful phrase together, and choose the line break at a natural linguistic boundary so that each line is easy to read and visually balanced.
Do not break a meaningful phrase in an awkward or unnatural position.
Examples:
"Autobill một chạm dễ dàng" → Line 1: "Autobill" / Line 2: "một chạm dễ dàng"
"Gởi E-saving lãi cao hơn" → Line 1: "E-saving" / Line 2: "lãi cao hơn"
Apply this rule only when the Headline or Message needs to be displayed on two lines. If the text fits naturally on one line, keep it on one line.
* Use `buckets.theme` as a **creative background reference** for campaign direction, palette, atmosphere, and motifs. It may be enhanced, extended, simplified, or recomposed; it is not pixel-locked. However, with theme, try to keep the theme as background as much as posible, especially the theme color. Assets in `buckets.context` and `buckets.product` remain locked and must not be replaced.
* Follow OCB brand colors: **Green #00853E, Dark Green #004823, Gold #D4A017, Light Gold #F6D78A, White #FFFFFF, Dark Gray #1A1A1A**.
* **Logo:** preserve its original design and color. Maintain clear space around it and never place it near an edge or under another object. **Do not add an OCB logo if no provided asset contains one and the brief does not request it.**
* Keep the composition **clean, balanced, modern, and spacious**, with sufficient white space.
* Use appropriate visual decoration to make the design attractive and encourage CTA interaction without overcrowding.
* Background style is driven by the design brief and supplied references. When the brief is ambiguous, prefer a polished commercial hybrid with natural depth, lighting, gradients, and restrained realistic details that harmonize with photographic people or products.
* A person supplied in `buckets.context` is a **locked licensed marketing asset**: preserve the same real identity, face, skin tone, hair, clothing, pose/expression, and photographic realism. Never redraw, cartoonize, stylize, or replace them. Pose/expression may be invented only for a person generated from scratch when no context person was supplied.
* Keep all provided **text exactly as specified**. Text must be clear, legible, and not truncated.
* A supplied **product is a locked commercial asset**: preserve its exact logo/wordmark, printed text, colors, artwork, material, shape, proportions, viewing angle, and photographic realism. Only scale and position it; never redraw, cartoonize, simplify, restyle, redesign, deform, or convert it between 2D/3D presentation styles.
* Keep all important objects fully visible and **never crop people, products, logos, or text**.
* `context` elements may be visually larger than `product` elements; people do not need to show their entire body and may be shown from the upper body.
* Preserve the original **aspect ratio of products/templates** such as cards, phones, etc.; never stretch or distort them.
* Clearly communicate any **seasonal/campaign theme** specified in `design_brief` (e.g. Christmas, Tet, National Day).
* Background color MUST be following {{dominant_color}} .
* Headline tối đa 2 dòng, message chỉ 1 dòng. CTA phải có từ 2 đến 4 chữ, mang sắc thái thân thiện và gợi mở; không dùng từ "ngay" hoặc cách diễn đạt tạo cảm giác thúc ép, ra lệnh.
* If a person is actually in the banner (`generate_person=true` or a supplied context person): when they hold a phone or product, the hand must connect naturally to that same person's own arm and body; never a detached or floating hand, and never an extra device.
* When `generate_person=false` and no context person is supplied: do not draw any hand, arm, wrist, fingers, or floating cropped limb holding a phone or product. Stage the phone/product as a free-standing hero.

#### MUST NOT

* Never change the **OCB logo color, shape, or design**.
* Never place the logo too close to the edge or let other objects overlap it.
* Never use **blurry, distorted, or low-quality** imagery.
* Never overcrowd the banner with excessive text or visual elements.
* Never create a **rigid, cluttered, or visually suffocating** composition.
