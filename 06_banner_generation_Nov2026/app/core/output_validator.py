"""Post-generation validation over the FINAL (post-crop) image, run once
per (model, version) right after `crop_canvas_to_output()` in
`app/core/pipeline.py::run_generate()` -- closes the loop the prompt-side
guards (`guideline_rewriter.py` / `prompt_bleed_guard.py`) can only ASK the
image model for, never confirm actually happened.

2026-08-25 (entry (18)): a real run -- the first one where BOTH of those
prompt-side guards finally ran successfully (see entry (17)) -- still showed
2 concrete defects prompt-engineering alone didn't catch:
(1) the model copied the SAFE AREA GUIDE's own magenta border + caption
    text INTO its generated scene, directly against the prompt's explicit
    "this picture is NOT a design asset -- do not reproduce it" wording.
    Happened to land outside the final crop box that one time, purely by
    luck -- nothing guarantees that next time.
(2) the product asset (a bank card) was simply missing from 2 of 3
    generated versions, despite the MUST rule to use every populated
    bucket.

Checks below combine a cheap deterministic magenta scan with best-effort
vision checks. Supplied person/product assets are compared with their real
references (including realism/style fidelity), and a separate check rejects
color-independent technical artifacts such as rulers, arrows and px labels.
An empty bucket is never treated as "missing".

2026-08-26: entry (19) briefly replaced checks (2) with a comparison
against a COMPOSITED image instead of an AI-drawn one, then that whole
architecture was reverted (see `4-project-evaluation-results.md` entry
(19)'s follow-up note -- the composited result looked crude/pasted-on) --
back to checking the model's own drawn output, as below."""
from __future__ import annotations

import io
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.core.asset_resolver import resolve_assets
from app.core.text_style_profiles import profile_for
from app.llm.base import LLMProviderError
from app.llm.factory import LLMFactory
from app.logging_conf import get_logger
from app.models.schemas import CreativeRequest
from app.settings import get_settings

logger = get_logger("output_validator")

# Same color as `app/core/safe_area_guide.py::_BOX_COLOR` -- chosen there
# specifically because it collides with no real OCB brand color, so its
# presence in a FINAL (post-crop) image is an unambiguous signal the guide
# itself leaked into the design instead of staying a non-visual reference.
_GUIDE_MAGENTA = (255, 0, 220)
_GUIDE_COLOR_TOLERANCE = 12
_GUIDE_LEAK_MIN_PIXELS = 40

# Repair policy (standard GenAI: do not spend a full image call on one
# minor warning; edit a good concept; regenerate only when the object is wrong).
# - blocker + concept → generate again from the original prompt
# - blocker + layout  → targeted edit of the failed canvas
# - 2+ warnings       → targeted edit
# - 0-1 warning, no blocker → keep the image
_CHECK_META: dict[str, dict[str, str]] = {
    "guide_color_leak": {
        "severity": "blocker",
        "family": "layout",
        "title_vi": "Rò rỉ khung safe-area (magenta)",
        "fail_vi": "Ảnh còn đường/khung kỹ thuật của guide, không phải artwork chiến dịch.",
    },
    "product_reference": {
        "severity": "blocker",
        "family": "concept",
        "title_vi": "Product khóa bị thiếu hoặc đổi",
        "fail_vi": "Sản phẩm được cung cấp không còn đúng trên banner (mất, vẽ lại, méo tỷ lệ).",
    },
    "context_reference": {
        "severity": "blocker",
        "family": "concept",
        "title_vi": "Context khóa bị thiếu hoặc đổi",
        "fail_vi": "Nhân vật/context được cung cấp không được giữ, hoặc bị thay bằng người không được phép.",
    },
    "unauthorized_person": {
        "severity": "blocker",
        "family": "concept",
        "title_vi": "Sinh người khi request cấm",
        "fail_vi": "Banner có nhân vật dù generate_person=false và không có context người.",
    },
    "logo_presence": {
        "severity": "blocker",
        "family": "layout",
        "title_vi": "Thiếu logo bắt buộc",
        "fail_vi": "Request yêu cầu logo nhưng không thấy logo trên ảnh.",
    },
    "content_containment": {
        "severity": "warning",
        "family": "layout",
        "title_vi": "Chữ/sản phẩm sát mép hoặc bị cắt",
        "fail_vi": "Vision cho rằng chữ hoặc sản phẩm quá sát mép. Đây thường là cảnh báo (false positive khá cao), không đủ để tự gen lại nếu đứng một mình.",
    },
    "technical_artifact": {
        "severity": "blocker",
        "family": "layout",
        "title_vi": "Artifact / logo thừa / metadata",
        "fail_vi": "Có khung kỹ thuật, mã màu hex, hoặc logo OCB đứng riêng khi không được phép.",
    },
    "no_unwanted_text": {
        "severity": "blocker",
        "family": "layout",
        "title_vi": "Model tự vẽ chữ (text_by_code)",
        "fail_vi": "Request để code ghép chữ nhưng model đã vẽ headline/CTA trên ảnh.",
    },
    "reserved_zone": {
        "severity": "blocker",
        "family": "layout",
        "title_vi": "Tràn cột chữ dành cho code",
        "fail_vi": "Product/người/icon đè vào vùng bên phải dành cho text_by_code.",
    },
    "vision_unavailable": {
        "severity": "warning",
        "family": "layout",
        "title_vi": "Không gọi được LLM Vision",
        "fail_vi": "Hạ tầng vision lỗi nên không có câu trả lời yes/no cho các check nội dung.",
    },
}


@dataclass
class ValidationCheck:
    name: str
    status: str  # pass | fail | skip
    severity: str  # blocker | warning | info
    family: str  # concept | layout | none
    title_vi: str
    fail_vi: str
    question: str
    answer: str  # yes | no | skip | n/a
    problem_message: str = ""


@dataclass
class ValidationReport:
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        return [c.problem_message for c in self.checks if c.status == "fail" and c.problem_message]

    @property
    def blockers(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "fail" and c.severity == "blocker"]

    @property
    def warnings(self) -> list[ValidationCheck]:
        return [c for c in self.checks if c.status == "fail" and c.severity == "warning"]

    @property
    def score(self) -> int:
        """Lower is better. Blockers outweigh warnings so keep-best prefers a
        slightly-close-to-edge image over a missing-product image."""
        return 10 * len(self.blockers) + len(self.warnings)

    @property
    def repair_mode(self) -> str:
        """none | edit | regenerate"""
        if any(c.family == "concept" for c in self.blockers):
            return "regenerate"
        if self.blockers:
            return "edit"
        if len(self.warnings) >= 2:
            return "edit"
        return "none"

    @property
    def decision_vi(self) -> str:
        mode = self.repair_mode
        if mode == "none":
            if not self.problems:
                return "GIỮ ẢNH — không phát hiện lỗi."
            return (
                "GIỮ ẢNH — chỉ có cảnh báo nhỏ, không gen lại "
                "(tránh tốn 1 lần gọi image model vì lỗi sát mép / tương đương)."
            )
        if mode == "edit":
            return "SỬA TRÊN ẢNH CŨ (edit) — đính kèm ảnh vừa gen + danh sách lỗi layout."
        return "GEN LẠI TỪ PROMPT GỐC (regenerate) — sai concept/object, không vá trên ảnh đang sai."


def _meta_check(
    name: str,
    *,
    status: str,
    question: str = "",
    answer: str = "n/a",
    problem_message: str = "",
) -> ValidationCheck:
    meta = _CHECK_META[name]
    severity = meta["severity"] if status == "fail" else "info"
    return ValidationCheck(
        name=name,
        status=status,
        severity=severity,
        family=meta["family"] if status == "fail" else "none",
        title_vi=meta["title_vi"],
        fail_vi=meta["fail_vi"],
        question=question,
        answer=answer,
        problem_message=problem_message if status == "fail" else "",
    )


def format_validation_report_text(
    report: ValidationReport,
    *,
    request_name: str,
    model: str,
    version: int,
    attempt: int,
) -> str:
    """UTF-8 text for nghiệp vụ: đọc được không cần mở code."""
    lines = [
        "BÁO CÁO VALIDATE ẢNH (LLM Vision + check kỹ thuật)",
        "=" * 72,
        f"Request     : {request_name}",
        f"Model       : {model}",
        f"Version     : {version}",
        f"Lần gen     : attempt {attempt}",
        f"Thời điểm   : {datetime.now(timezone.utc).isoformat()}",
        "",
        f"QUYẾT ĐỊNH  : {report.decision_vi}",
        f"repair_mode : {report.repair_mode}",
        f"Điểm keep-best (thấp hơn = tốt hơn): {report.score}  "
        f"(blocker x10 + warning x1)",
        "",
        f"Tổng lỗi chặn : {len(report.blockers)}",
        f"Tổng cảnh báo : {len(report.warnings)}",
        "",
        "CHI TIẾT TỪNG CHECK",
        "-" * 72,
    ]
    for index, check in enumerate(report.checks, start=1):
        lines.append(f"{index}. [{check.status.upper()}] {check.name} — {check.title_vi}")
        lines.append(f"   Mức: {check.severity} | Nhóm: {check.family}")
        if check.question:
            q = check.question.replace("\n", " ").strip()
            if len(q) > 400:
                q = q[:400] + "…"
            lines.append(f"   Câu hỏi gửi vision LLM: {q}")
        lines.append(f"   Trả lời vision/check: {check.answer}")
        if check.status == "fail":
            lines.append(f"   Ý nghĩa nghiệp vụ: {check.fail_vi}")
            if check.problem_message:
                lines.append(f"   Mô tả kỹ thuật: {check.problem_message}")
        lines.append("")
    if not report.checks:
        lines.append("(Không chạy được check nào — thường do LLM vision không khởi tạo được.)")
        lines.append("")
    lines.append("Ghi chú: 'skip' = không gọi được vision (lỗi hạ tầng), không tính là fail.")
    lines.append("Chính sách repair: 0-1 cảnh báo và 0 lỗi chặn → giữ ảnh; lỗi layout chặn → edit;")
    lines.append("lỗi concept (sai product/người) → gen lại từ prompt gốc.")
    return "\n".join(lines) + "\n"


def has_guide_color_leak(image_bytes: bytes) -> bool:
    """True if the final image contains a non-trivial number of pixels
    matching the SAFE AREA GUIDE's distinctive magenta -- i.e. the model
    copied the guide's own border/caption into its output instead of
    treating it as a non-visual spatial reference. Never raises -- an
    undecodable image just can't be checked, not a failure of this check."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:  # noqa: BLE001
        logger.exception("Could not decode image for guide-color-leak check -- skipping this check.")
        return False

    # 2026-08-25: an earlier version downscaled to a 300px thumbnail before
    # scanning, for speed -- verified against a real case (the ORG v2 image
    # of DB_NOTI_DEMO_TRUNGTHU_huy, which DOES contain the leaked magenta
    # border) that this makes the check FALSE-NEGATIVE: the border is only
    # a few pixels wide, and downscaling blends it into the surrounding
    # background before the color comparison ever runs. Scanning at full
    # resolution is the only way to reliably catch a thin leaked line.
    r0, g0, b0 = _GUIDE_MAGENTA
    hits = 0
    for r, g, b in img.getdata():
        if abs(r - r0) <= _GUIDE_COLOR_TOLERANCE and abs(g - g0) <= _GUIDE_COLOR_TOLERANCE and abs(b - b0) <= _GUIDE_COLOR_TOLERANCE:
            hits += 1
            if hits >= _GUIDE_LEAK_MIN_PIXELS:
                return True
    return False


_GENERIC_LOGO_QUESTION = (
    "Does the FIRST image (a generated marketing banner) show a distinct brand logo mark "
    "(not just app-UI icons)? Answer only 'yes' or 'no'."
)

# `context` and `product` are approved marketing assets, not loose visual
# inspiration. Both checks therefore compare against the supplied source
# and reject a merely similar cartoon/redrawing.
_CONTEXT_REFERENCE_QUESTION = (
    "The FIRST image is a generated marketing banner. The SECOND image is the approved context asset. A context "
    "asset may be a person, icon, object, symbol, or scene; do not assume it is a person. Does the FIRST image "
    "preserve the SECOND image's actual subject and visual identity faithfully? If the reference contains a real "
    "person, the same identity, clothing, pose/expression and photographic realism must remain. If the reference "
    "contains no person, answer 'no' if the banner replaced or supplemented it with an invented photographed "
    "person or family. Do not fail this check only because the subject sits close to an edge — that is a "
    "separate containment warning. Answer only 'yes' or 'no'."
)

_UNAUTHORIZED_PERSON_QUESTION = (
    "This request explicitly forbids generating people and supplies no person/context asset. Does this finished "
    "banner contain any real, photographic, illustrated, 3D, or silhouetted human character, face, body, couple, "
    "family, or crowd? Ignore tiny people that are inseparable incidental details of a distant background photo. "
    "Answer only 'yes' or 'no'."
)

# 2026-08-25: an earlier version of the product check asked a generic
# yes/no question ("does this image show a physical product, e.g. a bank
# card, phone, ...") with NO reference image -- verified against a real
# case (v2 of DB_NOTI_DEMO_TRUNGTHU_huy, missing the product card
# entirely) that it FALSE-POSITIVES: the vision model answered "yes"
# because a phone mockup (part of the `context`/character composition,
# not the `product` bucket) satisfied the vague wording. Comparing
# against the ACTUAL resolved bucket asset image is what makes this check
# mean what it says.
_PRODUCT_REFERENCE_QUESTION = (
    "The FIRST image is a generated marketing banner. The SECOND image is the reference "
    "product asset that banner was required to preserve. Does the SAME physical product clearly appear in "
    "the FIRST image with its logo/wordmark, printed text, colors, artwork, shape, proportions, material, "
    "and realistic commercial appearance intact -- not redrawn, cartoonized, simplified, or redesigned? Its "
    "original width-to-height aspect ratio, portrait/landscape orientation, front/back orientation, reading "
    "direction, and handedness must match the reference. Answer 'no' if it was mirrored, flipped, stretched, "
    "squashed, warped, rotated from portrait to landscape or vice versa, or redrawn with different proportions, "
    "material, or casing than the reference. A deliberate camera angle, 3D-style tilt, or rotation of the whole "
    "unchanged object (as if photographed from a different angle) is EXPECTED and should be answered 'yes' as long "
    "as the object's real shape, proportions, and artwork stay recognizable and internally consistent -- only "
    "answer 'no' if the change reveals genuinely new geometry the source never showed (e.g. an extra side/face that "
    "shouldn't exist, an impossible thickness, or a shape that contradicts the reference), not merely because the "
    "viewing angle differs from the reference photo. "
    "Do not fail this check only because the product sits close to an edge or has a tight crop — that is a "
    "separate containment warning. Answer 'no' only when the product is missing, swapped, or its identity/"
    "artwork/proportions are wrong. Answer only 'yes' or 'no'."
)

_TECHNICAL_ARTIFACT_QUESTION = (
    "Does this finished marketing banner contain a PRODUCTION artifact that is not advertising content? "
    "Answer 'yes' only for: a crop/safe-area rectangle, dotted guide, ruler, dimension arrow, coordinate, "
    "pixel or margin label such as '48px'/'60px'; a hex color token WITH OR WITHOUT '#' (E39717, #E39717, "
    "dE39717, 00804B) glued onto copy or printed as its own line; a font name (Manrope) or layout "
    "instruction printed as text; OR a decorative border/frame/vignette that wraps the WHOLE banner so the "
    "campaign background does not bleed to all four outer edges (including a thin gold or metallic line around "
    "the entire image). "
    "Answer 'no' if the only rectangles, panels, screens, windows, doors, cups, cards, phones, laptops, "
    "menu boards, or rounded UI you see are real objects inside the scene. Do not flag partner/platform "
    "icons from the campaign (Google, TikTok, Meta, Starbucks, and similar) or a logo printed on a supplied "
    "product or cup. Approved advertising copy and percentages are not artifacts. Answer only 'yes' or 'no'."
)


_NO_UNWANTED_TEXT_QUESTION = (
    "This banner's headline/message/CTA text is meant to be added separately by code, NOT drawn by you. Does "
    "this image contain ANY readable text, wordmark, slogan, button, or lettering of any kind (in any language, "
    "including a brand name or app-UI style text)? Answer only 'yes' or 'no'."
)


def _reserved_zone_question(fraction: float) -> str:
    pct = int(fraction * 100)
    return (
        f"This banner has code-rendered text that will be drawn afterward inside the right {pct}% of the image "
        "width (a vertical column along the right edge). Does the PRODUCT, the person/character, or any "
        f"prominent decorative motif/icon currently extend into that right {pct}%-wide column, instead of "
        "staying within the remaining left portion? Ignore plain background color/gradient/texture -- only "
        "answer 'yes' if a real foreground subject (product, person, or a distinct icon/motif) is there. "
        "Answer only 'yes' or 'no'."
    )


def _content_containment_question(req: CreativeRequest) -> str:
    # `text_by_code` requests never want the image model drawing text at
    # all (see prompt_builder.py's TEXT section) -- so this containment
    # check must not demand the real headline/message/CTA be visible in the
    # model's OWN drawing; that's `_NO_UNWANTED_TEXT_QUESTION`'s job instead.
    required_text = (
        []
        if req.design_options.text_by_code
        else [value for value in (req.text.headline, req.text.message, req.text.cta_label) if value]
    )
    text_requirement = (
        "Every one of these exact required strings must be fully visible and untruncated: "
        f"{json.dumps(required_text, ensure_ascii=False)}. "
        if required_text
        else "There is no required advertising text for this request. "
    )
    required_assets: list[str] = []
    if req.buckets.product:
        required_assets.append("the supplied product")
    if req.buckets.context:
        required_assets.append("the supplied context subject")
    if req.design_options.has_logo:
        required_assets.append("the required logo")
    asset_requirement = ", ".join(required_assets) or "any intentionally prominent advertising subject"
    return (
        "Inspect this finished marketing banner, especially all four output edges. "
        + text_requirement
        + f"The complete outer boundary of {asset_requirement} must also be visible. "
        "Are ALL required text and required advertising elements completely inside the image with comfortable "
        "separation from the edges? Answer 'no' if any required character, word, letter, logo, product corner, "
        "or product detail touches an edge, extends beyond it, or appears clipped. Ignore background scenery "
        "and purely decorative full-bleed elements. Answer only 'yes' or 'no'."
    )


def _mime_for(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _ask_yes_no(
    provider,
    images: list[tuple[bytes, str]],
    question: str,
    *,
    usage_tracker=None,
    version: int | None = None,
    check_name: str = "",
) -> tuple[bool | None, str]:
    """Returns `(result, answer_for_log)`. `result is None` means the check
    itself couldn't be completed (LLM call failed) -- callers must treat
    that as "skip", never as a failure: this is a best-effort enhancement
    and must never turn an infra hiccup into a false rejection.

    `usage_tracker`, when passed, records THIS single call as its own
    billable row (2026-09-06 user request) -- `check_required_elements()`
    can make up to ~7-8 of these for one (model, version) validation pass,
    and each is a separate, individually-priced LLM call, not an aggregate."""
    usage_sink: dict = {} if usage_tracker is not None else None
    try:
        answer = provider.generate_vision(
            question, images, system="You answer only 'yes' or 'no', nothing else.", usage_sink=usage_sink
        )
        raw = (answer or "").strip()
        yes = raw.lower().startswith("y")
        token = "yes" if yes else "no"
        lowered = raw.lower()
        if lowered not in {"yes", "no", "y", "n"} and raw:
            token = f"{token} (raw: {raw[:120]})"
        return yes, token
    except LLMProviderError:
        logger.exception("Vision presence-check call failed -- skipping this check.")
        return None, "skip"
    except Exception:  # noqa: BLE001
        logger.exception("Could not run vision presence-check -- skipping this check.")
        return None, "skip"
    finally:
        if usage_tracker is not None:
            usage_tracker.record_llm(
                usage_sink,
                stage=f"output_validator:{check_name or 'presence_check'}",
                version=version,
                fallback_model=get_settings().llm_model_name,
            )


def _vision_check(
    name: str,
    *,
    result: bool | None,
    answer: str,
    question: str,
    fail_if: bool,
    problem_message: str,
) -> ValidationCheck:
    if result is None:
        return _meta_check(name, status="skip", question=question, answer=answer or "skip")
    if result is fail_if:
        return _meta_check(
            name,
            status="fail",
            question=question,
            answer=answer,
            problem_message=problem_message,
        )
    return _meta_check(name, status="pass", question=question, answer=answer)


def _collect_required_element_checks(
    image_bytes: bytes, req: CreativeRequest, *, usage_tracker=None, version: int | None = None
) -> list[ValidationCheck]:
    """Only checks buckets ACTUALLY populated on `req` (`buckets.product`,
    `buckets.context`, `design_options.has_logo`) -- an empty/absent bucket
    is never treated as a missing requirement. `product`/`context` are
    checked by comparing against the real resolved reference asset image,
    not a generic guess (see module note above for why that matters).

    `usage_tracker`/`version`, when passed, are threaded down to every
    individual `_ask_yes_no()` vision call made below (2026-09-06 user
    request -- see `app/core/usage_tracker.py`)."""
    settings = get_settings()
    try:
        provider = LLMFactory.create(settings)
    except Exception:  # noqa: BLE001
        logger.exception("Could not create LLM provider for presence checks -- skipping.")
        return [
            _meta_check(
                "vision_unavailable",
                status="skip",
                question="(không gửi được câu hỏi — LLM provider không khởi tạo được)",
                answer="skip",
            )
        ]
    if not hasattr(provider, "generate_vision"):
        return [
            _meta_check(
                "vision_unavailable",
                status="skip",
                question="(provider hiện tại không hỗ trợ generate_vision)",
                answer="skip",
            )
        ]

    checks: list[ValidationCheck] = []
    assets = resolve_assets(req)
    banner = [(image_bytes, "image/png")]

    if assets.product:
        try:
            ref_bytes = assets.product[0].read_bytes()
        except OSError:
            logger.exception("Could not read reference product asset %s -- skipping this check.", assets.product[0])
            checks.append(
                _meta_check("product_reference", status="skip", question=_PRODUCT_REFERENCE_QUESTION, answer="skip")
            )
        else:
            images = [(image_bytes, "image/png"), (ref_bytes, _mime_for(assets.product[0]))]
            result, answer = _ask_yes_no(
                provider, images, _PRODUCT_REFERENCE_QUESTION,
                usage_tracker=usage_tracker, version=version, check_name="product_reference",
            )
            checks.append(
                _vision_check(
                    "product_reference",
                    result=result,
                    answer=answer,
                    question=_PRODUCT_REFERENCE_QUESTION,
                    fail_if=False,
                    problem_message=(
                        "product asset missing or changed: preserve exact artwork, proportions, and realism; do not "
                        "recreate, mirror, flip, stretch, squash, warp, or rotate into the opposite orientation; a "
                        "deliberate camera-angle tilt of the unchanged object is fine, but do not invent new geometry "
                        "(extra side/face, impossible thickness) that contradicts the reference"
                    ),
                )
            )

    if assets.context:
        try:
            ref_bytes = assets.context[0].read_bytes()
        except OSError:
            logger.exception("Could not read reference context asset %s -- skipping this check.", assets.context[0])
            checks.append(
                _meta_check("context_reference", status="skip", question=_CONTEXT_REFERENCE_QUESTION, answer="skip")
            )
        else:
            images = [(image_bytes, "image/png"), (ref_bytes, _mime_for(assets.context[0]))]
            result, answer = _ask_yes_no(
                provider, images, _CONTEXT_REFERENCE_QUESTION,
                usage_tracker=usage_tracker, version=version, check_name="context_reference",
            )
            checks.append(
                _vision_check(
                    "context_reference",
                    result=result,
                    answer=answer,
                    question=_CONTEXT_REFERENCE_QUESTION,
                    fail_if=False,
                    problem_message="context asset missing/changed, or an unauthorized person was invented",
                )
            )

    # 2026-09-04 (user request): `context` may now be a free-text
    # description instead of an image (`assets.context_text` --
    # `app/core/asset_resolver.py::resolve_assets()`). A description that
    # intentionally calls for a character is exactly like a supplied
    # context IMAGE from this check's point of view -- the resulting
    # person is authorized, not invented -- so it must gate this check the
    # same way `assets.context` already does, otherwise every text-
    # described character would be flagged as "unauthorized".
    if not req.design_options.generate_person and not assets.context and not assets.context_text:
        result, answer = _ask_yes_no(
            provider, banner, _UNAUTHORIZED_PERSON_QUESTION,
            usage_tracker=usage_tracker, version=version, check_name="unauthorized_person",
        )
        checks.append(
            _vision_check(
                "unauthorized_person",
                result=result,
                answer=answer,
                question=_UNAUTHORIZED_PERSON_QUESTION,
                fail_if=True,
                problem_message="unauthorized person/character was generated although generate_person=false",
            )
        )

    if req.design_options.has_logo:
        result, answer = _ask_yes_no(
            provider, banner, _GENERIC_LOGO_QUESTION,
            usage_tracker=usage_tracker, version=version, check_name="logo_presence",
        )
        checks.append(
            _vision_check(
                "logo_presence",
                result=result,
                answer=answer,
                question=_GENERIC_LOGO_QUESTION,
                fail_if=False,
                problem_message="logo not visible in the generated image",
            )
        )

    containment_q = _content_containment_question(req)
    result, answer = _ask_yes_no(
        provider, banner, containment_q,
        usage_tracker=usage_tracker, version=version, check_name="content_containment",
    )
    checks.append(
        _vision_check(
            "content_containment",
            result=result,
            answer=answer,
            question=containment_q,
            fail_if=False,
            problem_message="required text or advertising asset is clipped or too close to an output edge",
        )
    )

    technical_question = _TECHNICAL_ARTIFACT_QUESTION
    if not req.design_options.has_logo:
        technical_question += (
            " This request also has has_logo=false. Answer 'yes' only if a standalone OCB bank logo, OCB lockup, "
            "OCB corner mark, or OCB watermark appears outside the supplied product/card. Do not flag partner "
            "or platform marks that belong to the campaign scene. A logo printed on the supplied product itself "
            "is allowed."
        )
    result, answer = _ask_yes_no(
        provider, banner, technical_question,
        usage_tracker=usage_tracker, version=version, check_name="technical_artifact",
    )
    checks.append(
        _vision_check(
            "technical_artifact",
            result=result,
            answer=answer,
            question=technical_question,
            fail_if=True,
            problem_message=(
                "forbidden standalone logo, safe-area/frame artifact, hex color code, or other production metadata is visible"
            ),
        )
    )

    if req.design_options.text_by_code:
        result, answer = _ask_yes_no(
            provider, banner, _NO_UNWANTED_TEXT_QUESTION,
            usage_tracker=usage_tracker, version=version, check_name="no_unwanted_text",
        )
        checks.append(
            _vision_check(
                "no_unwanted_text",
                result=result,
                answer=answer,
                question=_NO_UNWANTED_TEXT_QUESTION,
                fail_if=True,
                problem_message="image model drew its own text/wordmark even though text_by_code=true -- text must be added by code only",
            )
        )

        profile = profile_for(req.final_output_wh())
        if profile is not None:
            reserved_q = _reserved_zone_question(profile.text_zone_right_fraction)
            result, answer = _ask_yes_no(
                provider, banner, reserved_q,
                usage_tracker=usage_tracker, version=version, check_name="reserved_zone",
            )
            checks.append(
                _vision_check(
                    "reserved_zone",
                    result=result,
                    answer=answer,
                    question=reserved_q,
                    fail_if=True,
                    problem_message=(
                        f"product/context/decorative content spills into the reserved right {int(profile.text_zone_right_fraction * 100)}% "
                        "text column -- code-rendered text will overlap it; keep that column empty background only"
                    ),
                )
            )

    return checks


def check_required_elements(
    image_bytes: bytes, req: CreativeRequest, *, usage_tracker=None, version: int | None = None
) -> list[str]:
    """Problem strings only (compat). Prefer `validate_output_report()` for
    severity, vision Q&A, and the human-readable txt log."""
    return [
        check.problem_message
        for check in _collect_required_element_checks(
            image_bytes, req, usage_tracker=usage_tracker, version=version
        )
        if check.status == "fail" and check.problem_message
    ]


def validate_output_report(
    image_bytes: bytes, req: CreativeRequest, *, usage_tracker=None, version: int | None = None
) -> ValidationReport:
    """Full validation with severity so the generate-repair graph can decide
    keep / edit / regenerate, and so nghiệp vụ can read the txt log."""
    checks: list[ValidationCheck] = []
    if has_guide_color_leak(image_bytes):
        checks.append(
            _meta_check(
                "guide_color_leak",
                status="fail",
                question="Deterministic pixel scan for the safe-area guide magenta (255, 0, 220).",
                answer="detected",
                problem_message="safe-area guide's own magenta border/caption leaked into the generated image",
            )
        )
    else:
        checks.append(
            _meta_check(
                "guide_color_leak",
                status="pass",
                question="Deterministic pixel scan for the safe-area guide magenta (255, 0, 220).",
                answer="not detected",
            )
        )
    checks.extend(
        _collect_required_element_checks(image_bytes, req, usage_tracker=usage_tracker, version=version)
    )
    return ValidationReport(checks=checks)


def write_validation_report_file(
    report: ValidationReport,
    path: Path,
    *,
    request_name: str,
    model: str,
    version: int,
    attempt: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_validation_report_text(
            report,
            request_name=request_name,
            model=model,
            version=version,
            attempt=attempt,
        ),
        encoding="utf-8",
    )
    return path


def validate_output(
    image_bytes: bytes, req: CreativeRequest, *, usage_tracker=None, version: int | None = None
) -> list[str]:
    """Returns human-readable problems found in the FINAL (post-crop)
    image -- an empty list means it passed every check that applies to
    this request. Never raises."""
    return validate_output_report(image_bytes, req, usage_tracker=usage_tracker, version=version).problems


# ---------------------------------------------------------------------------
# Chạy độc lập file này (kiểm tra 1 ảnh đã có sẵn trên đĩa):
#   python -m app.core.output_validator --image /path/to/final.png --request DB_NOTI_DEMO_TRUNGTHU_huy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.core.pipeline import load_request_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--request", required=True, help="request_name (reads input/<request>.json for its buckets)")
    args = parser.parse_args()

    req = load_request_json(args.request)
    data = open(args.image, "rb").read()
    report = validate_output_report(data, req)
    print(
        format_validation_report_text(
            report,
            request_name=args.request,
            model="offline",
            version=0,
            attempt=1,
        ),
        end="",
    )
