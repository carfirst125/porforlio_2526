"""Pydantic models for `request.json` (REQUEST.md [1]) and small internal
DTOs shared between `app/core/*` and `app/api/routes.py`.

Schema notes (2026-08-20 correction -- see `4-project-evaluation-results.md`
for the full story): the request.json schema below follows the LITERAL
template printed at the top of REQUEST.md, matched field-for-field against
the real, current `input/DB_NOTI_AUTOPAY_NOEL.json` on disk. An EARLIER pass
of this file (2026-08-19) had instead reverse-engineered a flatter, DIFFERENT
schema (`output.size` as a `"W,H"` string = the generation canvas,
`design_options.protect_area` as an optional `"W,H"` string = the real final
output) from what turned out to be a STALE leftover `input/*.json` file the
user had forgotten to delete -- that flat schema is now WRONG and has been
fully replaced by the one below. Do not reintroduce `protect_area`.

- `generation.canvas` (`{width, height}`, REQUIRED) is the size the Image
  GenAI model is actually asked to generate at.
- `generation.safe_area` (`{width, height, position, x, y}`, REQUIRED) is the
  box, in canvas pixel coordinates, that all important design content
  (text/product/character/logo) must stay fully inside -- everything outside
  it (but still inside the canvas) may only be background/decorative
  extension, because it gets cropped away and discarded after generation.
  `x`/`y` are the safe area's top-left corner offset inside the canvas.
- `generation.instruction` (str, optional) is free-text guidance for the
  image model about the canvas/safe-area relationship, folded into the
  prompt verbatim (see `app/core/prompt_builder.py`).
- `output.size` (`{width, height}`, REQUIRED) is the REAL final output pixel
  size written to disk -- REQUEST.md [1]: "chính là kích thước trong
  safe_area" (this is, in every real request.json seen so far, identical to
  `generation.safe_area.width/height`).
- `output.crop` (optional) explicitly says how to derive the crop box from
  the canvas -- `{source: "generation.canvas", area: {x, y, width, height}}`.
  When present its `area` is used as the crop box; when ABSENT (a request.json
  is allowed to omit it, since it's redundant with `generation.safe_area`),
  the crop box falls back to `generation.safe_area`'s own `x`/`y`/`width`/
  `height` -- both describe the exact same box in every real file so far.
- `output.clone` (`list[str]`, `"W,H"` strings, default `[]`) -- other sizes
  to additionally render via `--mode=clone` / `POST /clone`, reusing the
  released design as a reference image (REQUEST.md [4]).
- `buckets.theme` / `.context` / `.product` can each be a single path
  string, a list of path strings (multiple assets in one bucket), or `[]`
  (empty bucket, e.g. no character supplied).
- 2026-09-04 (user request): any entry in `buckets.theme` / `.context` /
  `.product` can ALSO be a free-text description instead of an image path
  -- e.g. `"theme": "không gian xanh lá gradient, quầy tính tiền..."`
  describes the desired background in words rather than attaching a
  reference image. `app/core/asset_resolver.py::looks_like_image_path()`
  classifies each entry purely by its file extension (`.png`/`.jpg`/etc. ->
  image path; anything else -> free-text description) -- no filesystem
  access, so a genuinely missing/typo'd image file still surfaces as a real
  "file not found" error instead of silently being treated as descriptive
  text. Text descriptions are folded into the generation prompt as creative
  direction for what the image model should draw in that bucket's role
  (`app/core/prompt_builder.py`), never attached as an image and never
  compared against a reference in `app/core/output_validator.py`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def parse_wh(raw: str) -> tuple[int, int]:
    """`"1035,420"` -> `(1035, 420)`. Also tolerates `"1035x420"` /
    `"1035 , 420"` since users hand-edit these files. Only used for the
    free-form `"W,H"` size strings still in the schema (`output.clone`
    entries, and the `display_sizes` a user types into the webapp's Clone
    page) -- `generation.canvas` / `generation.safe_area` / `output.size`
    are structured `{width, height}` objects instead, see module docstring."""
    if not raw:
        raise ValueError("empty size string")
    cleaned = raw.strip().lower().replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"could not parse size string: {raw!r}")
    return int(parts[0]), int(parts[1])


def wh_slug(w: int, h: int) -> str:
    return f"{w}x{h}"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


class Buckets(BaseModel):
    theme: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    product: list[str] = Field(default_factory=list)
    brand_text: list[str] = Field(default_factory=list)

    @field_validator("theme", "context", "product", "brand_text", mode="before")
    @classmethod
    def _normalize(cls, v):
        return _as_list(v)


class TextContent(BaseModel):
    headline: str = ""
    message: str = ""
    cta_label: str = ""
    cta_deep_link: str = ""

    @field_validator("headline", "message", mode="before")
    @classmethod
    def _normalize_display_lines(cls, value):
        """Accept a string or an explicit list of display lines without
        forcing every existing string consumer to understand two shapes."""
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(line).strip() for line in value if str(line).strip())
        return str(value)


class TechniqueAutoConfig(BaseModel):
    """`design_options.technique_selection.auto_config` -- only consulted
    when `technique_selection.mode == "auto"`; completely ignored in manual
    mode (2026-09-09 user request).

    - `max_background_techniques` / `max_product_techniques` /
      `max_decoration_techniques`: EITHER a per-GROUP maximum technique
      COUNT (`int`, the original behavior -- the LLM selector in
      `app/core/technique_selector.py` picks up to that many keys from the
      catalog for that group) OR a literal LIST of exact catalog technique
      keys (2026-09-10 user request) -- when a list is given, that group is
      no longer selected by the LLM at all: the selector is told to return
      an empty list for it (effective max becomes 0) and the given keys are
      applied directly instead, after being validated against the catalog
      (must exist, belong to that exact group, and satisfy this request's
      `observe_brand`) by
      `app/core/technique_selector.py::_apply_fixed_techniques()`. An
      unknown/wrong-group/disallowed-brand key in the list is dropped with a
      warning rather than failing the request. The other group(s) may
      independently stay as a plain `int` (still auto-selected by the LLM)
      -- int and list forms are mixed freely per group.
    - `selection_constraints`: list of extra guidance file paths, resolved
      relative to `ASSET_LIBRARY_DIR` (`./lib`) -- the SAME lib-relative
      convention as `buckets.*` / `reference` / `guidelines`
      (`app/core/asset_resolver.py::_resolve_one()`). Every listed file's
      full text content is loaded and folded into the technique-selection
      LLM prompt verbatim (see
      `app/core/technique_selector.py::_resolve_selection_constraints_text()`),
      after filling any `{{dominant_color}}` placeholder with
      `design_options.dominant_color` (or a sensible fallback phrase when
      that field is empty, mirroring `app/core/prompt_builder.py`'s existing
      "OCB's approved palette" fallback for the same field). A missing/
      unreadable file is skipped and logged -- never fatal, consistent with
      every other optional-enhancement file in this project.
    """

    model_config = ConfigDict(extra="ignore")

    max_background_techniques: int | list[str] = Field(default=3)
    max_product_techniques: int | list[str] = Field(default=3)
    max_decoration_techniques: int | list[str] = Field(default=3)
    selection_constraints: list[str] = Field(default_factory=list)

    @field_validator(
        "max_background_techniques", "max_product_techniques", "max_decoration_techniques", mode="before"
    )
    @classmethod
    def _normalize_group_max(cls, v):
        """A bare string/single key is treated as a one-item fixed list
        (same forgiving convention as `Buckets`/`manual_techniques` via
        `_as_list()`); a real list has its entries stringified and blank
        entries dropped. Plain ints/numeric strings pass through unchanged
        so the original count behavior is fully preserved."""
        if isinstance(v, bool):  # bool is an int subclass -- reject explicitly, never a valid count or key
            raise ValueError("must be a non-negative integer count or a list of technique keys, not a bool")
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
            # A non-numeric string is a single fixed technique key, e.g. "theme_color_gradient_wash".
            return [v.strip()] if v.strip() else v
        return v

    @field_validator(
        "max_background_techniques", "max_product_techniques", "max_decoration_techniques"
    )
    @classmethod
    def _validate_group_max(cls, v):
        if isinstance(v, int) and v < 0:
            raise ValueError("technique count must be >= 0")
        return v

    @field_validator("selection_constraints", mode="before")
    @classmethod
    def _normalize_constraints(cls, v):
        return _as_list(v)


class TechniqueSelectionConfig(BaseModel):
    """`design_options.technique_selection` (2026-09-09 user request --
    replaces the old flat `technique_selection_mode` + `techniques` pair).

    - `mode: "manual"` (default): use `manual_techniques` exactly as given;
      `auto_config` is completely ignored, even as a fallback ("Không quan
      tâm auto_config khi mode là manual" -- user's own words).
    - `mode: "auto"`: `manual_techniques` is completely ignored (never used
      as a fallback either); `app/core/technique_selector.py` selects
      techniques per version via LLM, constrained by `auto_config`.

    See `app/core/pipeline.py::_node_load_request()` for where this gets
    resolved into the internal, derived `DesignOptions.techniques` list that
    the rest of the pipeline (prompt_builder, technique_selector) already
    knows how to consume -- kept deliberately unchanged downstream so this
    schema change stays isolated to request loading + the selector itself.
    """

    model_config = ConfigDict(extra="ignore")

    mode: Literal["manual", "auto"] = "manual"
    manual_techniques: list[str] = Field(default_factory=list)
    # 2026-09-09 (user request): only consulted when `mode == "auto"` --
    # scopes which BRAND(s)' techniques the auto selector is even allowed to
    # see/choose from.
    #
    # 2026-09-14 (user request): each brand code named here maps to its OWN
    # catalog file, `config/design_techniques_<brand>.json` -- `"general"`
    # reads `config/design_techniques_general.json`, `"acb"` reads
    # `config/design_techniques_acb.json`, and so on (replaces the old fixed
    # pair `design_techniques.json` (general only) +
    # `design_techniques_ref.json` (every other brand merged together);
    # adding a brand is now just dropping in a new file, not editing a
    # shared ref file). `app/core/design_techniques.py::load_catalog()`
    # loads and merges exactly the brand files requested across a batch;
    # `app/core/technique_selector.py` then (a) shows the LLM only the
    # subset of that merged catalog whose `brand` is in this list, and (b)
    # rejects any selected technique whose brand isn't in this list, even if
    # it slipped in from a wider batch-level catalog. Defaults to
    # `["general"]` so an existing request.json that never sets this field
    # keeps behaving exactly as before (general-only techniques visible).
    observe_brand: list[str] = Field(default_factory=lambda: ["general"])
    auto_config: TechniqueAutoConfig = Field(default_factory=TechniqueAutoConfig)

    @field_validator("manual_techniques", mode="before")
    @classmethod
    def _normalize_manual(cls, v):
        return _as_list(v)

    @field_validator("observe_brand", mode="before")
    @classmethod
    def _normalize_observe_brand(cls, v):
        normalized = _as_list(v)
        return normalized or ["general"]


class DesignOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate a stray legacy `protect_area` -- ignored, never used

    generate_person: bool = False
    has_logo: bool = False
    # 2026-08-27: opt-in per-request flag for the 3 new promo-banner use
    # cases (AI_DESIGN_REQUEST_INPUT_3usecase.xlsx) -- when true, the image
    # model is told to leave the text/CTA area empty and
    # `app/core/text_compositor.py` burns in the real headline/message/CTA
    # from JSON after crop, instead of the model drawing it. Defaults to
    # `False` so every existing request (AI-drawn text) is unaffected.
    text_by_code: bool = False
    # 2026-08-27 (4) (user request): the Excel's "TEXT luôn ghép bằng code"
    # clause turned out to be a mistake on the requester's side and was
    # removed from the source sheet -- `text_by_code` now defaults to
    # `False` for these use cases too (AI draws the text again, like
    # version_1). This flag is a SEPARATE, narrower opt-in: independent of
    # `text_by_code`, it still asks `app/core/content_writer.py` to draft
    # any headline/message/cta_label left as an empty string via the LLM
    # (the Esaving use case's "AI tự sáng tạo" cells) -- the drafted text
    # then flows into the normal AI-drawn-text prompt path instead of
    # being code-rendered. Defaults `False`: an empty text field elsewhere
    # in this project legitimately means "no text wanted", not "invent one".
    auto_generate_text: bool = False
    # 2026-09-04 (user request): SOLE gate for calling the LLM to draft copy.
    # When True, `app/core/text_gen.py::generate_creative_texts()` always
    # asks the LLM for a COMPLETE headline/message/cta set for every one of
    # `num_version` versions (never a partial/field-restricted request).
    # When False, the LLM is never called for this feature at all -- the 3
    # `generate_text_<field>` flags below have no effect either, since there
    # is no generated content for them to select.
    generate_text: bool = False
    # Field-level USAGE selector, evaluated only when `generate_text=True`.
    # Each flag independently decides whether that field's FINAL value in
    # the design comes from the LLM-generated copy (flag True) or from the
    # static `text.<field>` in request.json (flag False -- used as-is if
    # non-empty, otherwise that field is simply not rendered). This allows
    # mixing sources per field, e.g. a fixed approved headline alongside an
    # LLM-generated message. There is no "use all 3 automatically" shortcut
    # tied to `generate_text` alone -- each field's flag must be explicitly
    # True for that field's LLM copy to be used.
    generate_text_headline: bool = False
    generate_text_message: bool = False
    generate_text_cta_label: bool = False
    # Optional upper bound for purely decorative supporting items. None
    # intentionally preserves the current design behavior; supplied assets,
    # text, background structure and essential visual concepts do not count.
    num_decorative_elements: int | None = Field(default=None, ge=0)
    # 2026-08-27 (user request): when there's no `theme` bucket asset, the
    # background falls back to a plain dominant-color gradient (see
    # `app/core/prompt_builder.py::_theme_instruction()`) -- this lets each
    # request pick ITS OWN dominant color (e.g. a hex like "#213887" or a
    # brand color name) instead of always the one fixed OCB default gradient
    # from `banner_general_standard`. Empty string (default) keeps that
    # existing OCB-standard-gradient fallback unchanged.
    dominant_color: str = ""
    dominant_color_tolerance_percent: int | None = Field(default=None, ge=0, le=100)
    # 2026-09-09 (user request): replaces the old flat
    # `technique_selection_mode` + `techniques` pair with one nested object --
    # see `TechniqueSelectionConfig` above for the manual/auto contract.
    technique_selection: TechniqueSelectionConfig = Field(default_factory=TechniqueSelectionConfig)
    # INTERNAL/DERIVED -- not part of the documented request.json schema
    # anymore (do not set this directly in request.json; it is ignored on
    # load). `app/core/pipeline.py::_node_load_request()` populates this from
    # `technique_selection.manual_techniques` (manual mode) or leaves it
    # empty for later auto-selection write-back (auto mode) -- every
    # downstream consumer (`app/core/prompt_builder.py`,
    # `app/core/technique_selector.py`) still reads this ONE field uniformly,
    # regardless of which mode produced it, so that plumbing is intentionally
    # unchanged by this schema migration.
    techniques: list[str] = Field(default_factory=list)


class CanvasSpec(BaseModel):
    """`generation.canvas` -- the size the Image GenAI model is asked to
    generate at."""

    width: int
    height: int

    def wh(self) -> tuple[int, int]:
        return (self.width, self.height)


class SafeAreaSpec(BaseModel):
    """`generation.safe_area` -- the box (in canvas pixel coordinates) that
    all important design content must stay fully inside."""

    width: int
    height: int
    position: str = "center"
    x: int = 0
    y: int = 0

    def wh(self) -> tuple[int, int]:
        return (self.width, self.height)

    def box(self) -> tuple[int, int, int, int]:
        """`(x, y, width, height)` offset box inside the canvas."""
        return (self.x, self.y, self.width, self.height)


class GenerationSpec(BaseModel):
    canvas: CanvasSpec
    safe_area: SafeAreaSpec
    instruction: str = ""


class CropAreaSpec(BaseModel):
    x: int
    y: int
    width: int
    height: int

    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class CropSpec(BaseModel):
    source: str = "generation.canvas"
    area: CropAreaSpec | None = None


class OutputSizeSpec(BaseModel):
    width: int
    height: int

    def wh(self) -> tuple[int, int]:
        return (self.width, self.height)


class OutputSpec(BaseModel):
    size: OutputSizeSpec
    crop: CropSpec | None = None
    clone: list[str] = Field(default_factory=list)


class AudienceConfig(BaseModel):
    """Optional WHO-layer on top of MASS/MASS_AFF/AFF `format_type`.

    2026-09-29 (user request, RB chi tiêu thẻ v4.1 slide 4): keep the 3 bank
    segments as the visual system; inside a segment the caller may pick ONE
    of 8 card-spend personas. Empty/omitted = current behavior (no persona).
    Unknown ids are ignored at resolve time (`app/core/personas.py`), not
    rejected here, so a mistype never blocks generate.
    """

    model_config = ConfigDict(extra="ignore")

    persona: str | None = None

    @field_validator("persona", mode="before")
    @classmethod
    def _blank_persona_to_none(cls, v):
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned or None


class CreativeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate stray/legacy top-level fields, never crash on them

    creative_id: str
    format_type: str
    design_brief: str = ""
    buckets: Buckets = Field(default_factory=Buckets)
    reference: str = ""
    text: TextContent = Field(default_factory=TextContent)
    guidelines: str = ""
    num_version: int = 1
    design_options: DesignOptions = Field(default_factory=DesignOptions)
    audience: AudienceConfig = Field(default_factory=AudienceConfig)
    generation: GenerationSpec
    output: OutputSpec

    # ---- derived geometry -------------------------------------------------
    def canvas_wh(self) -> tuple[int, int]:
        """Size the image model actually generates at (`generation.canvas`)."""
        return self.generation.canvas.wh()

    def safe_area_wh(self) -> tuple[int, int]:
        """`generation.safe_area` size -- where all important content must fit."""
        return self.generation.safe_area.wh()

    def crop_box(self) -> tuple[int, int, int, int]:
        """`(x, y, width, height)` box to crop out of the generated canvas.
        Prefers `output.crop.area` when the request.json declares one;
        otherwise falls back to `generation.safe_area`'s own box (REQUEST.md
        [1]: "output.size chính là kích thước trong safe_area" -- both
        describe the identical crop box in every real file seen so far)."""
        if self.output.crop is not None and self.output.crop.area is not None:
            return self.output.crop.area.box()
        return self.generation.safe_area.box()

    def final_output_wh(self) -> tuple[int, int]:
        """Real pixel size of the PNG written to disk (`output.size`)."""
        return self.output.size.wh()

    def has_crop(self) -> bool:
        """False when the generation canvas IS the final output (no
        cropping needed -- e.g. a square banner with no background bleed)."""
        return self.canvas_wh() != self.final_output_wh()

    def display_size_slug(self) -> str:
        return wh_slug(*self.final_output_wh())


class ResolvedAsset(BaseModel):
    """1 resolved bucket/reference image -- used by `GET /requests/{name}`
    to let the webapp preview every attached asset before Run."""

    label: Literal["theme", "context", "product", "brand_text", "reference"]
    value: str
    index: int = 1
    count: int = 1
    ok: bool = False
    path: str = ""
    error: str | None = None
    # "text": `value` is a free-text description (no image attached, no
    # file resolution attempted) -- see `Buckets`' docstring note above and
    # `app/core/asset_resolver.py::looks_like_image_path()`. Defaults to
    # "image" so the `reference` entry (still image-only) and any old
    # serialized data without this field keep their previous meaning.
    kind: Literal["image", "text"] = "image"


class GenerateRequestBody(BaseModel):
    request_name: str
    mode: Literal["default", "quality", "eco", "custom", "all"] = "default"
    custom_models: list[str] = Field(default_factory=list)
    # 2026-09-22 (user request): lets a caller ask for fewer versions than
    # request.json's own `num_version` -- used by script/run_db_usecase.ps1's
    # top-up retry loop to re-request only the versions still missing after a
    # partial failure, instead of regenerating all 3 from scratch.
    num_version_override: int | None = None
    # Lets the webapp toggle "has a person in the image?" per-run without
    # editing request.json's own `design_options.generate_person` -- same
    # override-not-persist pattern as `num_version_override` above.
    generate_person_override: bool | None = None
    # Optional per-run persona id from `lib/personas/personas.json`. None =
    # keep request.json `audience.persona`; "" = clear persona for this run
    # only; a catalog id = use that persona without writing the JSON.
    persona_override: str | None = None


class CloneRequestBody(BaseModel):
    request_name: str
    source_image_path: str
    select_model: str


class ReleaseRequestBody(BaseModel):
    request_name: str
    selected_output_paths: list[str]


class CloneFromReleaseBody(BaseModel):
    request_name: str
    released_image_path: str
    display_sizes: list[str]
    image_model: str


class GenerateResult(BaseModel):
    ok: bool
    image_model: str
    version: int = 1
    display_size: str = ""
    path: str | None = None
    filename: str | None = None
    error: str | None = None
