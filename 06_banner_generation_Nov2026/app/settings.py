"""Centralized app configuration -- the ONE source of config for every
module under `app/` (per the `nhantn-mlai-project-structure` standard: no
module reads `os.environ` or hard-codes a tunable value directly, everything
goes through `get_settings()`).

Values come from 2 sources, in this priority order: a real OS/shell
environment variable (highest, pydantic-settings always checks this first),
then whichever `.env` file is found (`version_1/.env` if present, else the
shared parent `dco_dynamic_creative_opt_v2/.env`), then the default declared
on the field below (lowest).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# version_1/app/settings.py -> parents[1] == version_1/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Most specific `.env` first (`version_1/.env`, only if the user ever adds
# one), falling back to the shared parent `.env`.
_ENV_FILES = tuple(
    str(p) for p in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env") if p.exists()
) or (str(PROJECT_ROOT.parent / ".env"),)


def _resolve_dir(raw: Path, fallback: Path) -> Path:
    """A configured directory that doesn't exist ON THIS MACHINE (e.g. a
    Windows path left over in `.env` while running on Linux, or vice versa)
    is treated as unusable and silently replaced by a project-relative
    default instead of crashing later when something does `Path(...).glob()`
    on a path that was never real to begin with.

    2026-08-20 (self-discovered while investigating a user-reported
    ConnectionResetError -- see 2-project-technote.md's run instructions
    for the actual root cause of that bug, an unrelated uvicorn --reload
    issue): a RELATIVE default like `Path("./lib")` used to be checked with
    `raw.exists()` directly, which resolves against the process's CURRENT
    WORKING DIRECTORY at the moment uvicorn/streamlit/pytest happened to be
    launched from -- NOT necessarily `version_1/`. A real run's own log
    showed `asset_library_dir` resolving to a `lib/` folder inside a
    DIFFERENT, unrelated parent project directory, because the server had
    been started from that directory (which happened to also contain a
    `lib/` subfolder) instead of `version_1/`. Every other field in this
    module is a plain relative default (`./lib`, `./input`, `./output`,
    ...), so this silently wrong resolution could recur for ANY of them
    depending on where the process is launched from. Fixed by anchoring a
    relative `raw` at `PROJECT_ROOT` before checking existence, so the
    resolved path no longer depends on the process's cwd at all -- only an
    ABSOLUTE path explicitly set in `.env`/env can still point somewhere
    else, which remains correct (an absolute override should be honored
    regardless of cwd)."""
    try:
        if str(raw).strip():
            candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
            if candidate.exists():
                return candidate.resolve()
    except OSError:
        pass
    return fallback.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "dco-dynamic-creative-opt"
    environment: str = Field(default="development")  # development | staging | production
    log_level: str = "INFO"

    # --- Paths (validators below resolve each to a real, existing Path --
    # falling back to a project-relative default if the configured value
    # doesn't exist on this machine, see `_resolve_dir`) ---
    asset_library_dir: Path = Path("./lib")
    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")
    script_output_dir: Path = Path("./script/output")
    prompt_rules_file: Path = Path("./prompts/design_rules.md")

    # 2026-08-20 (user request): a SHARED, request-independent cache of
    # SAFE AREA GUIDE images (app/core/safe_area_guide.py) -- 1 image per
    # DISTINCT (canvas_size, safe_area_size) combination, reused across
    # every request/run that happens to need that same geometry, instead of
    # each request generating (and paying the render cost for) its own copy
    # under output/<request_name>/_resized_refs/. Override with
    # `SAFE_AREA_GUIDE_DIR` in `.env` (this project's other directory
    # settings all use a `_DIR` env-var suffix, e.g. `ASSET_LIBRARY_DIR`,
    # `INPUT_DIR` -- kept consistent here rather than the literal
    # `SAFE_AREA_GUIDE_PATH` name from the request, since pydantic-settings
    # derives the env var name from the Python field name).
    safe_area_guide_dir: Path = Path("./lib/safe_area_guide")

    @field_validator("asset_library_dir", mode="after")
    @classmethod
    def _v_asset_library_dir(cls, v: Path) -> Path:
        return _resolve_dir(v, PROJECT_ROOT / "lib")

    @field_validator("safe_area_guide_dir", mode="after")
    @classmethod
    def _v_safe_area_guide_dir(cls, v: Path) -> Path:
        return _resolve_dir(v, PROJECT_ROOT / "lib" / "safe_area_guide")

    @field_validator("input_dir", mode="after")
    @classmethod
    def _v_input_dir(cls, v: Path) -> Path:
        return _resolve_dir(v, PROJECT_ROOT / "input")

    @field_validator("output_dir", mode="after")
    @classmethod
    def _v_output_dir(cls, v: Path) -> Path:
        return _resolve_dir(v, PROJECT_ROOT / "output")

    @field_validator("script_output_dir", mode="after")
    @classmethod
    def _v_script_output_dir(cls, v: Path) -> Path:
        return _resolve_dir(v, PROJECT_ROOT / "script" / "output")

    @field_validator("prompt_rules_file", mode="after")
    @classmethod
    def _v_prompt_rules_file(cls, v: Path) -> Path:
        # Same cwd-dependence bug as `_resolve_dir()` above -- fixed the
        # same way (anchor a relative default at PROJECT_ROOT before
        # checking existence, instead of the process's current directory).
        try:
            if str(v).strip():
                candidate = v if v.is_absolute() else (PROJECT_ROOT / v)
                if candidate.exists():
                    return candidate.resolve()
        except OSError:
            pass
        return (PROJECT_ROOT / "prompts" / "design_rules.md").resolve()

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_base_url: str = "http://localhost:8000"

    # --- LLM provider selection (app/llm/ -- prompt refinement, REQUEST.md:
    # "sử dụng GPT-5-MINI ... để viết lại prompt tổng cho cô đọng hơn").
    # Swappable via .env alone, no code change (nhantn-mlai-project-structure
    # BaseLLMProvider + LLMFactory pattern).
    llm_provider: str = Field(default="openai")  # openai | claude | llama | deepseek | ollama
    llm_model_name: str = Field(default="gpt-5-mini")
    llm_text_gen: str = Field(default="gpt-5-mini")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=2048)
    llm_base_url: str | None = Field(default=None)  # custom/self-host endpoint override
    use_llm_prompt_refinement: bool = True
    # Maximum number of creative versions planned by one Technique Selector
    # LLM call. Larger runs are processed sequentially in batches so each
    # later batch can see a compact summary of earlier selections.
    technique_selection_batch_size: int = Field(default=10, ge=1)
    technique_selection_cache_enabled: bool = Field(
        default=True,
        description="Cache technique-selection LLM results by content hash. "
        "Same request brief + same catalog → skip the ~30s LLM call on re-runs. "
        "Set to false to always re-select (e.g. during catalog tuning).",
    )

    # --- LLM provider credentials (override via .env; None/"" = not
    # configured -- that provider raises a clear error at call time rather
    # than silently falling back to a different one) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None
    llama_api_key: str | None = None  # hosted cloud llama (Groq/Together/Bedrock-style)
    ollama_base_url: str = "http://localhost:11434"

    # This project's real deployed LLM today is Azure AI Foundry's
    # gpt-5-mini, reachable at an Azure-specific URL/key rather than the
    # public OpenAI API -- `app/llm/providers/openai_provider.py` (the
    # "openai" entry in LLMFactory, since Azure OpenAI/AI-Foundry speaks the
    # OpenAI-compatible API) falls back to these whenever
    # `llm_base_url`/`openai_api_key` above are left unset, so the project
    # keeps working out of the box with the credentials already in `.env`.
    az_gpt5mini_endpoint: str = ""
    az_gpt5mini_key: str = ""
    az_gpt5mini_api_version: str = "2025-08-07"

    # --- Image GenAI model credentials (app/providers/ -- this app's own
    # domain-specific external-API layer, see 2-project-technote.md for why
    # this is separate from app/llm/) ---
    gcp_gemini_31_flash_image_endpoint: str = ""
    gcp_gemini_3_pro_image_endpoint: str = ""
    gcp_vertex_api_key: str = ""

    az_mai_image_25_endpoint: str = ""
    az_mai_image_25_key: str = ""
    az_mai_image_25_deployment: str = "mai-image-2.5"
    az_mai_image_25_api_version: str = "2026-06-02"

    az_mai_image_25_flash_endpoint: str = ""
    az_mai_image_25_flash_key: str = ""
    az_mai_image_25_flash_deployment: str = "mai-image-2.5-flash"
    az_mai_image_25_flash_api_version: str = "2026-06-02"

    az_mai_image_25_pro_endpoint: str = ""
    az_mai_image_25_pro_key: str = ""
    az_mai_image_25_pro_deployment: str = "mai-image-2.5-pro"

    az_flux_2_pro_endpoint: str = ""
    az_flux_2_pro_key: str = ""
    az_flux_2_pro_deployment: str = "flux-2-pro"
    az_flux_2_pro_api_version: str = "1"

    az_gpt_image_2_endpoint: str = ""
    az_gpt_image_2_key: str = ""

    http_timeout_seconds: int = 600
    http_max_retries: int = 3
    # Versions for one image model are independent after their prompts have
    # been materialized. Run a small number concurrently to reduce wall time
    # without overwhelming provider quotas. Set to 1 to restore sequential
    # generation; override with GENERATION_MAX_WORKERS in .env.
    generation_max_workers: int = Field(default=2, ge=1, le=16)

    # Image-attachment downsampling (REQUEST.md [1] step [4]).
    attach_image_max_dim: int = 1024
    attach_image_jpeg_quality: int = 85

    # `guidelines.json` / `model_usage.json` are always at a fixed path
    # inside the repo (not user-configurable), so these are computed
    # properties rather than `.env` fields.
    @property
    def guidelines_file(self) -> Path:
        return PROJECT_ROOT / "lib" / "guidelines" / "guidelines.json"

    @property
    def model_usage_file(self) -> Path:
        return PROJECT_ROOT / "config" / "model_usage.json"

    @property
    def design_techniques_dir(self) -> Path:
        """Directory holding one design-techniques catalog file PER BRAND
        (2026-09-14 user request -- replaces the old fixed pair
        `design_techniques.json` (general) + `design_techniques_ref.json`
        (every other brand merged together)). See
        `design_techniques_file_for_brand()`."""
        return PROJECT_ROOT / "config"

    def design_techniques_file_for_brand(self, brand: str) -> Path:
        """`config/design_techniques_<brand>.json` for one brand code (e.g.
        `"general"`, `"acb"`, `"ocb_db"`).

        2026-09-14 (user request): each brand named in a request's
        `design_options.technique_selection.observe_brand` now maps to its
        OWN catalog file instead of a shared `design_techniques.json` +
        `design_techniques_ref.json` pair -- `observe_brand: ["acb"]` reads
        `config/design_techniques_acb.json`, `["general"]` reads
        `config/design_techniques_general.json`, and so on. See
        `app/core/design_techniques.py::load_catalog()`, which is the only
        caller and is responsible for sanitizing `brand` (untrusted, comes
        from request.json) before it ever reaches this method -- this
        method itself does no validation, so it must never be called
        directly with a raw, unsanitized brand string."""
        return self.design_techniques_dir / f"design_techniques_{brand}.json"

    @property
    def model_profiles_file(self) -> Path:
        """Consolidated model catalog: supported aspect ratios AND
        per-model input/output token pricing (2026-09-06 user request --
        replaces `config/model_ratio.json`, which no code actually read as
        JSON; see `app/core/model_profiles.py` module docstring). Used by
        `app/core/usage_tracker.py` to cost every GenAI call made during a
        run."""
        return PROJECT_ROOT / "config" / "model_profiles.json"

    @property
    def usage_log_db_file(self) -> Path:
        """SQLite usage/cost-tracking database (2026-09-06 (3), user
        follow-up request): lives in its own `usage/` directory, a SIBLING
        of `app/` directly at the project root -- deliberately NOT nested
        under `output_dir` (a user-configurable field that could point
        anywhere via `.env`), so this accounting database always sits at
        one fixed, predictable location. Not user-configurable, same as
        `model_usage_file`/`guidelines_file` above. See
        `app/core/usage_tracker.py` / `app/core/output_manager.usage_db_file()`."""
        return PROJECT_ROOT / "usage" / "usage_log.db"


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor -- import and call this, never instantiate
    `Settings()` directly elsewhere (keeps `.env` parsed exactly once, and
    keeps unit tests able to override config by monkeypatching this one
    function instead of environment variables)."""
    return Settings()


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in ra toàn bộ config đã resolve, hữu ích để kiểm tra
# .env có được đọc đúng không trước khi chạy cả app):
#   python -m app.settings
#
# Yêu cầu trước khi chạy: có 1 trong 2 file `.env` tồn tại
# (`version_1/.env` hoặc `dco_dynamic_creative_opt_v2/.env`, xem _ENV_FILES).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    s = get_settings()
    for name in sorted(s.model_fields):
        value = getattr(s, name)
        if "key" in name and value:
            value = f"{str(value)[:4]}...(hidden)"
        print(f"{name} = {value!r}")
    print(f"guidelines_file = {s.guidelines_file!r} (exists={s.guidelines_file.exists()})")
    print(f"model_usage_file = {s.model_usage_file!r} (exists={s.model_usage_file.exists()})")
    print(f"design_techniques_dir = {s.design_techniques_dir!r} (exists={s.design_techniques_dir.exists()})")
    print(
        f"design_techniques_file_for_brand('general') = {s.design_techniques_file_for_brand('general')!r} "
        f"(exists={s.design_techniques_file_for_brand('general').exists()})"
    )
    print(f"model_profiles_file = {s.model_profiles_file!r} (exists={s.model_profiles_file.exists()})")
    print(f"usage_log_db_file = {s.usage_log_db_file!r} (exists={s.usage_log_db_file.exists()})")
