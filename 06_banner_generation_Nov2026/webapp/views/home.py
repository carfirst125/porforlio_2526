"""Home view -- loaded by the router (`webapp/Home.py`) via `st.navigation()`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from webapp.utils import api_client  # noqa: E402
from webapp.utils.theme import card, page_title_block  # noqa: E402

# UPDATE 2026-08-16 (theo yêu cầu user): bỏ logo OCB khỏi phần đầu trang bên
# phải, đưa title về sát biên trái -- cùng cách đã làm ở Generate/Release
# (`page_title_block(..., show_logo=False)`, xem `webapp/utils/theme.py`).
page_title_block(
    "OCB — Dynamic Creative Optimization",
    "Demo app generating ad banners with Image GenAI",
    show_logo=False,
)

with card():
    st.markdown(
        """
This is a demo webapp for OCB's **DCO (Dynamic Creative Optimization)** system --
automatically generating ad banners from a *design brief*, *brand guideline*, and
assets (theme, character, product) using multiple Image GenAI models.

1. Go to the **Generate** page -- pick a `request.json`, choose a run mode (model), click Run, and view the results.
2. Check the images you like to **release** them.
3. Go to the **Release** page -- review released versions, and clone them into other sizes.
"""
    )

# ---------------------------------------------------------------------------
# UPDATE 2026-08-16 (bug #36, theo yêu cầu user): "Available models" đổi từ
# danh sách text (dù đã bo khung/xếp cột ở bug #29/#30) sang 1 BẢNG thật với
# các cột Image Model / Provider (hãng sản xuất) / Hosted in (Azure hay GCP)
# / Benchmark / Overall score. 3 cột đầu lấy từ REQUEST.md (mô hình sinh ảnh
# + nơi host, xem đầu file REQUEST.md) -- cố định, không đổi theo thời gian.
# 2 cột benchmark/overall score CẦN dữ liệu thật (không tự bịa số) -- đã
# WebSearch + WebFetch nguồn công khai (Artificial Analysis, đơn vị chấm
# điểm Elo qua blind vote phổ biến nhất cho image-gen model) NGAY TRƯỚC khi
# viết bảng này, snapshot lúc 2026-08-16. `mai-image-2.5-pro` KHÔNG xuất hiện
# trên leaderboard Text-to-Image công khai (đã kiểm tra riêng, trang model
# card + trang leaderboard đầy đủ 146 model đều không có "MAI-Image-2.5-Pro"
# -- chỉ có "MAI-Image-2.5" thường) -- để "N/A" thay vì đoán, kèm ghi chú rõ
# trong bảng. Elo là con số tuyệt đối (không có "cận trên" cố định), nên
# "Overall score" không chuẩn hoá 0-100 riêng (tránh tạo cảm giác 1 chỉ số
# "chính thức" không có thật) -- dùng luôn Elo + rank trong Arena làm overall
# score, đúng tinh thần "điểm tổng" mà cộng đồng dùng để so các model ảnh
# hiện nay.
# ---------------------------------------------------------------------------
_MODEL_INFO = {
    "gemini-3.1-flash-image": {
        "provider": "Google",
        "hosted_in": "GCP",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1327,
        "rank": 3,
        "sort_rank": 3,
        "score_display": "1,327 Elo (Arena rank #3)",
    },
    "gemini-3-pro-image": {
        "provider": "Google",
        "hosted_in": "GCP",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1303,
        "rank": 6,
        "sort_rank": 6,
        "score_display": "1,303 Elo (Arena rank #6)",
    },
    "gpt-image-2": {
        "provider": "OpenAI",
        "hosted_in": "Azure",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1376,
        "rank": 1,
        "sort_rank": 1,
        "score_display": "1,376 Elo (Arena rank #1)",
    },
    "mai-image-2.5": {
        "provider": "Microsoft AI",
        "hosted_in": "Azure",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1312,
        "rank": 5,
        "sort_rank": 5,
        "score_display": "1,312 Elo (Arena rank #5)",
    },
    "mai-image-2.5-flash": {
        "provider": "Microsoft AI",
        "hosted_in": "Azure",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1235,
        "rank": 12,
        "sort_rank": 12,
        "score_display": "1,235 Elo (Arena rank #12)",
    },
    "mai-image-2.5-pro": {
        "provider": "Microsoft AI",
        "hosted_in": "Azure",
        # UPDATE 2026-08-16 (bug #38, theo user cung cấp trực tiếp): user báo
        # mai-image-2.5-pro xếp hạng CAO HƠN mai-image-2.5, muốn đưa lên vị
        # trí #3 (ngay trên mai-image-2.5). Bug #36/#37 đã tìm kỹ (cả 2
        # leaderboard Text-to-Image + Image-Editing của Artificial Analysis,
        # trang model riêng) và không thấy Elo/rank độc lập công khai cho
        # riêng "-Pro" -- vẫn đúng như vậy, KHÔNG có nguồn benchmark công
        # khai mới nào xuất hiện. Not fabricating a fake Elo number gán cho
        # Artificial Analysis (nguồn đó không thực sự có số này). Thay vào
        # đó: dùng `sort_rank` RIÊNG (không phải Elo) để xếp đúng vị trí user
        # yêu cầu -- chèn giữa gemini-3.1-flash-image (Arena rank 3) và
        # mai-image-2.5 (Arena rank 5), tức sort_rank=4 -- và "Overall score"
        # ghi RÕ đây là theo thông tin user cung cấp, không phải Elo đo được
        # từ Arena, để bảng không đánh lừa là 2 loại dữ liệu giống nhau.
        "benchmark": "Not on public Arena leaderboards -- position below is per user-provided info",
        "elo": None,
        "rank": None,
        "sort_rank": 4,
        "score_display": "Above MAI-Image-2.5 (per user input, no independent public score)",
    },
    "flux.2-pro": {
        "provider": "Black Forest Labs",
        "hosted_in": "Azure",
        "benchmark": "Artificial Analysis -- Text-to-Image Arena (Elo)",
        "elo": 1213,
        "rank": 26,
        "sort_rank": 26,
        "score_display": "1,213 Elo (Arena rank #26)",
    },
}

st.subheader("System status")

with card():
    st.markdown("**Backend FastAPI**")
    h = api_client.health()
    if h.get("ok"):
        st.success(f"Running OK -- {h.get('app')} ({h.get('environment')})")
    else:
        st.error(f"Could not connect: {h.get('error')}")
        st.caption("Run `uvicorn app.main:app --reload` before using the webapp.")

with card():
    st.markdown("**Available image models**")
    m = api_client.list_models()
    providers = m.get("available_providers", [])
    if providers:
        st.success(f"{len(providers)} models registered")

        # UPDATE 2026-08-16 (bug #38): sắp xếp theo `sort_rank` (thứ hạng
        # thấp = tốt hơn, cùng chiều với Arena rank thật) thay vì trực tiếp
        # theo Elo -- cần tách riêng vì `mai-image-2.5-pro` giờ có VỊ TRÍ xếp
        # hạng (do user cung cấp) nhưng KHÔNG có Elo thật. Model nào thiếu cả
        # `sort_rank` lẫn Elo (trường hợp tương lai, model mới hoàn toàn
        # không rõ nguồn nào) mới rơi xuống cuối bảng.
        rows_with_meta = [(p, _MODEL_INFO.get(p, {})) for p in providers]
        rows_with_meta.sort(key=lambda item: (item[1].get("sort_rank") is None, item[1].get("sort_rank") or 0))

        table_rows = []
        for p, info in rows_with_meta:
            table_rows.append(
                {
                    "Image Model": p,
                    "Provider": info.get("provider", "Unknown"),
                    "Hosted in": info.get("hosted_in", "Unknown"),
                    "Benchmark": info.get("benchmark", "N/A -- not yet benchmarked"),
                    "Overall score": info.get("score_display", "N/A"),
                }
            )

        st.dataframe(table_rows, hide_index=True, use_container_width=True)
        st.caption(
            "Sorted by Overall score, highest first. Benchmark source for the Elo-scored "
            "rows: Artificial Analysis Text-to-Image Arena (blind human preference votes) "
            "-- snapshot taken 2026-08-16, scores shift over time as the Arena collects "
            "more votes. `mai-image-2.5-pro` is not on Artificial Analysis's public "
            "Text-to-Image or Image-Editing leaderboards as of this snapshot, so it has no "
            "independently-measured Elo -- its position here (above MAI-Image-2.5) reflects "
            "ranking info provided directly by the app owner, not a public benchmark score."
        )
    else:
        st.warning("Could not fetch the model list -- check the backend connection.")
