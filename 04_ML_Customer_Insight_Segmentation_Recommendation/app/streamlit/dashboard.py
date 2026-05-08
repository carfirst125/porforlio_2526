from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TX_PATH = PROJECT_ROOT / "dataset" / "transaction_data.csv"
INSIGHTS_PATH = PROJECT_ROOT / "app" / "artifacts" / "customer_insights.parquet"
EVAL_DIR = PROJECT_ROOT / "app" / "artifacts" / "evaluation"
EVAL_LATEST_PATH = EVAL_DIR / "evaluation_report.json"
EVAL_HISTORY_DIR = EVAL_DIR / "history"


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(TX_PATH, encoding="ISO-8859-1", low_memory=False)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    return df


@st.cache_data(show_spinner=False)
def load_insights() -> pd.DataFrame:
    if not INSIGHTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(INSIGHTS_PATH)


def _gradient_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
    text_col: str | None = None,
    text: str | None = None,
) -> go.Figure:
    # For horizontal bars, numeric value is on x; for vertical bars, on y.
    default_text_field = x if orientation == "h" else y
    effective_text = text_col if text_col is not None else (text if text is not None else default_text_field)
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=y,
        color_continuous_scale="Blues",
        orientation=orientation,
        title=title,
        text=effective_text if effective_text else y,
    )
    fig.update_coloraxes(showscale=False)
    if orientation == "h":
        # Horizontal bars with long labels often clip outside text;
        # "auto" keeps values visible on/near bars.
        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="auto",
            cliponaxis=False,
            showlegend=False,
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=80, t=60, b=20))
    else:
        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
        )
        fig.update_layout(showlegend=False)
    return fig


def _extract_numeric_value(value: object) -> float | None:
    """Extract numeric value from raw values or mixed strings like 'revenue=1234'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    matches = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _segment_three_groups(insights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = insights.copy()
    clv_adjusted = out["clv_estimate"].astype(float) * (1.0 - out["churn_risk_score"].astype(float) / 100.0).clip(
        lower=0.05
    )
    clv_scaled = pd.Series(ExtendedMinMax.scale(clv_adjusted), index=out.index)

    score = (
        0.30 * out["rfm_composite_score"].astype(float)
        + 0.30 * out["transactional_promoter_score"].astype(float)
        + 0.20 * (100.0 - out["churn_risk_score"].astype(float))
        + 0.20 * (clv_scaled * 100.0)
    )
    out["segment_score"] = score
    out["clv_adjusted"] = clv_adjusted
    out["customer_group"] = pd.qcut(
        out["segment_score"].rank(method="first"),
        q=3,
        labels=["At-Risk / Win-Back", "Growth / Nurture", "VIP / Loyal"],
    )

    group_summary = (
        out.groupby("customer_group", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            AvgMonetary=("Monetary", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgRecencyDays=("RecencyDays", "mean"),
            AvgChurnRisk=("churn_risk_score", "mean"),
            AvgCLV=("clv_estimate", "mean"),
            MedCLV=("clv_estimate", "median"),
            AvgAdjCLV=("clv_adjusted", "mean"),
            MedAdjCLV=("clv_adjusted", "median"),
        )
        .sort_values("Customers", ascending=False)
    )
    desc = {
        "VIP / Loyal": "High loyalty score and strongest churn-adjusted value; prioritize loyalty rewards and premium upsell.",
        "Growth / Nurture": "Healthy base with room to grow; focus on cross-sell and basket expansion campaigns.",
        "At-Risk / Win-Back": "Higher churn signals and weaker adjusted future value; run win-back and reactivation journeys.",
    }
    return group_summary, desc


class ExtendedMinMax:
    @staticmethod
    def scale(series: pd.Series) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
        lo = float(x.min())
        hi = float(x.max())
        if hi - lo < 1e-12:
            return pd.Series(0.0, index=x.index)
        return (x - lo) / (hi - lo)


def build_overview_charts(df: pd.DataFrame, insights: pd.DataFrame) -> None:
    st.subheader("Part 1: Data Analysis - Customer Understanding through Transaction Data")

    clean = df.dropna(subset=["CustomerID"]).copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int)
    positive = clean.loc[(~clean["IsCancellation"]) & (clean["LineTotal"] > 0)].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(clean):,}")
    col2.metric("Unique Customers", f"{clean['CustomerID'].nunique():,}")
    col3.metric("Net Revenue", f"{clean['LineTotal'].sum():,.0f}")
    col4.metric("Cancellation Rate", f"{clean['IsCancellation'].mean() * 100:.2f}%")

    # MTD comparison: last month to current day-of-month vs previous month same day-of-month
    latest_date = positive["InvoiceDate"].max().normalize()
    last_month_start = latest_date.replace(day=1)
    prev_month_start = (last_month_start - pd.offsets.MonthBegin(1)).normalize()
    day_cutoff = int(latest_date.day)
    last_month_label = latest_date.strftime("%Y-%m")
    prev_month_label = prev_month_start.strftime("%Y-%m")

    mtd_last = positive.loc[
        (positive["InvoiceDate"] >= last_month_start)
        & (positive["InvoiceDate"].dt.day <= day_cutoff),
        "LineTotal",
    ].sum()
    mtd_prev = positive.loc[
        (positive["InvoiceDate"] >= prev_month_start)
        & (positive["InvoiceDate"] < last_month_start)
        & (positive["InvoiceDate"].dt.day <= day_cutoff),
        "LineTotal",
    ].sum()
    mtd_growth = ((mtd_last - mtd_prev) / mtd_prev * 100.0) if mtd_prev > 0 else 0.0

    st.markdown("#### MTD Performance (same day-of-month cutoff)")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"MTD Revenue {last_month_label}", f"{mtd_last:,.0f}")
    m2.metric(f"MTD Revenue {prev_month_label}", f"{mtd_prev:,.0f}")
    m3.metric("MTD Growth %", f"{mtd_growth:+.2f}%")

    mtd_compare = pd.DataFrame(
        {
            "Period": [f"{prev_month_label} MTD", f"{last_month_label} MTD"],
            "Revenue": [mtd_prev, mtd_last],
        }
    )
    fig_mtd_bar = _gradient_bar(
        mtd_compare,
        x="Period",
        y="Revenue",
        title="MTD Revenue Comparison: Last Month vs Previous Month (same day cutoff)",
    )
    st.plotly_chart(fig_mtd_bar, use_container_width=True)

    last_mtd_daily = (
        positive.loc[
            (positive["InvoiceDate"] >= last_month_start)
            & (positive["InvoiceDate"].dt.day <= day_cutoff)
        ]
        .assign(Day=lambda d: d["InvoiceDate"].dt.day)
        .groupby("Day", as_index=False)["LineTotal"]
        .sum()
        .rename(columns={"LineTotal": f"{last_month_label} MTD"})
    )
    prev_mtd_daily = (
        positive.loc[
            (positive["InvoiceDate"] >= prev_month_start)
            & (positive["InvoiceDate"] < last_month_start)
            & (positive["InvoiceDate"].dt.day <= day_cutoff)
        ]
        .assign(Day=lambda d: d["InvoiceDate"].dt.day)
        .groupby("Day", as_index=False)["LineTotal"]
        .sum()
        .rename(columns={"LineTotal": f"{prev_month_label} MTD"})
    )
    mtd_daily = pd.merge(last_mtd_daily, prev_mtd_daily, on="Day", how="outer").fillna(0.0).sort_values("Day")
    mtd_daily = mtd_daily.melt(id_vars="Day", var_name="Series", value_name="Revenue")
    fig_mtd_line = px.line(
        mtd_daily,
        x="Day",
        y="Revenue",
        color="Series",
        markers=True,
        title="MTD Daily Revenue Trajectory Comparison",
        text="Revenue",
    )
    fig_mtd_line.update_traces(texttemplate="%{text:,.0f}", textposition="top center")
    st.plotly_chart(fig_mtd_line, use_container_width=True)

    monthly = (
        positive.assign(Month=positive["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["LineTotal"]
        .sum()
    )
    fig_monthly = px.line(
        monthly,
        x="Month",
        y="LineTotal",
        markers=True,
        title="Monthly Revenue Trend (Non-cancelled Transactions)",
        text="LineTotal",
    )
    fig_monthly.update_traces(texttemplate="%{text:,.0f}", textposition="top center")
    st.plotly_chart(fig_monthly, use_container_width=True)

    country = (
        positive.groupby("Country", as_index=False)["LineTotal"]
        .sum()
        .sort_values("LineTotal", ascending=False)
        .head(12)
    )
    fig_country = _gradient_bar(
        country,
        x="Country",
        y="LineTotal",
        title="Top Countries by Revenue",
    )
    st.plotly_chart(fig_country, use_container_width=True)

    product_stats = (
        positive.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            Revenue=("LineTotal", "sum"),
            PurchaseQuantity=("Quantity", "sum"),
        )
    )

    top_products = (
        product_stats.sort_values("Revenue", ascending=False)
        .head(15)
    )
    top_products["Revenue"] = top_products["Revenue"].apply(_extract_numeric_value)
    top_products = top_products.dropna(subset=["Revenue"]).copy()
    top_products["Revenue"] = top_products["Revenue"].astype(float)
    top_products = top_products.sort_values("Revenue", ascending=False).head(15)
    top_products["Description"] = top_products["Description"].fillna("Unknown Description")
    top_products["Label"] = top_products["StockCode"].astype(str) + " | " + top_products["Description"].astype(str)
    fig_products = _gradient_bar(
        top_products,
        x="Revenue",
        y="Label",
        orientation="h",
        title="Top 15 Purchased Products by Revenue",
    )
    fig_products.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_products, use_container_width=True)

    top_purchased = (
        product_stats
        .sort_values("PurchaseQuantity", ascending=False)
        .head(15)
    )
    top_purchased["PurchaseQuantity"] = top_purchased["PurchaseQuantity"].apply(_extract_numeric_value)
    top_purchased = top_purchased.dropna(subset=["PurchaseQuantity"]).copy()
    top_purchased["PurchaseQuantity"] = top_purchased["PurchaseQuantity"].astype(float)
    top_purchased = top_purchased.sort_values("PurchaseQuantity", ascending=False).head(15)
    top_purchased["Description"] = top_purchased["Description"].fillna("Unknown Description")
    top_purchased["Label"] = (
        top_purchased["StockCode"].astype(str) + " | " + top_purchased["Description"].astype(str)
    )
    fig_top_qty = _gradient_bar(
        top_purchased,
        x="PurchaseQuantity",
        y="Label",
        orientation="h",
        title="Top 15 Purchased Products by Quantity",
    )
    fig_top_qty.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top_qty, use_container_width=True)

    hourly = (
        positive.assign(Hour=positive["InvoiceDate"].dt.hour)
        .groupby("Hour", as_index=False)["LineTotal"]
        .sum()
    )
    fig_hour = _gradient_bar(hourly, x="Hour", y="LineTotal", title="Revenue by Hour of Day")
    st.plotly_chart(fig_hour, use_container_width=True)

    if not insights.empty and {"segment_id", "Monetary", "RecencyDays", "Frequency"}.issubset(insights.columns):
        seg = (
            insights.groupby("segment_id", as_index=False)
            .agg(
                Customers=("CustomerID", "nunique"),
                AvgMonetary=("Monetary", "mean"),
                AvgRecencyDays=("RecencyDays", "mean"),
                AvgFrequency=("Frequency", "mean"),
            )
            .sort_values("Customers", ascending=False)
        )
        fig_seg = _gradient_bar(
            seg,
            x="segment_id",
            y="Customers",
            title="Cluster Distribution",
            text="Customers",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        fig_bubble = px.scatter(
            seg,
            x="AvgRecencyDays",
            y="AvgMonetary",
            size="Customers",
            color="segment_id",
            hover_data=["AvgFrequency"],
            title="Segment Profile (Recency vs Monetary, bubble=Customers)",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        needed = {"rfm_composite_score", "churn_risk_score", "transactional_promoter_score", "clv_estimate"}
        if needed.issubset(insights.columns):
            st.markdown("#### 3 Customer Groups and Their Characteristics")
            group_summary, group_desc = _segment_three_groups(insights)
            fig_group = _gradient_bar(
                group_summary,
                x="customer_group",
                y="Customers",
                title="Three-Group Customer Distribution",
            )
            st.plotly_chart(fig_group, use_container_width=True)

            display_table = group_summary.copy()
            for col in [
                "AvgMonetary",
                "AvgFrequency",
                "AvgRecencyDays",
                "AvgChurnRisk",
                "AvgCLV",
                "MedCLV",
                "AvgAdjCLV",
                "MedAdjCLV",
            ]:
                display_table[col] = display_table[col].round(2)
            st.dataframe(display_table, use_container_width=True, hide_index=True)

            for grp in ["VIP / Loyal", "Growth / Nurture", "At-Risk / Win-Back"]:
                st.markdown(f"- **{grp}**: {group_desc[grp]}")

        dist_cols = st.columns(2)
        fig_monetary = px.histogram(
            insights,
            x="Monetary",
            nbins=60,
            title="Customer Monetary Distribution",
        )
        dist_cols[0].plotly_chart(fig_monetary, use_container_width=True)
        fig_churn = px.histogram(
            insights,
            x="churn_risk_score",
            nbins=40,
            title="Customer Churn Risk Distribution",
        )
        dist_cols[1].plotly_chart(fig_churn, use_container_width=True)


def _safe_parse_recommendation(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            return [value]
    return []


def _local_customer_inference(insights: pd.DataFrame, customer_id: int) -> dict | None:
    if insights.empty:
        return None
    hit = insights.loc[insights["CustomerID"] == customer_id]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    row["recommended_stock_codes"] = _safe_parse_recommendation(row.get("recommended_stock_codes"))
    return row


def _api_customer_inference(base_url: str, customer_id: int) -> tuple[dict | None, str | None]:
    url = f"{base_url.rstrip('/')}/v1/customers/{customer_id}"
    try:
        r = requests.get(url, timeout=4)
    except requests.RequestException as exc:
        return None, f"API call failed: {exc}"
    if r.status_code == 200:
        return r.json(), None
    return None, f"API returned {r.status_code}: {r.text}"


def build_personalization(df: pd.DataFrame, insights: pd.DataFrame) -> None:
    st.subheader("Part 2: Customer Personalization")
    st.caption("Nhap Customer ID de xem lich su giao dich, features, va ket qua inference tu API.")

    left, right = st.columns([2, 1])
    customer_id = left.number_input("Customer ID", min_value=1, value=17850, step=1)
    api_base = right.text_input("API Base URL", value="http://127.0.0.1:8000")

    cid = int(customer_id)
    tx_c = df.loc[df["CustomerID"] == cid].copy()
    if tx_c.empty:
        st.warning(f"Khong tim thay giao dich cho CustomerID={cid} trong dataset.")
        return

    tx_c = tx_c.sort_values("InvoiceDate")
    daily = tx_c.groupby(tx_c["InvoiceDate"].dt.date, as_index=False)["LineTotal"].sum()
    daily = daily.rename(columns={"InvoiceDate": "Date"})
    fig_customer = px.bar(
        daily,
        x="Date",
        y="LineTotal",
        title=f"Transaction Value Over Time - Customer {cid}",
    )
    st.plotly_chart(fig_customer, use_container_width=True)

    local_row = _local_customer_inference(insights, cid)
    if local_row:
        st.markdown("**Customer Features (from latest insights store)**")
        show_keys = [
            "segment_id",
            "Monetary",
            "Frequency",
            "RecencyDays",
            "TenureDays",
            "AvgOrderValue",
            "OrdersPerMonth",
            "UniqueProducts",
            "CancelRate",
            "rfm_composite_score",
            "churn_risk_score",
            "clv_estimate",
            "transactional_promoter_score",
            "campaign_hints",
            "recommended_stock_codes",
        ]
        feature_rows = []
        for key in show_keys:
            if key in local_row:
                feature_rows.append({"Feature": key, "Value": local_row[key]})
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Customer nay chua co trong insights parquet offline.")

    st.markdown("**API Inference Result**")
    api_payload, api_error = _api_customer_inference(api_base, cid)
    if api_error:
        st.error(api_error)
    else:
        st.json(api_payload)


@st.cache_data(show_spinner=False)
def load_evaluation_report() -> dict | None:
    if not EVAL_LATEST_PATH.exists():
        return None
    try:
        with open(EVAL_LATEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_evaluation_history() -> list[dict]:
    if not EVAL_HISTORY_DIR.exists():
        return []
    runs: list[dict] = []
    for path in sorted(EVAL_HISTORY_DIR.glob("run_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: str(r.get("run_id", "")))
    return runs


_STATUS_BADGE = {
    "ok": "OK",
    "warning": "WARNING",
    "failed": "FAILED",
    "info": "INFO",
}


def _status_color(status: str) -> str:
    return {
        "ok": "#16a34a",
        "warning": "#d97706",
        "failed": "#dc2626",
        "info": "#6b7280",
    }.get(status, "#6b7280")


def _checks_to_dataframe(checks_block: list[dict]) -> pd.DataFrame:
    if not checks_block:
        return pd.DataFrame(columns=["status", "name", "value", "threshold", "message"])
    rows = []

    def _as_display_text(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, ensure_ascii=False, default=str)
        return str(v)

    for c in checks_block:
        rows.append(
            {
                "status": _STATUS_BADGE.get(str(c.get("status", "info")), str(c.get("status", "info")).upper()),
                "name": _as_display_text(c.get("name", "")),
                "value": _as_display_text(c.get("value")),
                "threshold": _as_display_text(c.get("threshold")),
                "message": _as_display_text(c.get("message", "")),
            }
        )
    return pd.DataFrame(rows)


def _extract_metric(checks_block: list[dict], name: str, key: str | None = None) -> float | None:
    for c in checks_block:
        if c.get("name") == name:
            v = c.get("value")
            if key is None:
                return float(v) if isinstance(v, (int, float)) else None
            if isinstance(v, dict) and key in v:
                try:
                    return float(v[key])
                except (TypeError, ValueError):
                    return None
            return None
    return None


def build_quality_monitoring() -> None:
    st.subheader("Part 3: Quality Monitoring")
    st.caption(
        "Automatic quality checks computed at the end of every refresh run. "
        "Source: app/artifacts/evaluation/evaluation_report.json"
    )

    report = load_evaluation_report()
    if report is None:
        st.info(
            "Khong tim thay evaluation_report.json. "
            "Chay 'python -m app.jobs.refresh_insights' de tao bao cao dau tien."
        )
        return

    summary = report.get("summary") or {}
    overall = str(summary.get("overall_status", "info"))
    badge_color = _status_color(overall)
    st.markdown(
        f"<div style='display:inline-block;padding:8px 16px;border-radius:6px;"
        f"background:{badge_color};color:white;font-weight:600;font-size:14px;"
        f"letter-spacing:0.05em;'>OVERALL STATUS: {overall.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(5)
    cols[0].metric("Total checks", int(summary.get("total", 0)))
    cols[1].metric("Passed", int(summary.get("passed", 0)))
    cols[2].metric("Warnings", int(summary.get("warnings", 0)))
    cols[3].metric("Failed", int(summary.get("failed", 0)))
    cols[4].metric("Info", int(summary.get("info", 0)))

    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Run ID**: `{report.get('run_id', '-')}`")
    meta_cols[1].markdown(f"**Previous run**: `{report.get('previous_run_id') or '-'}`")
    meta_cols[2].markdown(f"**Generated at (UTC)**: {report.get('generated_at_utc', '-')}")

    failed_names = summary.get("failed_names") or []
    warning_names = summary.get("warning_names") or []
    if failed_names:
        st.error("Failed checks: " + ", ".join(failed_names))
    if warning_names:
        st.warning("Warning checks: " + ", ".join(warning_names))

    sections = [
        ("Data Quality", report.get("data_quality") or []),
        ("Feature Quality", report.get("feature_quality") or []),
        ("Model Quality", report.get("model_quality") or []),
        ("Business Metrics", report.get("business_metrics") or []),
        ("Regression vs Previous", report.get("regression_vs_previous") or []),
    ]
    for title, block in sections:
        st.markdown(f"#### {title}")
        df_block = _checks_to_dataframe(block)
        if df_block.empty:
            st.caption("No checks in this section.")
            continue
        st.dataframe(df_block, use_container_width=True, hide_index=True)

    history = load_evaluation_history()
    if len(history) >= 2:
        st.markdown("#### Trend Across Recent Runs")
        rows = []
        for r in history:
            run_id = r.get("run_id", "")
            model = r.get("model_quality") or []
            feat = r.get("feature_quality") or []
            biz = r.get("business_metrics") or []
            rows.append(
                {
                    "run_id": run_id,
                    "silhouette": _extract_metric(model, "silhouette_score"),
                    "davies_bouldin": _extract_metric(model, "davies_bouldin_score"),
                    "calinski_harabasz": _extract_metric(model, "calinski_harabasz_score"),
                    "n_customers": _extract_metric(feat, "n_customers"),
                    "recommendation_coverage": _extract_metric(biz, "recommendation_coverage"),
                    "churn_mean": _extract_metric(biz, "churn_risk_score_distribution", key="mean"),
                    "clv_median": _extract_metric(biz, "clv_estimate_distribution", key="median"),
                }
            )
        trend_df = pd.DataFrame(rows).sort_values("run_id")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

        chart_specs = [
            ("silhouette", "Silhouette Score Trend"),
            ("davies_bouldin", "Davies-Bouldin Trend (lower is better)"),
            ("n_customers", "Customer Count Trend"),
            ("recommendation_coverage", "Recommendation Coverage Trend"),
            ("churn_mean", "Mean Churn Risk Score Trend"),
            ("clv_median", "Median CLV Estimate Trend"),
        ]
        chart_cols = st.columns(2)
        for idx, (col, title) in enumerate(chart_specs):
            sub = trend_df.dropna(subset=[col])
            if sub.empty:
                continue
            fig = px.line(sub, x="run_id", y=col, markers=True, title=title, text=col)
            fig.update_traces(texttemplate="%{text:.4f}", textposition="top center")
            chart_cols[idx % 2].plotly_chart(fig, use_container_width=True)
    elif history:
        st.caption("Only one run in history; trend charts will appear after the next refresh.")
    else:
        st.caption("No history snapshots yet.")


def main() -> None:
    st.set_page_config(page_title="Customer Insight Dashboard", layout="wide")
    st.title("Customer Insight Dashboard")
    st.caption("Dashboard for transaction analytics, customer segmentation insights, and personalization lookup.")

    if not TX_PATH.exists():
        st.error(f"Khong tim thay file dataset: {TX_PATH}")
        st.stop()

    df = load_transactions()
    insights = load_insights()

    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Customer Personalization", "Quality Monitoring"])
    with tab1:
        build_overview_charts(df, insights)
    with tab2:
        build_personalization(df, insights)
    with tab3:
        build_quality_monitoring()


if __name__ == "__main__":
    main()
