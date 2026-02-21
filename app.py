import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import streamlit.components.v1 as components
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import altair as alt
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    make_subplots = None


import base64
import json
import math
import html
from datetime import date

def _json_safe(x):
    """Convert common non-JSON-serializable objects (pandas/numpy/datetime/Decimal/NaN) to safe types."""
    # pandas Timestamp / datetime-like
    if hasattr(x, "to_pydatetime"):
        try:
            x = x.to_pydatetime()
        except Exception:
            pass

    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # numpy scalars -> python scalars
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass

    # Decimal -> float
    try:
        from decimal import Decimal
        if isinstance(x, Decimal):
            return float(x)
    except Exception:
        pass

    # NaN / inf -> None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None

    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    return x
import requests
import re

def clean_markdown_spacing(text: str) -> str:
    """
    Fix common Markdown spacing issues that cause words to collapse.
    """
    if not text:
        return text

    # Ensure space after commas and periods if missing
    text = re.sub(r",([A-Za-z])", r", \1", text)
    text = re.sub(r"\.([A-Za-z])", r". \1", text)

    # Ensure space before markdown italics if stuck to previous token
    text = re.sub(r"([^\s])(\*)", r"\1 \2", text)

    # Ensure space after closing italics
    text = re.sub(r"(\*)([^\s])", r"\1 \2", text)

    return text


def render_ai_text(text: str):
    """
    Render AI output as plain wrapped text (not Markdown) to avoid accidental
    style glitches from tokens like '*' or '_' inside model output.
    """
    safe = html.escape(text or "").replace("\n", "<br>")
    st.markdown(f"<div class='ai-output-wrap'>{safe}</div>", unsafe_allow_html=True)
# --- Optional: ChatGPT / OpenAI API ---
# Install: pip install openai
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

def extract_sku_from_campaign_name(name: str):
    """
    Extract SKU from campaign name like:
    'anything - IQ010106MAGG99'
    Works with letters+numbers+_ and trims spaces.
    Returns None if not found.
    """
    if name is None:
        return None
    s = str(name).strip()

    # Take the LAST " - " segment (or last hyphen) as SKU
    parts = re.split(r"\s*-\s*", s)
    if len(parts) < 2:
        return None

    sku = parts[-1].strip()
    # Optional: validate SKU shape (at least 6 chars, alnum/_ only)
    if not re.fullmatch(r"[A-Za-z0-9_]+", sku):
        return None
    if len(sku) < 6:
        return None

    return sku

def build_sku_to_name_map(orders_df: pd.DataFrame) -> dict:
    """
    Returns {sku: product_name} from orders file.
    Picks the most frequent product name per SKU.
    """
    if orders_df is None or orders_df.empty:
        return {}

    # try common column names (Arabic + possible English)
    sku_col = None
    name_col = None

    for c in ["كود_المنتج", "product_sku", "sku", "product_code"]:
        if c in orders_df.columns:
            sku_col = c
            break

    for c in ["اسم_المنتج", "product_name", "name"]:
        if c in orders_df.columns:
            name_col = c
            break

    if not sku_col or not name_col:
        return {}

    tmp = orders_df[[sku_col, name_col]].copy()
    tmp[sku_col] = tmp[sku_col].astype(str).str.strip()
    tmp[name_col] = tmp[name_col].astype(str).str.strip()

    tmp = tmp[(tmp[sku_col] != "") & (tmp[name_col] != "")]

    # most common name per sku
    sku_name = (
        tmp.groupby(sku_col)[name_col]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return sku_name


def compute_taager_kpis(orders_df: pd.DataFrame, campaigns_df: pd.DataFrame, taager_fx: float = 1602.0):
    if orders_df is None or campaigns_df is None:
        return None, None

    spend_usd = float(campaigns_df["Amount spent (USD)"].sum()) if "Amount spent (USD)" in campaigns_df.columns else 0.0

    delivered_profit_iqd = float(orders_df["delivered_profit_iqd"].sum()) if "delivered_profit_iqd" in orders_df.columns else 0.0
    confirmed_profit_iqd = float(orders_df["confirmed_profit_iqd"].sum()) if "confirmed_profit_iqd" in orders_df.columns else 0.0

    delivered_profit_usd_taager = delivered_profit_iqd / taager_fx
    confirmed_profit_usd_taager = confirmed_profit_iqd / taager_fx

    net_profit_usd_taager = delivered_profit_usd_taager - spend_usd
    potential_net_usd_taager = confirmed_profit_usd_taager - spend_usd
    return net_profit_usd_taager, potential_net_usd_taager


def github_get_file_bytes(token: str, repo: str, path: str, branch: str = "main") -> bytes:
    """
    Download file bytes from GitHub.
    - For small files: decode base64 from the Contents API response.
    - For >1MB files: use the raw media type (content field can be empty / encoding "none").
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers_json = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    r = requests.get(api_url, headers=headers_json, params={"ref": branch}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub GET failed ({r.status_code}): {r.text}")

    data = r.json()
    enc = data.get("encoding")

    # Small file path: content is base64
    if enc == "base64":
        return base64.b64decode(data.get("content", ""))

    # Large file path: request raw content
    headers_raw = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.raw",
    }
    r2 = requests.get(api_url, headers=headers_raw, params={"ref": branch}, timeout=30)
    if r2.status_code != 200:
        raise RuntimeError(f"GitHub GET raw failed ({r2.status_code}): {r2.text}")
    return r2.content

def load_latest_from_github():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    if not token or not repo:
        return None

    try:
        kpis_bytes = github_get_file_bytes(token, repo, "data/latest_kpis.json", branch=branch)
        payload = json.loads(kpis_bytes.decode("utf-8"))

        orders_bytes = github_get_file_bytes(token, repo, "data/latest_orders.csv", branch=branch)
        campaigns_bytes = github_get_file_bytes(token, repo, "data/latest_campaigns.csv", branch=branch)
        # Optional: daily orders (may not exist yet)
        daily_orders_bytes = None
        try:
            daily_orders_bytes = github_get_file_bytes(token, repo, "data/latest_daily_orders.xlsx", branch=branch)
        except Exception:
            daily_orders_bytes = None
    except Exception:
        return None

    return {
        "generated_at": payload.get("generated_at"),
        "kpis": payload.get("kpis"),
        "orders_csv_bytes": orders_bytes,
        "campaigns_csv_bytes": campaigns_bytes,
        "daily_orders_xlsx_bytes": daily_orders_bytes,
    }

def campaigns_metric_explorer_sku(campaigns_df: pd.DataFrame, orders_df: pd.DataFrame):
    df = campaigns_df.copy()

    required = ["Reporting starts", "Campaign name", "Campaign delivery"]
    for c in required:
        if c not in df.columns:
            st.error(f"Campaigns file missing column: {c}")
            return

    df["Reporting starts"] = pd.to_datetime(df["Reporting starts"], errors="coerce")
    df = df.dropna(subset=["Reporting starts"])
    df["day"] = df["Reporting starts"].dt.floor("D")

    df["Campaign delivery"] = df["Campaign delivery"].astype(str).str.lower().str.strip()

    # Clean numeric columns if present
    numeric_candidates = ["Amount spent (USD)", "Impressions", "Reach", "Results"]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = to_num(df[c])

    # Build SKU
    df["sku"] = df["Campaign name"].apply(extract_sku_from_campaign_name)

    sku_to_name = build_sku_to_name_map(orders_df)  # <-- uses orders file
    df["product_name"] = df["sku"].map(sku_to_name)
    df["label"] = df.apply(
        lambda r: f"{r['product_name']} — {r['sku']}" if pd.notna(r["product_name"]) and r["product_name"] else (r["sku"] if r["sku"] else "UNKNOWN"),
        axis=1
    )

    # ---- UI ----
    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("### Settings")
        group_by = st.selectbox("Group by", ["SKU (product)", "Campaign"], index=0)
        only_active = st.checkbox("Only active", value=True)

        metric_options = []
        if "Results" in df.columns: metric_options.append(("Results", "Results"))
        if "Amount spent (USD)" in df.columns: metric_options.append(("Spend (USD)", "Amount spent (USD)"))
        if "Impressions" in df.columns: metric_options.append(("Impressions", "Impressions"))
        if "Reach" in df.columns: metric_options.append(("Reach", "Reach"))
        if "Amount spent (USD)" in df.columns and "Impressions" in df.columns:
            metric_options.append(("CPM (USD)", "DERIVED_CPM"))
        if "Amount spent (USD)" in df.columns and "Results" in df.columns:
            metric_options.append(("Cost per Result (USD)", "DERIVED_CPR"))

        metric_label, metric_key = st.selectbox(
            "Metric",
            options=metric_options,
            format_func=lambda x: x[0],
            index=0
        )

        top_n = st.slider("Top N", min_value=3, max_value=30, value=10, step=1)

        if group_by == "SKU (product)":
            include_unknown = st.checkbox("Include campaigns without SKU", value=False)

    df2 = df[df["Campaign delivery"].eq("active")].copy() if only_active else df.copy()
    if df2.empty:
        st.info("No data after filtering.")
        return

    # Choose grouping column
    if group_by == "SKU (product)":
        if not include_unknown:
            df2 = df2.dropna(subset=["sku"])
        df2["group"] = df2["label"].fillna("UNKNOWN")
    else:
        df2["group"] = df2["Campaign name"].astype(str)

    if df2.empty:
        st.info("No data after SKU filtering.")
        return

    # Build daily aggregated "value"
    if metric_key == "DERIVED_CPM":
        tmp = (
            df2.groupby(["day", "group"], as_index=False)[["Amount spent (USD)", "Impressions"]]
            .sum()
        )
        tmp["value"] = tmp["Amount spent (USD)"] / tmp["Impressions"].replace({0: pd.NA}) * 1000
        daily = tmp[["day", "group", "value"]].dropna(subset=["value"])

    elif metric_key == "DERIVED_CPR":
        tmp = (
            df2.groupby(["day", "group"], as_index=False)[["Amount spent (USD)", "Results"]]
            .sum()
        )
        tmp["value"] = tmp["Amount spent (USD)"] / tmp["Results"].replace({0: pd.NA})
        daily = tmp[["day", "group", "value"]].dropna(subset=["value"])

    else:
        daily = (
            df2.groupby(["day", "group"], as_index=False)[metric_key]
            .sum()
            .rename(columns={metric_key: "value"})
        )

    # Top N groups by total value
    top = (
        daily.groupby("group", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .head(top_n)
    )
    top_groups = top["group"].tolist()

    daily = daily[daily["group"].isin(top_groups)].sort_values("day")

    # Date range
    min_d = daily["day"].min().date()
    max_d = daily["day"].max().date()

    with left:
        st.markdown("### Date range")
        d_from, d_to = st.date_input("Range", value=(min_d, max_d))

        st.markdown("### Selection")
        selected = st.multiselect(
            "Select items",
            options=top_groups,
            default=top_groups[:3] if len(top_groups) >= 3 else top_groups
        )

    if not selected:
        with right:
            st.info("Select at least 1 item.")
        return

    daily = daily[(daily["day"].dt.date >= d_from) & (daily["day"].dt.date <= d_to)]
    daily = daily[daily["group"].isin(selected)]

    if daily.empty:
        with right:
            st.info("No data for this selection.")
        return

    # Interactive chart
    with right:
        title = f"{metric_label} per day — grouped by {'SKU' if group_by=='SKU (product)' else 'Campaign'}"
        st.markdown(f"### {title}")

        nearest = alt.selection_point(nearest=True, on="mouseover", fields=["day"], empty=False)

        base = alt.Chart(daily).encode(
            x=alt.X("day:T", title="Day"),
            y=alt.Y("value:Q", title=metric_label),
            color=alt.Color("group:N", legend=alt.Legend(title="SKU" if group_by == "SKU (product)" else "Campaign")),
        )

        lines = base.mark_line(point=True).encode(
            tooltip=[
                alt.Tooltip("day:T", title="Day"),
                alt.Tooltip("group:N", title=("SKU" if group_by == "SKU (product)" else "Campaign")),
                alt.Tooltip("value:Q", title=metric_label),
            ]
        )

        selectors = base.mark_point().encode(opacity=alt.value(0)).add_params(nearest)
        points = base.mark_point(size=80).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        rule = alt.Chart(daily).mark_rule().encode(x="day:T").transform_filter(nearest)

        chart = (lines + selectors + points + rule).properties(height=430).interactive()
        st.altair_chart(chart, use_container_width=True)

        with st.expander(f"Top {top_n} by {metric_label}", expanded=False):
            top_view = top.rename(columns={"group": ("SKU" if group_by == "SKU (product)" else "Campaign"),
                                           "value": metric_label})
            st.dataframe(top_view, use_container_width=True)


def product_deep_dive(orders_df: pd.DataFrame, campaigns_df: pd.DataFrame, fx: float, currency: str):
    if orders_df is None or campaigns_df is None:
        st.info("Upload BOTH Orders + Campaigns to use Product Deep Dive.")
        return

    # --- Build SKU map from orders ---
    sku_to_name = build_sku_to_name_map(orders_df)

    # detect SKU column in orders
    sku_col = None
    for c in ["كود_المنتج", "product_sku", "sku", "product_code"]:
        if c in orders_df.columns:
            sku_col = c
            break
    if not sku_col:
        st.error("Orders file missing SKU column (expected كود_المنتج / sku / product_sku).")
        return

    # --- SKU selector (from orders list) ---
    all_skus = sorted(orders_df[sku_col].astype(str).str.strip().unique().tolist())

    def sku_label(sku: str) -> str:
        name = sku_to_name.get(sku, "")
        return f"{name} — {sku}" if name else str(sku)

    selected_sku = st.selectbox(
        "Select product",
        options=all_skus,
        format_func=sku_label
    )

    product_name = sku_to_name.get(selected_sku, "")
    st.subheader(f"{product_name} — {selected_sku}" if product_name else str(selected_sku))


    # --- Orders slice for this SKU ---
    o = orders_df[orders_df[sku_col].astype(str).str.strip() == str(selected_sku)].copy()
    if o.empty:
        st.warning("No orders found for this SKU in the Orders file.")
        return

    # Product KPIs from orders
    requested = float(o["requested_units"].sum()) if "requested_units" in o.columns else 0.0
    confirmed = float(o["confirmed_units"].sum()) if "confirmed_units" in o.columns else 0.0
    delivered = float(o["delivered_units"].sum()) if "delivered_units" in o.columns else 0.0

    delivered_profit_usd = float(o["delivered_profit_usd"].sum()) if "delivered_profit_usd" in o.columns else 0.0
    confirmed_profit_usd = float(o["confirmed_profit_usd"].sum()) if "confirmed_profit_usd" in o.columns else 0.0

    # --- Campaigns slice for this SKU ---
    c = campaigns_df.copy()
    if "Campaign name" not in c.columns:
        st.error("Campaigns file missing 'Campaign name' column.")
        return

    c["sku"] = c["Campaign name"].apply(extract_sku_from_campaign_name)
    c = c[c["sku"].astype(str) == str(selected_sku)].copy()

    if "Amount spent (USD)" in c.columns:
        c["Amount spent (USD)"] = to_num(c["Amount spent (USD)"])
        spend_usd = float(c["Amount spent (USD)"].sum())
    else:
        spend_usd = 0.0

    net_profit_usd = delivered_profit_usd - spend_usd
    potential_net_usd = confirmed_profit_usd - spend_usd

    # --- KPI Cards (with tooltips) ---
    def _disp_usd_to_currency(x_usd: float) -> float:
        return x_usd * fx if currency == "IQD" else x_usd

    spend_disp = _disp_usd_to_currency(spend_usd)
    delivered_profit_disp = _disp_usd_to_currency(delivered_profit_usd)
    net_after_ads_disp = _disp_usd_to_currency(net_profit_usd)
    potential_net_disp = _disp_usd_to_currency(potential_net_usd)

    roas = safe_ratio(delivered_profit_usd, spend_usd)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card(
            "Spend",
            money_ccy(spend_disp, currency),
            "",
            "neutral",
            tip="Spend = Sum of Amount spent (USD) from campaigns for this SKU."
        )

    with c2:
        _card(
            "Delivered units",
            f"{int(delivered):,}",
            "",
            "neutral",
            tip="Delivered units = Sum of delivered_units from Orders for this SKU."
        )

    with c3:
        _card(
            "Delivered profit",
            money_ccy(delivered_profit_disp, currency),
            "",
            _tone(delivered_profit_usd),
            tip="Delivered profit = Sum of delivered_profit from Orders for this SKU."
        )

    with c4:
        _card(
            "Net after ads",
            money_ccy(net_after_ads_disp, currency),
            "",
            _tone(net_profit_usd),
            tip="Net after ads = Delivered profit − Spend."
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        _card("Requested", f"{int(requested):,}", "", "neutral",
            tip="Requested = Sum of requested_units from Orders.")

    with c6:
        _card("Confirmed", f"{int(confirmed):,}", "", "neutral",
            tip="Confirmed = Sum of confirmed_units from Orders.")

    with c7:
        _card(
            "Potential net",
            money_ccy(potential_net_disp, currency),
            "",
            _tone(potential_net_usd),
            tip="Potential net = Confirmed profit − Spend."
        )

    with c8:
        _card(
            "ROAS",
            "N/A" if roas is None else f"{roas:.2f}",
            "",
            "neutral",
            tip="ROAS = Delivered profit ÷ Spend."
        )


    st.divider()

    # --- Daily trend: Orders vs Spend (interactive) ---
    # Orders file must have a date column to do daily trend.
    # We'll try common names; if none exists, we show tables only.
    date_col = None
    for dcol in ["order_date", "created_at", "date", "تاريخ_الطلب"]:
        if dcol in o.columns:
            date_col = dcol
            break

    if date_col and not c.empty and "Reporting starts" in c.columns:
        o["day"] = pd.to_datetime(o[date_col], errors="coerce").dt.floor("D")
        o = o.dropna(subset=["day"])

        c["day"] = pd.to_datetime(c["Reporting starts"], errors="coerce").dt.floor("D")
        c = c.dropna(subset=["day"])

        daily_orders = o.groupby("day", as_index=False)["delivered_units"].sum().rename(columns={"delivered_units": "orders"})
        daily_spend = c.groupby("day", as_index=False)["Amount spent (USD)"].sum().rename(columns={"Amount spent (USD)": "spend_usd"})

        daily = pd.merge(daily_orders, daily_spend, on="day", how="outer").fillna(0).sort_values("day")

        # Currency display
        daily["spend_disp"] = daily["spend_usd"] * fx if currency == "IQD" else daily["spend_usd"]

        st.subheader("Daily trend (Orders vs Spend)")

        base = alt.Chart(daily).encode(x=alt.X("day:T", title="Day"))

        orders_line = base.mark_line(point=True).encode(
            y=alt.Y("orders:Q", title="Delivered units"),
            tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("orders:Q", title="Delivered units")]
        )

        spend_line = base.mark_line(point=True).encode(
            y=alt.Y("spend_disp:Q", title=f"Spend ({currency})"),
            tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("spend_disp:Q", title=f"Spend ({currency})")]
        )

        st.altair_chart((orders_line + spend_line).properties(height=420).interactive(), use_container_width=True)
    else:
        st.info("Daily trend needs an order date column in Orders + 'Reporting starts' in Campaigns. Showing breakdown tables instead.")

    # --- Breakdown tables ---
    st.subheader("Campaigns for this product")
    if c.empty:
        st.info("No campaigns detected for this SKU (based on Campaign name SKU extraction).")
    else:
        show_cols = [x for x in ["Campaign name", "Campaign delivery", "Amount spent (USD)", "Results", "Impressions", "Reach"] if x in c.columns]
        st.dataframe(c[show_cols].sort_values("Amount spent (USD)", ascending=False) if "Amount spent (USD)" in c.columns else c[show_cols],
                     use_container_width=True)

    st.subheader("Orders rows for this product")
    st.dataframe(o, use_container_width=True)


def github_put_file(token: str, repo: str, path: str, content_bytes: bytes, message: str, branch: str = "main"):
    """
    Create or update a file in a GitHub repo using the Contents API.
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Check if file exists to get its SHA (required for update)
    r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=30)
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha")
    elif r.status_code != 404:
        raise RuntimeError(f"GitHub GET failed ({r.status_code}): {r.text}")

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(api_url, headers=headers, json=payload, timeout=30)
    if r2.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed ({r2.status_code}): {r2.text}")

    return r2.json()


def save_latest_to_github(kpis: dict, pdf_bytes: bytes, xlsx_bytes: bytes,
                          orders_csv_bytes: bytes, campaigns_csv_bytes: bytes,
                          daily_orders_xlsx_bytes: Optional[bytes] = None):
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("GITHUB_REPO", None)
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    if not token or not repo:
        raise RuntimeError("Missing GitHub secrets. Please set GITHUB_TOKEN and GITHUB_REPO in Streamlit Secrets.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # 0) Save raw inputs (latest only)
    github_put_file(token, repo, "data/latest_orders.csv", orders_csv_bytes, f"Update latest Orders CSV ({now})", branch)
    github_put_file(token, repo, "data/latest_campaigns.csv", campaigns_csv_bytes, f"Update latest Campaigns CSV ({now})", branch)
    if daily_orders_xlsx_bytes:
        github_put_file(token, repo, "data/latest_daily_orders.xlsx", daily_orders_xlsx_bytes, f"Update latest Daily Orders XLSX ({now})", branch)

    # 1) KPIs JSON
    kpis_payload = {"generated_at": now, "kpis": kpis}
    kpis_bytes = json.dumps(kpis_payload, ensure_ascii=False, indent=2).encode("utf-8")
    github_put_file(token, repo, "data/latest_kpis.json", kpis_bytes, f"Update latest KPIs ({now})", branch)

    # 2) PDF
    github_put_file(token, repo, "data/latest_dashboard.pdf", pdf_bytes, f"Update latest PDF dashboard ({now})", branch)

    # 3) Excel
    github_put_file(token, repo, "data/latest_dashboard.xlsx", xlsx_bytes, f"Update latest Excel dashboard ({now})", branch)



# ------------------ Helpers ------------------
def to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("٬", "", regex=False)
    s = s.str.replace("٫", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0)


def parse_daily_orders(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Taager daily orders export.
    Expected columns (sample):
    - Created At, Status, orders.export.cashOnDelivery, Shipping Cost, VAT Profit, Order Profit
    - SKUs, Quantities, Prices
    """
    if daily_df is None or daily_df.empty:
        return daily_df

    df = daily_df.copy()

    # Parse datetime + day
    if "Created At" in df.columns:
        df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")
        df["day"] = df["Created At"].dt.floor("D")

    # Clean numeric columns (strings with commas)
    for c in ["orders.export.cashOnDelivery", "Shipping Cost", "VAT Profit", "Order Profit", "Prices"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    if "Quantities" in df.columns:
        df["Quantities"] = to_num(df["Quantities"])

    return df



def render_fixed_quick_kpis(
    daily_orders_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame],
    fx: float,
    currency: str,
):
    """Floating bottom-right accordion (expander) that stays across tabs."""

    # Helper to display COD amounts in selected currency
    def disp_money_iqd(iqd: float) -> str:
        if currency == "USD":
            return money_ccy(iqd_to_usd(iqd, fx), "USD")
        return money_ccy(iqd, "IQD")

    # Build a dynamic expander title (works even when collapsed)
    if daily_orders_df is None or getattr(daily_orders_df, "empty", True):
        summary_title = "Summary — Upload Daily Orders (Taager) XLSX"
        df = None
        selected_day = None
    else:
        df = parse_daily_orders(daily_orders_df)
        if "day" not in df.columns or df["day"].isna().all():
            summary_title = "Summary — No usable date in Daily Orders"
            selected_day = None
        else:
            min_day = df["day"].min().date()
            max_day = df["day"].max().date()

            if "quick_kpi_day" not in st.session_state:
                st.session_state.quick_kpi_day = max_day

            # Use last selected day for the summary title
            selected_day = st.session_state.quick_kpi_day
            # Clamp to available range (in case file changed)
            if selected_day < min_day:
                selected_day = min_day
                st.session_state.quick_kpi_day = min_day
            if selected_day > max_day:
                selected_day = max_day
                st.session_state.quick_kpi_day = max_day

            d = df[df["day"] == pd.to_datetime(selected_day)].copy()

            # Remove "Cancelled by You"
            if "Status" in d.columns:
                status_clean = d["Status"].astype(str).str.strip().str.lower()
                d = d[~status_clean.str.contains("cancelled by you", na=False)].copy()

            id_col = get_daily_order_id_col(d)
            if id_col is None:
                d["__rowid__"] = range(len(d))
                id_col = "__rowid__"

            cod_col = "orders.export.cashOnDelivery"
            if cod_col in d.columns:
                d[cod_col] = to_num(d[cod_col])
            else:
                d[cod_col] = 0.0

            orders_count = int(d[id_col].nunique()) if len(d) else 0
            orders_amount_iqd = float(d[cod_col].sum()) if len(d) else 0.0

            delivered_mask = pd.Series(False, index=d.index)
            if "Status" in d.columns:
                delivered_mask = (
                    d["Status"].astype(str).str.strip().str.lower().str.contains("delivered", na=False)
                )
            deliveries_count = int(d.loc[delivered_mask, id_col].nunique()) if len(d) else 0
            deliveries_amount_iqd = float(d.loc[delivered_mask, cod_col].sum()) if len(d) else 0.0

            delivery_rate = (deliveries_count / orders_count * 100.0) if orders_count else 0.0

            summary_title = (
                f"Summary — Orders: {orders_count} | COD: {disp_money_iqd(orders_amount_iqd)} | Delivered: {deliveries_count} ({delivery_rate:.0f}%)"
            )

    # Expander (we'll pin it via JS+CSS)
    with st.expander(summary_title, expanded=False):
        # Marker: used by JS to locate the expander container and add a CSS class
        st.markdown('<div id="quick-kpi-marker"></div>', unsafe_allow_html=True)

        components.html(
            """
            <script>
            (function(){
              const marker = window.parent.document.getElementById('quick-kpi-marker');
              if (!marker) return;

              // Walk up to the expander root container
              let el = marker;
              for (let i = 0; i < 20; i++) {
                if (!el) break;
                // Streamlit expander root usually carries data-testid="stExpander"
                if (el.getAttribute && el.getAttribute('data-testid') === 'stExpander') break;
                el = el.parentElement;
              }
              if (!el) return;

              el.classList.add('kpi-fixed-expander');
            })();
            </script>
            """,
            height=0,
        )

        if df is None or selected_day is None:
            st.info("Upload Daily Orders (Taager) XLSX to enable this panel.")
            return

        # Date filter (inside accordion)
        min_day = df["day"].min().date()
        max_day = df["day"].max().date()
        st.date_input(
            "Select day",
            min_value=min_day,
            max_value=max_day,
            key="quick_kpi_day",
        )

        # Recompute for selected day (so metrics + table follow the date input)
        d = df[df["day"] == pd.to_datetime(st.session_state.quick_kpi_day)].copy()

        if "Status" in d.columns:
            status_clean = d["Status"].astype(str).str.strip().str.lower()
            d = d[~status_clean.str.contains("cancelled by you", na=False)].copy()

        id_col = get_daily_order_id_col(d)
        if id_col is None:
            d["__rowid__"] = range(len(d))
            id_col = "__rowid__"

        cod_col = "orders.export.cashOnDelivery"
        if cod_col in d.columns:
            d[cod_col] = to_num(d[cod_col])
        else:
            d[cod_col] = 0.0

        orders_count = int(d[id_col].nunique()) if len(d) else 0
        orders_amount_iqd = float(d[cod_col].sum()) if len(d) else 0.0

        delivered_mask = pd.Series(False, index=d.index)
        if "Status" in d.columns:
            delivered_mask = (
                d["Status"].astype(str).str.strip().str.lower().str.contains("delivered", na=False)
            )
        deliveries_count = int(d.loc[delivered_mask, id_col].nunique()) if len(d) else 0
        deliveries_amount_iqd = float(d.loc[delivered_mask, cod_col].sum()) if len(d) else 0.0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Orders", f"{orders_count:,}", disp_money_iqd(orders_amount_iqd))
        with c2:
            st.metric("Deliveries", f"{deliveries_count:,}", disp_money_iqd(deliveries_amount_iqd))

        st.divider()

        # Per product breakdown
        sku_to_name = build_sku_to_name_map(orders_df) if orders_df is not None else {}
        lines = _explode_order_lines(d)

        if lines is None or lines.empty:
            st.caption("No SKU lines found for this day.")
            return

        if "Status" in d.columns:
            tmp = d[[id_col, "Status"]].copy().rename(columns={id_col: "order_id"})
            tmp["status_clean"] = tmp["Status"].astype(str).str.strip().str.lower()
            tmp["is_delivered"] = tmp["status_clean"].str.contains("delivered", na=False)
        else:
            tmp = pd.DataFrame({"order_id": [], "is_delivered": []})

        lines = lines.merge(tmp[["order_id", "is_delivered"]], on="order_id", how="left")
        lines["is_delivered"] = lines["is_delivered"].fillna(False)

        lines["sku"] = lines["sku"].astype(str).str.strip()
        lines["product_name"] = lines["sku"].map(sku_to_name).fillna("")
        lines["Product"] = lines.apply(
            lambda r: f"{r['product_name']} — {r['sku']}" if r["product_name"] else r["sku"],
            axis=1,
        )

        out = (
            lines.groupby("Product", as_index=False)
            .agg(
                Orders=("order_id", "nunique"),
                Delivered=("is_delivered", "sum"),
                Qty=("qty", "sum"),
                Profit_IQD=("profit_iqd_alloc", "sum"),
            )
            .sort_values("Orders", ascending=False)
        )

        if currency == "USD":
            out["Profit"] = out["Profit_IQD"].apply(lambda v: iqd_to_usd(v, fx))
        else:
            out["Profit"] = out["Profit_IQD"]

        out = out.drop(columns=["Profit_IQD"])

        st.caption("Per product (selected day)")
        st.dataframe(out, use_container_width=True, height=220)

def get_daily_order_id_col(df: pd.DataFrame) -> Optional[str]:
    """Prefer Taager 'ID' (unique row/order identifier). Fallback to 'Store Order ID'."""
    if df is None:
        return None
    if "ID" in df.columns:
        return "ID"
    if "Store Order ID" in df.columns:
        return "Store Order ID"
    return None

def build_daily_summary(daily_df: pd.DataFrame, campaigns_df: pd.DataFrame, fx: float, currency: str) -> pd.DataFrame:
    """
    Build day-level KPIs:
    - orders_count
    - profit (Order Profit)
    - cash_on_delivery
    - shipping_cost
    - ads_spend
    - net_after_ads
    - status breakdown
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    df = parse_daily_orders(daily_df)

    if "day" not in df.columns:
        return pd.DataFrame()

    # Orders count (prefer Store Order ID if available, else ID, else row count)
    if "Store Order ID" in df.columns:
        orders_count = df.groupby("day")["Store Order ID"].nunique()
    elif "ID" in df.columns:
        orders_count = df.groupby("day")["ID"].nunique()
    else:
        orders_count = df.groupby("day").size()

    g = pd.DataFrame({"orders_count": orders_count}).reset_index()

    # Aggregations
    def _sum(col):
        return df.groupby("day")[col].sum() if col in df.columns else None

    for col, out_col in [
        ("orders.export.cashOnDelivery", "cod_iqd"),
        ("Order Profit", "profit_iqd"),
        ("VAT Profit", "vat_profit_iqd"),
        ("Shipping Cost", "shipping_iqd"),
        ("Quantities", "items_qty"),
    ]:
        s_col = _sum(col)
        if s_col is not None:
            g[out_col] = s_col.values
        else:
            g[out_col] = 0.0

    # Status counts (wide)
    if "Status" in df.columns:
        status_pivot = (
            df.pivot_table(index="day", columns="Status", values="ID" if "ID" in df.columns else "Customer Name",
                           aggfunc="count", fill_value=0)
            .reset_index()
        )
        g = pd.merge(g, status_pivot, on="day", how="left").fillna(0)

    # Ads spend by day
    spend_usd_by_day = None
    if campaigns_df is not None and not campaigns_df.empty and "Reporting starts" in campaigns_df.columns:
        c = campaigns_df.copy()
        c["Reporting starts"] = pd.to_datetime(c["Reporting starts"], errors="coerce")
        c = c.dropna(subset=["Reporting starts"])
        c["day"] = c["Reporting starts"].dt.floor("D")
        if "Amount spent (USD)" in c.columns:
            c["Amount spent (USD)"] = to_num(c["Amount spent (USD)"])
            spend_usd_by_day = c.groupby("day")["Amount spent (USD)"].sum().reset_index().rename(columns={"Amount spent (USD)": "spend_usd"})

    if spend_usd_by_day is None:
        g["spend_usd"] = 0.0
    else:
        g = pd.merge(g, spend_usd_by_day, on="day", how="left").fillna({"spend_usd": 0.0})

    # Net after ads (profit - spend), displayed in selected currency
    g["profit_usd"] = g["profit_iqd"].apply(lambda v: iqd_to_usd(v, fx))
    g["net_usd"] = g["profit_usd"] - g["spend_usd"]

    if currency == "IQD":
        g["profit_disp"] = g["profit_iqd"]
        g["spend_disp"] = g["spend_usd"] * fx
        g["net_disp"] = g["profit_disp"] - g["spend_disp"]
    else:
        g["profit_disp"] = g["profit_usd"]
        g["spend_disp"] = g["spend_usd"]
        g["net_disp"] = g["net_usd"]

    # Extra useful ratios
    g["profit_per_order_disp"] = g["profit_disp"] / g["orders_count"].replace({0: pd.NA})
    g["net_per_order_disp"] = g["net_disp"] / g["orders_count"].replace({0: pd.NA})

    return g.sort_values("day")

def build_daily_table(
    daily_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    fx_iqd_per_usd: float,
    currency: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Daily table for a selected month/year.
    Shows ALL days in the month (even if no data).

    Columns (base):
    - Date
    - Orders (all)
    - Ad Spend
    - Profit (sum of Order Profit for delivered orders only)
    - Net Profit (Profit - Ad Spend)

    Extra insights:
    - Delivered, Cancelled, Returned
    - Delivery Rate %
    - Avg Profit / Delivered
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    fx = float(fx_iqd_per_usd) if fx_iqd_per_usd else 0.0

    df = parse_daily_orders(daily_df)
    if "day" not in df.columns:
        return pd.DataFrame()

    # Month window
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    days = pd.date_range(start, end, freq="D")

    # Filter daily orders to the month
    dfm = df[(df["day"] >= start) & (df["day"] <= end)].copy()

    # Orders count (nunique if possible), excluding "Cancelled by You"
    id_col = get_daily_order_id_col(dfm)

    if "Status" in dfm.columns:
        _status_all = dfm["Status"].astype(str).str.strip().str.lower()
        _exclude = _status_all.str.contains("cancelled by you")
        dfm_orders = dfm[~_exclude].copy()
    else:
        dfm_orders = dfm

    if id_col:
        orders = dfm_orders.groupby("day")[id_col].nunique()
    else:
        orders = dfm_orders.groupby("day").size()

    # Profit should be counted ONLY for delivered orders (realized profit)
    if "Order Profit" in dfm.columns and "Status" in dfm.columns:
        _status = dfm["Status"].astype(str).str.strip().str.lower()
        _del_mask = _status.str.contains("delivered")
        profit_iqd = dfm[_del_mask].groupby("day")["Order Profit"].sum()
    elif "Order Profit" in dfm.columns:
        # Fallback: if Status is missing, use all rows
        profit_iqd = dfm.groupby("day")["Order Profit"].sum()
    else:
        profit_iqd = 0

    cod_iqd = dfm.groupby("day")["orders.export.cashOnDelivery"].sum() if "orders.export.cashOnDelivery" in dfm.columns else 0
    ship_iqd = dfm.groupby("day")["Shipping Cost"].sum() if "Shipping Cost" in dfm.columns else 0

    base = pd.DataFrame({
        "day": orders.index,
        "Orders": orders.values,
        "Profit_IQD": profit_iqd.reindex(orders.index, fill_value=0).values if hasattr(profit_iqd, "reindex") else 0,
        "COD_IQD": cod_iqd.reindex(orders.index, fill_value=0).values if hasattr(cod_iqd, "reindex") else 0,
        "Shipping_IQD": ship_iqd.reindex(orders.index, fill_value=0).values if hasattr(ship_iqd, "reindex") else 0,
    })

    # Status breakdown (keep a few common statuses as extra insights)
    if "Status" in dfm.columns and not dfm["Status"].isna().all():
        s = dfm.copy()
        s["Status"] = s["Status"].astype(str).str.strip()
        status_counts = s.pivot_table(index="day", columns="Status", values=id_col if id_col else "Status", aggfunc="count", fill_value=0)
        # Normalize some common buckets (best-effort)
        def _sum_cols_like(patterns):
            cols = []
            for p in patterns:
                cols += [c for c in status_counts.columns if p in str(c).lower()]
            cols = list(dict.fromkeys(cols))
            if not cols:
                return pd.Series(0, index=status_counts.index)
            return status_counts[cols].sum(axis=1)

        delivered = _sum_cols_like(["delivered"])
        # Cancelled bucket includes "Delivery Failed" as well
        cancelled = _sum_cols_like(["cancel", "delivery failed"])
        returned = _sum_cols_like(["return", "returned"])
        temp_suspended = _sum_cols_like(["temporary suspended"])
        in_progress = _sum_cols_like(["order received", "delivery in progress"])

        base = base.merge(delivered.rename("Delivered"), left_on="day", right_index=True, how="left")
        base = base.merge(cancelled.rename("Cancelled"), left_on="day", right_index=True, how="left")
        base = base.merge(returned.rename("Returned"), left_on="day", right_index=True, how="left")
        base = base.merge(temp_suspended.rename("Temporary Suspended"), left_on="day", right_index=True, how="left")
        base = base.merge(in_progress.rename("In Progress"), left_on="day", right_index=True, how="left")
    else:
        base["Delivered"] = 0
        base["Cancelled"] = 0
        base["Returned"] = 0
        base["Temporary Suspended"] = 0
        base["In Progress"] = 0

    # Ads spend per day (Meta export)
    spend = pd.DataFrame({"day": [], "Ad_Spend_USD": []})
    if campaigns_df is not None and not getattr(campaigns_df, "empty", True):
        c = campaigns_df.copy()
        if "Reporting starts" in c.columns:
            c["Reporting starts"] = pd.to_datetime(c["Reporting starts"], errors="coerce")
            c["day"] = c["Reporting starts"].dt.floor("D")
        elif "Date" in c.columns:
            c["day"] = pd.to_datetime(c["Date"], errors="coerce").dt.floor("D")
        if "Amount spent (USD)" in c.columns:
            c["Amount spent (USD)"] = pd.to_numeric(c["Amount spent (USD)"], errors="coerce").fillna(0)
            spend = c.groupby("day", as_index=False).agg(Ad_Spend_USD=("Amount spent (USD)", "sum"))
        spend = spend[(spend["day"] >= start) & (spend["day"] <= end)].copy()

    # Ensure full calendar days
    cal = pd.DataFrame({"day": days})
    out = cal.merge(base, on="day", how="left")
    out = out.merge(spend, on="day", how="left")
    out[["Orders", "Profit_IQD", "COD_IQD", "Shipping_IQD", "Delivered", "Cancelled", "Returned", "Temporary Suspended", "In Progress"]] = out[
        ["Orders", "Profit_IQD", "COD_IQD", "Shipping_IQD", "Delivered", "Cancelled", "Returned", "Temporary Suspended", "In Progress"]
    ].fillna(0)
    out["Ad_Spend_USD"] = out["Ad_Spend_USD"].fillna(0)

    # Cast count columns to int (avoid 0.000000 display)
    for _c in ["Orders", "Delivered", "Cancelled", "Returned", "Temporary Suspended", "In Progress"]:
        if _c in out.columns:
            out[_c] = pd.to_numeric(out[_c], errors="coerce").fillna(0).round(0).astype(int)

    # Currency conversion
    if currency.upper() == "USD":
        out["Ad Spend"] = out["Ad_Spend_USD"]
        out["Profit"] = out["Profit_IQD"] / fx if fx > 0 else np.nan
        out["Net Profit"] = out["Profit"] - out["Ad Spend"]
        out["Avg Profit / Delivered"] = np.where(out["Delivered"] > 0, out["Profit"] / out["Delivered"], 0.0)
    else:
        out["Ad Spend"] = out["Ad_Spend_USD"] * fx
        out["Profit"] = out["Profit_IQD"]
        out["Net Profit"] = out["Profit"] - out["Ad Spend"]
        out["Avg Profit / Delivered"] = np.where(out["Delivered"] > 0, out["Profit"] / out["Delivered"], 0.0)

    out["Delivery Rate %"] = np.where(out["Orders"] > 0, (out["Delivered"] / out["Orders"]) * 100, 0.0)

    out["Date"] = out["day"].dt.date
    # Final columns (keep a clean table)
    cols = [
        "Date",
        "Orders",
        "Delivered",
        "Cancelled",
        "Returned",
        "Temporary Suspended",
        "In Progress",
        "Ad Spend",
        "Profit",
        "Net Profit",
        "Avg Profit / Delivered",
        "Delivery Rate %",
    ]
    return out[cols]



def _split_list_cell(val) -> list:
    """Split Taager cells like 'SKU1, SKU2' into a list of strings."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    # Taager exports often separate with commas
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _explode_order_lines(dfm: pd.DataFrame) -> pd.DataFrame:
    """
    Explode Taager daily orders into order lines by SKU with quantities.
    Allocates order-level profit (Order Profit) across SKUs proportionally by quantity.
    """
    out_rows = []
    id_col = get_daily_order_id_col(dfm)

    for _, r in dfm.iterrows():
        order_id = r.get(id_col) if id_col else None
        skus = _split_list_cell(r.get("SKUs"))
        qtys = _split_list_cell(r.get("Quantities"))
        # quantities might be numeric or missing
        q = []
        for x in qtys:
            try:
                q.append(float(str(x).replace(",", "").strip()))
            except:
                q.append(np.nan)

        if not skus:
            continue

        # If quantities missing/mismatch, assume 1 for each SKU
        if len(q) != len(skus) or all(pd.isna(v) for v in q):
            q = [1.0] * len(skus)
        q = [1.0 if pd.isna(v) else float(v) for v in q]
        total_q = sum(q) if sum(q) > 0 else 1.0

        order_profit = r.get("Order Profit", 0.0)
        try:
            order_profit = float(order_profit)
        except:
            order_profit = 0.0

        for sku, qty in zip(skus, q):
            out_rows.append({
                "day": r.get("day"),
                "order_id": order_id,
                "sku": sku,
                "qty": qty,
                "profit_iqd_alloc": order_profit * (qty / total_q),
                "Status": r.get("Status"),
            })

    return pd.DataFrame(out_rows)



def _daily_meta_spend_usd(campaigns_df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Return Meta spend per day (USD) for the given month, indexed by day (Timestamp)."""
    if campaigns_df is None or getattr(campaigns_df, "empty", True):
        return pd.Series(dtype=float)

    df = campaigns_df.copy()
    if "Reporting starts" not in df.columns or "Amount spent (USD)" not in df.columns:
        return pd.Series(dtype=float)

    df["Reporting starts"] = pd.to_datetime(df["Reporting starts"], errors="coerce")
    df = df.dropna(subset=["Reporting starts"])
    df["day"] = df["Reporting starts"].dt.floor("D")

    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    dfm = df[(df["day"] >= start) & (df["day"] <= end)].copy()

    dfm["Amount spent (USD)"] = pd.to_numeric(dfm["Amount spent (USD)"], errors="coerce").fillna(0)
    spend = dfm.groupby("day")["Amount spent (USD)"].sum()
    return spend


def build_product_by_date_table(
    daily_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    fx_iqd_per_usd: float,
    currency: str,
    year: int,
    month: int,
    selected_skus: list[str],
) -> pd.DataFrame:
    """
    Daily table for selected product(s) (SKU) in a selected month/year.
    Shows ALL days in the month (even if no data).

    - Orders: count of orders containing selected SKU(s), excluding 'Cancelled by You'
    - Profit: delivered-only allocated profit for selected SKU(s)
    - Ad Spend: allocated by share of Orders that day (selected orders / total orders)
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    fx = float(fx_iqd_per_usd) if fx_iqd_per_usd else 0.0

    df = parse_daily_orders(daily_df)
    if "day" not in df.columns:
        return pd.DataFrame()

    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    days = pd.date_range(start, end, freq="D")

    dfm = df[(df["day"] >= start) & (df["day"] <= end)].copy()

    # Status normalization + exclusion mask
    status_all = dfm["Status"].astype(str).str.strip().str.lower() if "Status" in dfm.columns else pd.Series("", index=dfm.index)
    excl_cancelled_by_you = status_all.str.contains("cancelled by you") if "Status" in dfm.columns else pd.Series(False, index=dfm.index)

    id_col = get_daily_order_id_col(dfm)
    if id_col is None:
        # Worst-case fallback
        dfm["__rowid__"] = np.arange(len(dfm))
        id_col = "__rowid__"

    # Total orders per day (for spend allocation)
    dfm_orders_all = dfm[~excl_cancelled_by_you].copy()
    total_orders_per_day = dfm_orders_all.groupby("day")[id_col].nunique()

    # Explode to SKU lines for product filtering
    lines = _explode_order_lines(dfm)
    if lines.empty:
        # Return empty calendar table
        out = pd.DataFrame({"Date": days.date})
        for c in ["Orders", "Delivered", "Cancelled", "Returned", "Temporary Suspended", "In Progress", "Ad Spend", "Profit", "Net Profit", "Avg Profit / Delivered", "Delivery Rate %"]:
            out[c] = 0
        return out

    lines["sku"] = lines["sku"].astype(str).str.strip()
    # Normalize selected SKUs (sometimes a cell contains 'SKU1, SKU2' so split on commas)
    _flat = []
    for _s in (selected_skus or []):
        if _s is None:
            continue
        for part in str(_s).split(","):
            p = part.strip()
            if p:
                _flat.append(p)
    sel_set = set(_flat)
    if sel_set:
        lines = lines[lines["sku"].astype(str).str.strip().isin(sel_set)].copy()

    # Merge exclusion + status flags at order level (use dfm for order_id -> day/status)
    ord_map = dfm[[ "day", id_col, "Status"]].copy()
    ord_map = ord_map.rename(columns={id_col: "order_id"})
    ord_map["status_clean"] = ord_map["Status"].astype(str).str.strip().str.lower()
    ord_map["exclude_by_you"] = ord_map["status_clean"].str.contains("cancelled by you")

    # status flags
    ord_map["is_delivered"] = ord_map["status_clean"].str.contains("delivered")
    ord_map["is_cancelled"] = ord_map["status_clean"].str.contains("cancel") | ord_map["status_clean"].str.contains("delivery failed")
    ord_map["is_returned"] = ord_map["status_clean"].str.contains("return")
    ord_map["is_temp_suspended"] = ord_map["status_clean"].str.contains("temporary suspended")
    ord_map["is_in_progress"] = ord_map["status_clean"].str.contains("order received") | ord_map["status_clean"].str.contains("delivery in progress")

    lines = lines.merge(ord_map[["order_id","day","exclude_by_you","is_delivered","is_cancelled","is_returned","is_temp_suspended","is_in_progress"]], on=["order_id","day"], how="left")

    # Orders per day for selected SKU(s), excluding Cancelled by You
    lines_ok = lines[~lines["exclude_by_you"].fillna(False)].copy()
    orders_sel = lines_ok.groupby("day")["order_id"].nunique()

    # Delivered profit (allocated) per day for selected SKU(s)
    profit_iqd_sel = lines_ok[lines_ok["is_delivered"].fillna(False)].groupby("day")["profit_iqd_alloc"].sum()

    # Status breakdown (count unique orders) for selected SKU(s)
    delivered_sel = lines_ok[lines_ok["is_delivered"].fillna(False)].groupby("day")["order_id"].nunique()
    cancelled_sel = lines_ok[lines_ok["is_cancelled"].fillna(False)].groupby("day")["order_id"].nunique()
    returned_sel = lines_ok[lines_ok["is_returned"].fillna(False)].groupby("day")["order_id"].nunique()
    temp_sel = lines_ok[lines_ok["is_temp_suspended"].fillna(False)].groupby("day")["order_id"].nunique()
    inprog_sel = lines_ok[lines_ok["is_in_progress"].fillna(False)].groupby("day")["order_id"].nunique()

    # Ad spend per day (USD), then allocate by share of orders
    spend_usd = _daily_meta_spend_usd(campaigns_df, year, month)
    # Align index to Timestamp days
    share = (orders_sel / total_orders_per_day).replace([np.inf, -np.inf], np.nan).fillna(0)
    spend_alloc_usd = spend_usd.mul(share, fill_value=0)

    # Build calendar table
    out = pd.DataFrame({"Date": days.date})
    out["day"] = pd.to_datetime(out["Date"])

    def _series_to_col(s, col):
        if isinstance(s, (int, float)):
            out[col] = 0
        else:
            out[col] = out["day"].map(s).fillna(0)

    _series_to_col(orders_sel, "Orders")
    _series_to_col(delivered_sel, "Delivered")
    _series_to_col(cancelled_sel, "Cancelled")
    _series_to_col(returned_sel, "Returned")
    _series_to_col(temp_sel, "Temporary Suspended")
    _series_to_col(inprog_sel, "In Progress")

    # Profit / Spend / Net in currency
    profit_usd = out["day"].map(profit_iqd_sel).fillna(0) / fx if fx > 0 else 0
    out["Ad Spend"] = out["day"].map(spend_alloc_usd).fillna(0)

    if currency == "IQD":
        out["Profit"] = profit_usd * fx
        out["Ad Spend"] = out["Ad Spend"] * fx
    else:
        out["Profit"] = profit_usd

    out["Net Profit"] = out["Profit"] - out["Ad Spend"]

    # Delivery rate & avg profit
    out["Delivery Rate %"] = np.where(out["Orders"] > 0, (out["Delivered"] / out["Orders"]) * 100, 0)
    out["Avg Profit / Delivered"] = np.where(out["Delivered"] > 0, out["Profit"] / out["Delivered"], 0)

    # Ensure integer counts
    for c in ["Orders","Delivered","Cancelled","Returned","Temporary Suspended","In Progress"]:
        out[c] = out[c].astype(int)

    out = out.drop(columns=["day"])
    cols = [
        "Date",
        "Orders",
        "Delivered",
        "Cancelled",
        "Returned",
        "Temporary Suspended",
        "In Progress",
        "Ad Spend",
        "Profit",
        "Net Profit",
        "Avg Profit / Delivered",
        "Delivery Rate %",
    ]
    return out[cols]
def plot_results_top10_active_interactive(campaigns_df: pd.DataFrame):
    df = campaigns_df.copy()

    required = ["Reporting starts", "Campaign name", "Campaign delivery", "Results"]
    for c in required:
        if c not in df.columns:
            st.error(f"Campaigns file missing column: {c}")
            return

    # Parse + clean
    df["Reporting starts"] = pd.to_datetime(df["Reporting starts"], errors="coerce")
    df = df.dropna(subset=["Reporting starts"])
    df["day"] = df["Reporting starts"].dt.floor("D")
    df["Results"] = to_num(df["Results"])
    df["Campaign delivery"] = df["Campaign delivery"].astype(str).str.lower()

    # Filter active
    active = df[df["Campaign delivery"] == "active"].copy()
    if active.empty:
        st.info("No active campaigns found.")
        return

    # Pick top 10 by total results
    top10 = (
        active.groupby("Campaign name", as_index=False)["Results"]
        .sum()
        .sort_values("Results", ascending=False)
        .head(10)
    )
    top_names = top10["Campaign name"].tolist()

    # Daily results for top 10
    daily = (
        active[active["Campaign name"].isin(top_names)]
        .groupby(["day", "Campaign name"], as_index=False)["Results"]
        .sum()
        .rename(columns={"Results": "results"})
        .sort_values("day")
    )

    # ----- UI controls -----
    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("### Campaigns")
        selected = st.multiselect("Select campaigns", top_names, default=top_names[:3] if len(top_names) >= 3 else top_names)

        st.markdown("### Date filter")
        min_d = daily["day"].min().date()
        max_d = daily["day"].max().date()
        d_from, d_to = st.date_input("Range", value=(min_d, max_d))

    if not selected:
        with right:
            st.info("Select at least 1 campaign.")
        return

    # Apply filters
    daily_f = daily[daily["Campaign name"].isin(selected)].copy()
    daily_f = daily_f[(daily_f["day"].dt.date >= d_from) & (daily_f["day"].dt.date <= d_to)]

    if daily_f.empty:
        with right:
            st.info("No data in this date range.")
        return

    with right:
        st.markdown("### Daily Results (interactive)")

        # Interactive selection on hover
        nearest = alt.selection_point(nearest=True, on="mouseover", fields=["day"], empty=False)

        base = alt.Chart(daily_f).encode(
            x=alt.X("day:T", title="Day"),
            y=alt.Y("results:Q", title="Results"),
            color=alt.Color("Campaign name:N", legend=alt.Legend(title="Campaign")),
        )

        lines = base.mark_line(point=True).encode(
            tooltip=[
                alt.Tooltip("day:T", title="Day"),
                alt.Tooltip("Campaign name:N", title="Campaign"),
                alt.Tooltip("results:Q", title="Results"),
            ]
        )

        # Hover point + vertical rule
        selectors = base.mark_point().encode(opacity=alt.value(0)).add_params(nearest)
        points = base.mark_point(size=80).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        rule = alt.Chart(daily_f).mark_rule().encode(x="day:T").transform_filter(nearest)

        chart = (lines + selectors + points + rule).properties(height=420).interactive()

        st.altair_chart(chart, use_container_width=True)

        with st.expander("Top 10 (by total Results)", expanded=False):
            st.dataframe(top10, use_container_width=True)


def money_ccy(x: float, ccy: str) -> str:
    if x is None:
        return "N/A"
    if ccy == "IQD":
        return f"{x:,.0f} IQD"
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def safe_ratio(n: float, d: float):
    return None if d == 0 else n / d

def fmt_ratio(x):
    return "N/A" if x is None else f"{x:.2f}"

def fmt_money_or_na(x):
    return "N/A" if x is None else money_ccy(x, "USD")

def money(x: float) -> str:
    return f"${x:,.2f}"

def iqd_to_usd(x: float, iqd_per_usd: float) -> float:
    return x / iqd_per_usd if iqd_per_usd else 0.0


def parse_inputs(orders: pd.DataFrame, campaigns: pd.DataFrame, iqd_per_usd: float):
    # Rename orders columns from Arabic
    orders = orders.rename(columns={
        "عدد_القطع_المطلوبة": "requested_units",
        "عدد_القطع_المؤكدة": "confirmed_units",
        "عدد_القطع_الموصلة": "delivered_units",
        "عدد_القطع_المرتجعة": "returned_units",
        "القطع_المتوقع_توصيلها": "expected_delivered_units",
        "مجموع_الارباح_المؤكدة": "confirmed_profit_iqd",
        "مجموع_الارباح_الموصلة": "delivered_profit_iqd",
        "مجموع_الارباح_المتوقعة": "expected_profit_iqd",
    })

    needed_orders = [
        "requested_units","confirmed_units","delivered_units","returned_units",
        "expected_delivered_units","confirmed_profit_iqd","delivered_profit_iqd","expected_profit_iqd"
    ]
    for c in needed_orders:
        if c not in orders.columns:
            raise ValueError(f"Orders file missing column: {c}")
        orders[c] = to_num(orders[c])

    orders["confirmed_profit_usd"] = orders["confirmed_profit_iqd"].apply(lambda v: iqd_to_usd(v, iqd_per_usd))
    orders["delivered_profit_usd"] = orders["delivered_profit_iqd"].apply(lambda v: iqd_to_usd(v, iqd_per_usd))
    orders["expected_profit_usd"] = orders["expected_profit_iqd"].apply(lambda v: iqd_to_usd(v, iqd_per_usd))

    needed_campaigns = ["Amount spent (USD)", "Impressions", "Reach"]
    for c in needed_campaigns:
        if c not in campaigns.columns:
            raise ValueError(f"Campaigns file missing column: {c}")
    campaigns["Amount spent (USD)"] = to_num(campaigns["Amount spent (USD)"])
    campaigns["Impressions"] = to_num(campaigns["Impressions"])
    campaigns["Reach"] = to_num(campaigns["Reach"])

    # KPIs
    requested_units = float(orders["requested_units"].sum())
    confirmed_units = float(orders["confirmed_units"].sum())
    delivered_units = float(orders["delivered_units"].sum())
    returned_units = float(orders["returned_units"].sum())
    expected_delivered_units = float(orders["expected_delivered_units"].sum())

    confirmed_profit_usd = float(orders["confirmed_profit_usd"].sum())
    delivered_profit_usd = float(orders["delivered_profit_usd"].sum())
    expected_profit_usd = float(orders["expected_profit_usd"].sum())

    spend_usd = float(campaigns["Amount spent (USD)"].sum())
    impressions = float(campaigns["Impressions"].sum())
    reach = float(campaigns["Reach"].sum())

    net_profit_usd = delivered_profit_usd - spend_usd
    potential_net_profit_usd = confirmed_profit_usd - spend_usd

    confirmation_rate = (confirmed_units / requested_units) if requested_units else 0.0
    delivery_rate = (delivered_units / confirmed_units) if confirmed_units else 0.0
    return_rate = (returned_units / delivered_units) if delivered_units else 0.0

    roas_real = safe_ratio(delivered_profit_usd, spend_usd)
    roas_potential = safe_ratio(confirmed_profit_usd, spend_usd)

    cpm = None if impressions == 0 else (spend_usd / impressions * 1000)
    cost_per_reach = None if reach == 0 else (spend_usd / reach)



    kpis = dict(
        requested_units=requested_units,
        confirmed_units=confirmed_units,
        delivered_units=delivered_units,
        returned_units=returned_units,
        expected_delivered_units=expected_delivered_units,
        confirmed_profit_usd=confirmed_profit_usd,
        delivered_profit_usd=delivered_profit_usd,
        expected_profit_usd=expected_profit_usd,
        spend_usd=spend_usd,
        impressions=impressions,
        reach=reach,
        net_profit_usd=net_profit_usd,
        potential_net_profit_usd=potential_net_profit_usd,
        confirmation_rate=confirmation_rate,
        delivery_rate=delivery_rate,
        return_rate=return_rate,
        roas_real=roas_real,
        roas_potential=roas_potential,
        cpm=cpm,
        cost_per_reach=cost_per_reach,
    )

    return orders, campaigns, kpis

def kpis_in_currency(kpis: dict, fx: float, ccy: str) -> dict:
    kk = dict(kpis)  # copy

    if ccy == "IQD":
        kk["confirmed_profit_disp"] = kpis["confirmed_profit_usd"] * fx
        kk["delivered_profit_disp"] = kpis["delivered_profit_usd"] * fx
        kk["expected_profit_disp"]  = kpis["expected_profit_usd"] * fx
        kk["spend_disp"]            = kpis["spend_usd"] * fx

        kk["net_profit_disp"]       = kk["delivered_profit_disp"] - kk["spend_disp"]
        kk["potential_net_disp"]    = kk["confirmed_profit_disp"] - kk["spend_disp"]
    else:
        kk["confirmed_profit_disp"] = kpis["confirmed_profit_usd"]
        kk["delivered_profit_disp"] = kpis["delivered_profit_usd"]
        kk["expected_profit_disp"]  = kpis["expected_profit_usd"]
        kk["spend_disp"]            = kpis["spend_usd"]

        kk["net_profit_disp"]       = kpis["net_profit_usd"]
        kk["potential_net_disp"]    = kpis["potential_net_profit_usd"]

    return kk

def make_charts_bytes(kpis: dict):
    # Return PNG bytes for 3 charts
    def fig_to_png_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    # Funnel
    fig1 = plt.figure(figsize=(7.0, 3.0))
    plt.bar(["Requested", "Confirmed", "Delivered"], [kpis["requested_units"], kpis["confirmed_units"], kpis["delivered_units"]])
    plt.title("Order Funnel")
    plt.ylabel("Units")
    plt.tight_layout()
    funnel_png = fig_to_png_bytes(fig1)
    plt.close(fig1)

    # Realized
    fig2 = plt.figure(figsize=(7.0, 3.0))
    plt.bar(["Delivered Profit", "Ad Spend", "Net Profit"], [kpis["delivered_profit_usd"], kpis["spend_usd"], kpis["net_profit_usd"]])
    plt.title("Realized Profit (USD)")
    plt.ylabel("USD")
    plt.tight_layout()
    realized_png = fig_to_png_bytes(fig2)
    plt.close(fig2)

    # Potential
    fig3 = plt.figure(figsize=(7.0, 3.0))
    plt.bar(["Confirmed Profit", "Ad Spend", "Potential Net"], [kpis["confirmed_profit_usd"], kpis["spend_usd"], kpis["potential_net_profit_usd"]])
    plt.title("Potential Profit from Confirmed (USD)")
    plt.ylabel("USD")
    plt.tight_layout()
    potential_png = fig_to_png_bytes(fig3)
    plt.close(fig3)

    return funnel_png, realized_png, potential_png


def build_pdf_bytes(kpis: dict, fx: float, funnel_png: bytes, realized_png: bytes, potential_png: bytes) -> bytes:
    # Create PDF in memory
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.4*cm, bottomMargin=1.4*cm)

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, leading=24, spaceAfter=6),
        "muted": ParagraphStyle("muted", parent=base["Normal"], fontSize=10, leading=13, textColor=colors.HexColor("#667085")),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#101828")),
        "card_title": ParagraphStyle("card_title", parent=base["Normal"], fontSize=10, leading=12),
        "card_value": ParagraphStyle("card_value", parent=base["Normal"], fontSize=18, leading=20, textColor=colors.HexColor("#101828")),
        "card_sub": ParagraphStyle("card_sub", parent=base["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#667085")),
        "insight": ParagraphStyle("insight", parent=base["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#101828")),
    }

    def build_card(title, value, subtitle="", accent="neutral"):
        accent_map = {
            "good": colors.HexColor("#0B6B3A"),
            "bad": colors.HexColor("#B42318"),
            "neutral": colors.HexColor("#344054"),
            "info": colors.HexColor("#175CD3"),
        }
        accent_color = accent_map.get(accent, accent_map["neutral"])

        content = [
            Paragraph(f"<font color='{accent_color.hexval()}'><b>{title}</b></font>", styles["card_title"]),
            Spacer(1, 4),
            Paragraph(f"<b>{value}</b>", styles["card_value"]),
        ]
        if subtitle:
            content += [Spacer(1, 3), Paragraph(subtitle, styles["card_sub"])]

        t = Table([[content]], colWidths=[9.0 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D0D5DD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return t

    def section_title(t):
        return Paragraph(t, styles["section"])

    elements = []
    elements.append(Paragraph("E-commerce Dashboard", styles["title"]))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  FX: 1 USD = {fx:,.0f} IQD",
        styles["muted"]
    ))
    elements.append(Spacer(1, 10))

    realized_accent = "good" if kpis["net_profit_usd"] >= 0 else "bad"
    potential_accent = "good" if kpis["potential_net_profit_usd"] >= 0 else "bad"

    cards = Table(
        [[
            build_card("Confirmed Profit (USD)", money(kpis["confirmed_profit_usd"]), f"{int(kpis['confirmed_units']):,} confirmed orders", "info"),
            build_card("Delivered Profit (USD)", money(kpis["delivered_profit_usd"]), f"{int(kpis['delivered_units']):,} delivered orders", "info"),
        ],
         [
            build_card("Ad Spend (USD)", money(kpis["spend_usd"]), "Meta spend", "neutral"),
            build_card("Net Profit After Ads", money(kpis["net_profit_usd"]), "Delivered − spend", realized_accent),
         ],
         [
            build_card("Potential Net Profit", money(kpis["potential_net_profit_usd"]), "Confirmed − spend", potential_accent),
            build_card("ROAS (Realized)", fmt_ratio(kpis["roas_real"]), f"Potential ROAS: {fmt_ratio(kpis['roas_potential'])}", "neutral"),
         ]],
        colWidths=[9.2*cm, 9.2*cm],
        hAlign="LEFT"
    )
    cards.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(cards)

    elements.append(section_title("Operations & Ads Health"))
    health_lines = [
        f"• Confirmation rate: <b>{pct(kpis['confirmation_rate'])}</b>",
        f"• Delivery rate: <b>{pct(kpis['delivery_rate'])}</b>",
        f"• Return rate: <b>{pct(kpis['return_rate'])}</b>",
        f"• CPM: <b>{fmt_money_or_na(kpis['cpm'])}</b> • Cost/reach: <b>{fmt_money_or_na(kpis['cost_per_reach'])}</b>",
        f"• Expected profit (USD): <b>{money(kpis['expected_profit_usd'])}</b>",
    ]
    elements.append(Paragraph("<br/>".join(health_lines), styles["insight"]))

    elements.append(PageBreak())
    elements.append(Paragraph("Charts", styles["title"]))
    elements.append(Paragraph("Order funnel and profit views (USD).", styles["muted"]))
    elements.append(Spacer(1, 10))

    # Insert charts (from bytes)
    def add_png(png_bytes, title):
        elements.append(section_title(title))
        img = RLImage(io.BytesIO(png_bytes))
        # scale
        iw, ih = ImageReader(io.BytesIO(png_bytes)).getSize()
        page_width = A4[0] - (1.6*cm) - (1.6*cm)
        max_h = 7.2*cm
        scale = min(page_width / iw, max_h / ih)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        elements.append(img)
        elements.append(Spacer(1, 10))

    add_png(funnel_png, "Order Funnel")
    add_png(realized_png, "Realized Profit")
    add_png(potential_png, "Potential Profit (Confirmed)")

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


def build_excel_bytes(kpis: dict, fx: float, funnel_png: bytes, realized_png: bytes, potential_png: bytes) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Dashboard")
        writer.sheets["Dashboard"] = ws

        title_fmt = wb.add_format({"bold": True, "font_size": 18})
        sub_fmt = wb.add_format({"font_size": 10, "font_color": "#667085"})
        card_title = wb.add_format({"bold": True, "font_color": "#175CD3"})
        card_value = wb.add_format({"bold": True, "font_size": 14})
        box = wb.add_format({"border": 1, "border_color": "#D0D5DD", "bg_color": "#FFFFFF"})
        money_fmt = wb.add_format({"num_format": "$#,##0.00"})
        pct_fmt = wb.add_format({"num_format": "0.0%"})

        ws.set_column("A:A", 2)
        ws.set_column("B:K", 18)

        ws.write("B2", "E-commerce Dashboard", title_fmt)
        ws.write("B3", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  FX: 1 USD = {fx:,.0f} IQD", sub_fmt)

        def card(r, c, title, value, value_format=None, subtitle=None):
            for rr in range(r, r+4):
                for cc in range(c, c+3):
                    ws.write_blank(rr, cc, None, box)
            ws.write(r, c, title, card_title)
            if value_format:
                ws.write(r+1, c, value, value_format)
            else:
                ws.write(r+1, c, value, card_value)
            if subtitle:
                ws.write(r+2, c, subtitle, sub_fmt)

        card(4, 1, "Confirmed Profit (USD)", kpis["confirmed_profit_usd"], money_fmt, f"{int(kpis['confirmed_units']):,} confirmed")
        card(4, 5, "Delivered Profit (USD)", kpis["delivered_profit_usd"], money_fmt, f"{int(kpis['delivered_units']):,} delivered")

        card(9, 1, "Ad Spend (USD)", kpis["spend_usd"], money_fmt, "Meta spend")
        card(9, 5, "Net Profit After Ads", kpis["net_profit_usd"], money_fmt, "Delivered − spend")

        card(14, 1, "Potential Net Profit", kpis["potential_net_profit_usd"], money_fmt, "Confirmed − spend")
        card(14, 5, "ROAS (Realized)", fmt_ratio(kpis["roas_real"]), None, f"Potential ROAS: {fmt_ratio(kpis['roas_potential'])}")

        ws.write("B20", "Health", wb.add_format({"bold": True, "font_size": 12}))
        ws.write("B21", "Confirmation Rate", card_title); ws.write("C21", kpis["confirmation_rate"], pct_fmt)
        ws.write("B22", "Delivery Rate", card_title); ws.write("C22", kpis["delivery_rate"], pct_fmt)
        ws.write("B23", "Return Rate", card_title); ws.write("C23", kpis["return_rate"], pct_fmt)
        ws.write("B24", "Expected Profit (USD)", card_title); ws.write("C24", kpis["expected_profit_usd"], money_fmt)

        # Charts sheet
        ws2 = wb.add_worksheet("Charts")
        writer.sheets["Charts"] = ws2
        ws2.write("A1", "Charts", title_fmt)
        ws2.write("A2", "Funnel + realized + potential (USD).", sub_fmt)

        # Insert images from bytes
        ws2.insert_image("A4", "funnel.png", {"image_data": io.BytesIO(funnel_png), "x_scale": 0.9, "y_scale": 0.9})
        ws2.insert_image("A22", "realized.png", {"image_data": io.BytesIO(realized_png), "x_scale": 0.9, "y_scale": 0.9})
        ws2.insert_image("A40", "potential.png", {"image_data": io.BytesIO(potential_png), "x_scale": 0.9, "y_scale": 0.9})

    out.seek(0)
    return out.getvalue()


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
:root{
  --bg-0:#0b1116;
  --bg-1:#111923;
  --bg-2:#172331;
  --card:rgba(15,23,34,0.78);
  --card-strong:rgba(12,20,30,0.92);
  --stroke:rgba(154,182,207,0.20);
  --text:#eaf2f8;
  --muted:#a9bbc9;
  --good:#31c48d;
  --bad:#ff6a79;
  --accent:#5ec8ff;
}

html, body, [class*="css"]  { font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif; }
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% -5%, rgba(94,200,255,0.18), transparent 58%),
    radial-gradient(900px 500px at 95% 10%, rgba(49,196,141,0.12), transparent 52%),
    linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 45%, var(--bg-2) 100%);
  color: var(--text);
}
[data-testid="stHeader"]{ background: transparent; }
[data-testid="block-container"]{
  max-width: 1320px;
  padding-top: 1.2rem;
  padding-bottom: 4rem;
}
[data-testid="stSidebar"]{
  background: linear-gradient(175deg, rgba(9,15,22,0.98), rgba(12,19,27,0.92));
  border-right: 1px solid var(--stroke);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label{
  color: var(--muted) !important;
}

/* Hero */
.hero-wrap{
  border:1px solid var(--stroke);
  border-radius:18px;
  padding:18px 20px;
  background: linear-gradient(120deg, rgba(24,35,50,0.86), rgba(12,20,30,0.72));
  box-shadow: 0 16px 40px rgba(0,0,0,0.25);
  margin-bottom: 14px;
}
.hero-title{
  font-size: 1.65rem;
  letter-spacing: -0.02em;
  font-weight: 800;
  color: var(--text);
}
.hero-sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.95rem;
}

/* Tabs */
[data-testid="stTabs"] button[role="tab"]{
  border-radius: 12px;
  border: 1px solid transparent;
  padding: 8px 12px;
  color: var(--muted);
  background: rgba(255,255,255,0.02);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  color: var(--text);
  border-color: var(--stroke);
  background: rgba(94,200,255,0.10);
}

/* Inputs & buttons */
.stButton > button, .stDownloadButton > button{
  border-radius: 12px;
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, rgba(94,200,255,0.24), rgba(94,200,255,0.10));
  color: var(--text);
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stDateInput input{
  border-radius: 10px !important;
  border-color: var(--stroke) !important;
  background: rgba(255,255,255,0.02) !important;
  color: var(--text) !important;
}

/* AI panel */
.ai-panel{
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(12,20,30,0.55);
  margin: 4px 0 10px 0;
}
.ai-panel p{
  margin: 0;
  color: var(--muted);
  font-size: 0.92rem;
}
.ai-banner{
  border: 1px solid rgba(72,139,255,0.35);
  background: rgba(40,86,170,0.20);
  border-radius: 12px;
  padding: 10px 14px;
  color: #c8defd;
  margin: 4px 0 14px 0;
}
.ai-card{
  border: 1px solid var(--stroke);
  background: rgba(22,32,48,0.72);
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  margin: 10px 0 14px 0;
}
.ai-card-title{
  color: var(--text);
  font-size: 1.1rem;
  font-weight: 800;
  margin-bottom: 2px;
}
.ai-card-sub{
  color: var(--muted);
  font-size: 0.92rem;
  margin-bottom: 14px;
}
.ai-grid-4{
  display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 14px;
}
.ai-grid-3{
  display:grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 14px;
}
.ai-divider{
  height:1px;
  background: rgba(255,255,255,0.08);
  margin: 14px 0;
}
.ai-k-label{
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 2px;
}
.ai-k-value{
  color: var(--text);
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.1;
}
.ai-k-sub{
  color: rgba(180,198,215,0.72);
  font-size: 0.82rem;
  margin-top: 3px;
}
.ai-delta-pos{ color:#1fe58f; font-weight:700; }
.ai-delta-neg{ color:#ff7f8e; font-weight:700; }
.ai-status-pill{
  display:inline-block;
  border-radius: 999px;
  border: 1px solid rgba(34,197,94,0.35);
  background: rgba(18,95,61,0.38);
  color: #2df0a0;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 700;
}
.ai-empty{
  border: 1px dashed rgba(180,198,215,0.22);
  border-radius: 12px;
  min-height: 120px;
  display:flex;
  align-items:center;
  justify-content:center;
  color: var(--muted);
  background: rgba(9,15,22,0.42);
}
.ai-assistant-wrap{
  border: 1px solid var(--stroke);
  background: rgba(22,32,48,0.72);
  border-radius: 16px;
  padding: 18px;
  margin: 10px 0 14px 0;
}
.ai-assistant-wrap .stTextArea textarea{
  min-height: 140px !important;
  border-radius: 12px !important;
  border: 1px solid rgba(140,168,196,0.26) !important;
  background: rgba(15,22,36,0.86) !important;
}
.ai-assistant-wrap [data-testid="stExpander"]{
  border: 1px solid rgba(140,168,196,0.24);
  border-radius: 12px;
  background: rgba(8,14,26,0.28);
}
button.ai-btn-main,
button.ai-btn-chip,
button.ai-btn-ask,
button.ai-btn-clear{
  min-height: 52px !important;
  border-radius: 16px !important;
  border: 1px solid rgba(110,166,216,0.38) !important;
  background: linear-gradient(180deg, rgba(39,83,117,0.92), rgba(27,54,78,0.88)) !important;
  color: #eaf2f8 !important;
  font-weight: 700 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
button.ai-btn-main:hover,
button.ai-btn-chip:hover,
button.ai-btn-ask:hover,
button.ai-btn-clear:hover{
  border-color: rgba(130,191,245,0.55) !important;
  background: linear-gradient(180deg, rgba(45,95,132,0.96), rgba(31,64,91,0.94)) !important;
}
button.ai-btn-ask{
  background: linear-gradient(180deg, rgba(120,58,196,0.95), rgba(89,41,150,0.92)) !important;
  border-color: rgba(170,111,245,0.52) !important;
}
button.ai-btn-ask:hover{
  background: linear-gradient(180deg, rgba(136,68,221,0.98), rgba(101,47,171,0.96)) !important;
  border-color: rgba(187,134,255,0.62) !important;
}
button.ai-btn-chip{
  min-height: 48px !important;
}
button.ai-btn-clear{
  min-height: 46px !important;
  max-width: 220px !important;
}
.ai-output-wrap{
  white-space: pre-wrap;
  line-height: 1.65;
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 14px;
  background: rgba(9,15,22,0.48);
}

/* Floating Quick KPIs expander */
div.kpi-fixed-expander {
  position: fixed !important;
  right: 16px;
  bottom: 16px;
  width: 360px;
  z-index: 9999;
}
div.kpi-fixed-expander details {
  background: var(--card-strong);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.35);
  backdrop-filter: blur(8px);
}
div.kpi-fixed-expander summary { padding: 10px 12px !important; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 4px; }
.kpi-card {
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
  background: var(--card);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.kpi-title { font-size: 12px; color: var(--muted); margin-bottom: 8px; font-weight:600; }
.kpi-value { font-size: 28px; font-weight: 800; line-height: 1.1; margin-bottom: 6px; letter-spacing: -0.01em; }
.kpi-sub { font-size: 12px; color: var(--muted); }
.good { color: var(--good); }
.bad  { color: var(--bad); }
.neutral { color: var(--text); }
.kpi-title-row{ display:flex; align-items:center; justify-content:space-between; gap:8px; }
.kpi-tip{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:18px;
  height:18px;
  border-radius:50%;
  border:1px solid var(--stroke);
  color: var(--muted);
  font-size:12px;
  cursor: help;
  position: relative;
  flex: 0 0 auto;
}
.kpi-tip:hover{ border-color: rgba(255,255,255,0.40); color: var(--text); }
.kpi-tip[data-tip]:hover:after{
  content: attr(data-tip);
  position:absolute;
  left:50%;
  transform: translateX(-50%);
  bottom: 130%;
  width: 280px;
  max-width: 320px;
  background: rgba(8,12,18,0.98);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.35;
  color: var(--text);
  white-space: pre-wrap;
  z-index: 9999;
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}
.kpi-tip[data-tip]:hover:before{
  content:"";
  position:absolute;
  left:50%;
  transform: translateX(-50%);
  bottom: 118%;
  border-width: 7px;
  border-style: solid;
  border-color: rgba(8,12,18,0.98) transparent transparent transparent;
}

@media (max-width: 980px){
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .ai-grid-4 { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .ai-grid-3 { grid-template-columns: repeat(2, minmax(0,1fr)); }
  div.kpi-fixed-expander { width: calc(100vw - 24px); right: 12px; bottom: 12px; }
}
@media (max-width: 640px){
  .hero-title{ font-size: 1.25rem; }
  .kpi-grid { grid-template-columns: 1fr; }
  .ai-grid-4, .ai-grid-3 { grid-template-columns: 1fr; }
}
    </style>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-title">E-commerce Performance Console</div>
      <div class="hero-sub">Upload Orders + Campaigns to track profitability, delivery health, and ad efficiency in one place.</div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Inputs")
    fx = st.number_input("FX rate (IQD per 1 USD)", min_value=1.0, value=1310.0, step=1.0)
    currency = st.selectbox("Display currency (Orders)", ["USD", "IQD"], index=0)
    orders_file = st.file_uploader("Orders CSV (Taager File)", type=["csv"])
    campaigns_file = st.file_uploader("Campaigns CSV (Meta export)", type=["csv"])
    daily_orders_file = st.file_uploader("Daily Orders (Taager) XLSX", type=["xlsx"])

both_uploaded = orders_file is not None and campaigns_file is not None
one_uploaded = (orders_file is not None) ^ (campaigns_file is not None)

orders_df = None
campaigns_df = None
snap = None
daily_orders_df = None




# --- Taager FX (always defined) ---
TAAGER_FX = 1602.0
net_profit_usd_taager = None
potential_net_usd_taager = None

# CASE A: both uploaded -> use uploads
if both_uploaded:
    try:
        orders_df = pd.read_csv(orders_file, encoding="utf-8-sig")
        campaigns_df = pd.read_csv(campaigns_file, encoding="utf-8-sig")
        if daily_orders_file is not None:
            daily_orders_df = pd.read_excel(daily_orders_file)
        orders_df, campaigns_df, kpis = parse_inputs(orders_df, campaigns_df, fx)

        data_source = "uploads"
        kpis_disp = kpis_in_currency(kpis, fx, currency)
        net_profit_usd_taager, potential_net_usd_taager = compute_taager_kpis(orders_df, campaigns_df, TAAGER_FX)


    except Exception as e:
        st.error(str(e))
        st.stop()

# CASE B/C: not both uploaded -> try GitHub
else:
    try:
        snap = load_latest_from_github()
    except Exception as e:
        snap = None
        st.warning(f"Could not load latest data from GitHub: {e}")

    if snap is None:
        st.info("Upload BOTH files once and click **Save latest dashboard to GitHub**. After that, reload will work without uploads.")
        st.stop()

    try:
        orders_df = pd.read_csv(io.BytesIO(snap["orders_csv_bytes"]), encoding="utf-8-sig")
        campaigns_df = pd.read_csv(io.BytesIO(snap["campaigns_csv_bytes"]), encoding="utf-8-sig")
        # Prefer a freshly uploaded Daily Orders file (even if Orders/Campaigns are loaded from GitHub).
        # This fixes the common case: user opens the app (GitHub snapshot loads) then uploads only daily_orders.xlsx.
        if daily_orders_file is not None:
            daily_orders_df = pd.read_excel(daily_orders_file)
        elif snap.get("daily_orders_xlsx_bytes"):
            daily_orders_df = pd.read_excel(io.BytesIO(snap["daily_orders_xlsx_bytes"]))
        orders_df, campaigns_df, kpis = parse_inputs(orders_df, campaigns_df, fx)
        data_source = "github"
        kpis_disp = kpis_in_currency(kpis, fx, currency)
        net_profit_usd_taager, potential_net_usd_taager = compute_taager_kpis(orders_df, campaigns_df, TAAGER_FX)

    except Exception as e:
        st.error(f"Loaded from GitHub but failed to parse: {e}")
        st.stop()

if one_uploaded and data_source == "github":
    st.info("Only one file uploaded → showing LAST SAVED data from GitHub. Upload the missing file to refresh.")

# Ensure numeric
if orders_df is not None:
    for col in ["delivered_profit_iqd", "confirmed_profit_iqd"]:
        if col in orders_df.columns:
            orders_df[col] = to_num(orders_df[col])


# --------------------------------------------------------------

def _esc(s: str) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            )

def _tone(v):
    if v is None:
        return "neutral"
    return "good" if v >= 0 else "bad"

def _card(title: str, value_str: str, sub: str = "", tone: str = "neutral", tip: str = ""):
    tip_attr = f'data-tip="{_esc(tip)}"' if tip else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title-row">
            <div class="kpi-title">{_esc(title)}</div>
            {"<span class='kpi-tip' " + tip_attr + ">ⓘ</span>" if tip else ""}
          </div>
          <div class="kpi-value {tone}">{_esc(value_str)}</div>
          <div class="kpi-sub">{_esc(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )




def _safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _build_llm_payload(
    *,
    kpis: dict,
    kpis_disp: dict,
    today_row: dict,
    yesterday_row: dict,
    currency: str,
    windows: Optional[dict] = None,
    products_top: Optional[list] = None,
    campaigns_top: Optional[list] = None,
    detailed_context: Optional[dict] = None,
    data_quality: Optional[dict] = None,
    spend_allocation_method: str = "order_share",
):
    """
    Structured payload for the LLM.

    Keep it compact (numbers + small top lists) so it stays token-efficient,
    but rich enough for product/ads/profit analysis.
    """
    return {
        "currency": currency,
        "spend_allocation_method": spend_allocation_method,
        "data_quality": data_quality or {},
        "overall": {
            "spend": kpis_disp.get("spend_disp"),
            "delivered_profit": kpis_disp.get("delivered_profit_disp"),
            "confirmed_profit": kpis_disp.get("confirmed_profit_disp"),
            "net_after_ads": kpis.get("net_profit_usd"),
            "roas_delivered": kpis.get("roas_real"),
            "roas_potential": kpis.get("roas_potential"),
            "confirmation_rate": kpis.get("confirmation_rate"),
            "delivery_rate": kpis.get("delivery_rate"),
            "return_rate": kpis.get("return_rate"),
        },
        "windows": windows or {},
        "today": today_row,
        "yesterday": yesterday_row,
        "products_top": products_top or [],
        "campaigns_top": campaigns_top or [],
        "detailed_context": detailed_context or {},
    }
@st.cache_data(show_spinner=False)
def chatgpt_generate_store_summary(payload_json: str, user_focus: str = "") -> str:
    """Generate a narrative summary using OpenAI Responses API.

    Cached by (payload_json, user_focus) so it updates when data changes.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing **OPENAI_API_KEY** in Streamlit secrets. Add it to `.streamlit/secrets.toml` or your deployment secrets."
    if OpenAI is None:
        return "⚠️ `openai` Python package not installed. Run `pip install openai` in your environment."

    client = OpenAI(api_key=api_key)

    system = (
        "You are a sharp ecommerce performance analyst for a COD store using Taager + Meta ads. "
        "Write in a clear, direct style. Use numbers from the JSON payload. "
        "Avoid vague advice. If you make an assumption, label it as an assumption."
    )

    focus = user_focus.strip()
    focus_line = f"Extra focus requested: {focus}" if focus else ""

    prompt = (
        "You will receive store KPIs, time-window summaries, top products, and top Meta campaigns as JSON.\n\n"
        "Return Markdown with EXACTLY these sections (use the headings verbatim):\n\n"
        "## Executive Summary\n"
        "(3–5 sentences, no bullets. Mention: net after ads, delivery/return health, and what to do next.)\n\n"
        "## Trend (today vs yesterday, and vs last 7 days)\n"
        "Bullets with concrete deltas. Use `today`, `yesterday`, and `windows.last_7d` when available.\n\n"
        "If `detailed_context` is present and non-empty, use it to support deeper reasoning and specific examples.\n\n"
        "## Products (winners & losers)\n"
        "- 3 winners: product, why (numbers), and what action to take\n"
        "- 3 losers: product, what is wrong (numbers), and what action to take\n"
        "- For each product, compare recent performance (`*_7d`) versus lifetime (`*_lifetime`) from first campaign day to analysis day.\n"
        "- Treat new launches carefully using `campaign_age_days`, `is_new_launch`, and `is_new_no_delivery_expected`.\n"
        "- If `is_new_no_delivery_expected` is true, do NOT classify that product as a loser only due to low/zero deliveries.\n"
        "Use `products_top` only. If it is empty, say product-level data is missing.\n\n"
        "## Ads (scale / hold / cut)\n"
        "- Scale: up to 3 campaigns with reasons\n"
        "- Hold: up to 3 campaigns with reasons\n"
        "- Cut/Pause: up to 3 campaigns with reasons\n"
        "Use `campaigns_top` only. If it is empty, say campaign-level data is missing.\n\n"
        "## Profit & operations\n"
        "Discuss how delivery rate / cancellations / returns affect profit. Give 3 concrete operational actions.\n\n"
        "## 24–48h plan\n"
        "3 steps max. Each step must include a measurable target.\n\n"
        "Rules:\n"
        "- Use the provided JSON only; do not invent metrics.\n"
        "- In this app payload, `today` means the selected analysis day (usually the last completed day), not necessarily the calendar current day.\n"
        "- `yesterday` means the day before that selected analysis day.\n"
        "- Always reference at least one metric when making a recommendation.\n"
        "- Treat allocated ad spend at the product level as an ESTIMATE based on `spend_allocation_method`.\n"
        "- Respect product maturity: avoid harsh judgments for products with low `campaign_age_days`.\n"
        "- If data is missing, say exactly what's missing and how to fix it.\n"
        "- Do not mention that you are an AI model.\n\n"
        f"{focus_line}\n\n"
        "JSON:\n"
        f"{payload_json}"
    )

    # Use Responses API (recommended over deprecated Assistants/ChatCompletions patterns)
    try:
        r = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_output_tokens=900,
        )
        # SDK returns a unified output; `.output_text` is the convenience accessor in recent versions
        txt = getattr(r, "output_text", None)
        if txt:
            return txt
        # Fallback: best-effort extraction
        if hasattr(r, "output") and r.output:
            chunks = []
            for item in r.output:
                if isinstance(item, dict):
                    # try common shape
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                chunks.append(c.get("text", ""))
                else:
                    pass
            out = "\n".join([c for c in chunks if c])
            return out or "(No text returned)"
        return "(No text returned)"
    except Exception as e:
        return f"⚠️ ChatGPT call failed: {e}"

def chatgpt_answer_data_question(payload_json: str, question: str) -> str:
    """Answer a free-form question using only the provided app data payload."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing **OPENAI_API_KEY** in Streamlit secrets. Add it to `.streamlit/secrets.toml` or your deployment secrets."
    if OpenAI is None:
        return "⚠️ `openai` Python package not installed. Run `pip install openai` in your environment."

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are answering questions about ecommerce performance data.\n"
        "Rules:\n"
        "- Use only the JSON data provided below.\n"
        "- If `detailed_context` exists, prioritize it for deeper answers.\n"
        "- If a metric is missing, say it is missing.\n"
        "- Be direct and practical.\n"
        "- When relevant, include exact numbers from the JSON.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"JSON:\n{payload_json}"
    )
    try:
        r = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": "You are a precise ecommerce data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=800,
        )
        txt = getattr(r, "output_text", None)
        if txt:
            return txt
        if hasattr(r, "output") and r.output:
            chunks = []
            for item in r.output:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                chunks.append(c.get("text", ""))
            out = "\n".join([c for c in chunks if c])
            return out or "(No text returned)"
        return "(No text returned)"
    except Exception as e:
        return f"⚠️ ChatGPT call failed: {e}"


def render_ai_summary(
    *,
    kpis: dict,
    kpis_disp: dict,
    daily_orders_df: Optional[pd.DataFrame],
    campaigns_df: Optional[pd.DataFrame],
    orders_df: Optional[pd.DataFrame],
    fx: float,
    currency: str,
    last_saved: Optional[str] = None,
):
    """Rule-based 'AI' summary (no external API)."""
    if last_saved:
        st.markdown(
            f"<div class='ai-banner'>Showing last saved snapshot from GitHub • {last_saved}</div>",
            unsafe_allow_html=True,
        )

    # --- Overall snapshot (all-time, from Orders+Campaigns KPIs) ---
    spend_usd = float(kpis.get("spend_usd", 0.0))
    delivered_profit_usd = float(kpis.get("delivered_profit_usd", 0.0))
    confirmed_profit_usd = float(kpis.get("confirmed_profit_usd", 0.0))
    net_usd = float(kpis.get("net_profit_usd", delivered_profit_usd - spend_usd))

    roas_real = kpis.get("roas_real")
    roas_pot = kpis.get("roas_potential")
    conf_rate = float(kpis.get("confirmation_rate", 0.0))
    deliv_rate = float(kpis.get("delivery_rate", 0.0))
    ret_rate = float(kpis.get("return_rate", 0.0))

    spend_total = money_ccy(kpis_disp.get("spend_disp", 0.0), currency)
    delivered_total = money_ccy(kpis_disp.get("delivered_profit_disp", 0.0), currency)
    net_total = money_ccy(net_usd * fx, "IQD") if currency == "IQD" else money_ccy(net_usd, "USD")
    roas_total = "N/A" if roas_real is None else f"{roas_real:.2f}"
    snapshot_txt = last_saved if last_saved else "Live session"

    st.markdown(
        f"""
        <div class="ai-card">
          <div class="ai-card-title">Store Overview</div>
          <div class="ai-card-sub">Snapshot: {snapshot_txt} | Currency: {currency}</div>
          <div class="ai-grid-4">
            <div><div class="ai-k-label">Total spend</div><div class="ai-k-value">{spend_total}</div><div class="ai-k-sub">Total ad investment</div></div>
            <div><div class="ai-k-label">Delivered profit</div><div class="ai-k-value">{delivered_total}</div><div class="ai-k-sub">Gross profit</div></div>
            <div><div class="ai-k-label">Net after ads</div><div class="ai-k-value">{net_total}</div><div class="ai-k-sub">Net profit after ad costs</div></div>
            <div><div class="ai-k-label">Delivered ROAS</div><div class="ai-k-value">{roas_total}</div><div class="ai-k-sub">Return on ad spend</div></div>
          </div>
          <div class="ai-divider"></div>
          <div class="ai-grid-3">
            <div><div class="ai-k-label">Confirmation rate</div><div class="ai-k-value">{conf_rate:.1%}</div><div class="ai-k-sub">Orders confirmed</div></div>
            <div><div class="ai-k-label">Delivery rate</div><div class="ai-k-value">{deliv_rate:.1%}</div><div class="ai-k-sub">Successfully delivered</div></div>
            <div><div class="ai-k-label">Return rate</div><div class="ai-k-value">{ret_rate:.1%}</div><div class="ai-k-sub">Orders returned</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if daily_orders_df is None or getattr(daily_orders_df, "empty", True):
        st.info("Upload **Daily Orders (Taager) XLSX** to enable analysis-day summary + recommendations.")
        return

    ddf = parse_daily_orders(daily_orders_df)
    if "day" not in ddf.columns or ddf["day"].isna().all():
        st.warning("Couldn't read dates from the Daily Orders file (missing/invalid **Created At**).")
        return

    available_days = sorted(pd.to_datetime(ddf["day"], errors="coerce").dropna().dt.normalize().unique())
    if not available_days:
        st.info("No valid dates found in Daily Orders.")
        return

    include_latest_partial = st.checkbox(
        "Use latest day",
        value=False,
        key="ai_include_latest_partial",
        help="If off, AI uses the last completed day.",
    )

    latest_day = pd.Timestamp(available_days[-1]).normalize()
    if include_latest_partial:
        analysis_day = latest_day
        analysis_mode = "latest available day (can be partial)"
    else:
        if len(available_days) >= 2:
            analysis_day = pd.Timestamp(available_days[-2]).normalize()
            analysis_mode = "last completed day"
        else:
            analysis_day = latest_day
            analysis_mode = "latest available day (only one day in file)"
            st.caption("Only one day exists in the file, so the app is using that day.")

    compare_day = (analysis_day - pd.Timedelta(days=1)).normalize()

    # Keep variable names used below; here "today" means selected analysis day.
    today = analysis_day
    yesterday = compare_day

    daily_summary = build_daily_summary(ddf, campaigns_df if campaigns_df is not None else pd.DataFrame(), fx, currency)
    if daily_summary is None or daily_summary.empty or "day" not in daily_summary.columns:
        st.info("Not enough daily data to build a summary.")
        return

    def _row_for(day_ts: pd.Timestamp) -> dict:
        r = daily_summary[daily_summary["day"] == day_ts]
        return {} if r.empty else r.iloc[0].to_dict()

    t = _row_for(today)
    y = _row_for(yesterday)

    orders_t = int(t.get("orders_count", 0) or 0)
    profit_t = float(t.get("profit_disp", 0.0) or 0.0)
    spend_t = float(t.get("spend_disp", 0.0) or 0.0)
    net_t = float(t.get("net_disp", 0.0) or 0.0)

    orders_y = int(y.get("orders_count", 0) or 0)
    profit_y = float(y.get("profit_disp", 0.0) or 0.0)
    spend_y = float(y.get("spend_disp", 0.0) or 0.0)
    net_y = float(y.get("net_disp", 0.0) or 0.0)

    day_label = str(today.date())
    prev_label = str(yesterday.date())

    def _delta_class(v: float) -> str:
        return "ai-delta-pos" if v >= 0 else "ai-delta-neg"

    def _signed_num(v: int) -> str:
        return f"{v:+,}"

    def _signed_money(v: float) -> str:
        return f"+{money_ccy(v, currency)}" if v >= 0 else money_ccy(v, currency)

    orders_delta = orders_t - orders_y
    profit_delta = profit_t - profit_y
    spend_delta = spend_t - spend_y
    net_delta = net_t - net_y

    st.markdown(
        f"""
        <div class="ai-card">
          <div class="ai-card-title">Day Comparison</div>
          <div class="ai-card-sub">Analysis day: {day_label} ({analysis_mode}) | Comparison day: {prev_label}</div>
          <div class="ai-grid-4">
            <div><div class="ai-k-label">Orders</div><div class="ai-k-value">{orders_t:,}</div><div class="{_delta_class(orders_delta)}">{_signed_num(orders_delta)} vs {prev_label}</div></div>
            <div><div class="ai-k-label">Profit</div><div class="ai-k-value">{money_ccy(profit_t, currency)}</div><div class="{_delta_class(profit_delta)}">{_signed_money(profit_delta)}</div></div>
            <div><div class="ai-k-label">Ad spend</div><div class="ai-k-value">{money_ccy(spend_t, currency)}</div><div class="{_delta_class(spend_delta)}">{_signed_money(spend_delta)}</div></div>
            <div><div class="ai-k-label">Net</div><div class="ai-k-value">{money_ccy(net_t, currency)}</div><div class="{_delta_class(net_delta)}">{_signed_money(net_delta)}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_daily_graph = st.toggle(
        "Show daily graph",
        value=False,
        key="ai_show_daily_graph",
        help="Display daily trends for orders, profit, ad spend, and net.",
    )
    if show_daily_graph:
        daily_graph = daily_summary[["day", "orders_count", "profit_disp", "spend_disp", "net_disp"]].copy()
        daily_graph = daily_graph.dropna(subset=["day"]).sort_values("day")
        daily_graph["day"] = pd.to_datetime(daily_graph["day"], errors="coerce")
        daily_graph = daily_graph.dropna(subset=["day"])
        month_start = today.replace(day=1).normalize()
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        daily_graph = daily_graph[(daily_graph["day"] >= month_start) & (daily_graph["day"] <= month_end)].copy()
        st.markdown("#### Daily graph")
        if go is not None and make_subplots is not None:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Orders", f"Profit ({currency})", f"Ad spend ({currency})", f"Net ({currency})"),
                vertical_spacing=0.17,
                horizontal_spacing=0.08,
            )

            metric_specs = [
                ("orders_count", "Orders", "#64D2FF", "rgba(100,210,255,0.20)", ",.0f", 1, 1),
                ("profit_disp", f"Profit ({currency})", "#4EE3A3", "rgba(78,227,163,0.20)", ",.2f", 1, 2),
                ("spend_disp", f"Ad spend ({currency})", "#FFA66B", "rgba(255,166,107,0.20)", ",.2f", 2, 1),
                ("net_disp", f"Net ({currency})", "#B58DFF", "rgba(181,141,255,0.22)", ",.2f", 2, 2),
            ]

            for metric_col, metric_title, line_color, fill_color, fmt, row_i, col_i in metric_specs:
                chart_df = daily_graph[["day", metric_col]].rename(columns={metric_col: "value"}).copy()
                chart_df["value"] = pd.to_numeric(chart_df["value"], errors="coerce").fillna(0.0)
                chart_df = chart_df.sort_values("day")
                if chart_df.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=chart_df["day"],
                        y=chart_df["value"],
                        mode="lines",
                        line={"color": line_color, "width": 3, "shape": "spline", "smoothing": 0.65},
                        fill="tozeroy",
                        fillcolor=fill_color,
                        name=metric_title,
                        showlegend=True,
                        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{metric_title}: %{{y:{fmt}}}<extra></extra>",
                    ),
                    row=row_i,
                    col=col_i,
                )

                fig.add_trace(
                    go.Scatter(
                        x=chart_df["day"],
                        y=chart_df["value"],
                        mode="markers",
                        marker={"size": 6, "color": line_color, "line": {"color": "rgba(255,255,255,0.55)", "width": 1}},
                        name=f"{metric_title} points",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_i,
                    col=col_i,
                )

                last_point = chart_df.tail(1)
                fig.add_trace(
                    go.Scatter(
                        x=last_point["day"],
                        y=last_point["value"],
                        mode="markers+text",
                        marker={"size": 11, "color": "#EAF2F8", "line": {"color": line_color, "width": 2}},
                        text=[f"{float(last_point['value'].iloc[0]):{fmt}}"],
                        textposition="top right",
                        textfont={"size": 11, "color": "#EAF2F8"},
                        name=f"{metric_title} latest",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_i,
                    col=col_i,
                )

                for marker_day in [yesterday, today]:
                    marker_day = pd.Timestamp(marker_day).normalize()
                    if month_start <= marker_day <= month_end:
                        fig.add_vline(
                            x=marker_day,
                            line_dash="dash",
                            line_color="rgba(160,180,198,0.55)",
                            line_width=1.2,
                            row=row_i,
                            col=col_i,
                        )

                fig.update_xaxes(
                    showgrid=False,
                    tickformat="%d %b",
                    tickfont={"color": "#B8CAD9", "size": 11},
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikethickness=1,
                    spikecolor="rgba(184,202,217,0.40)",
                    row=row_i,
                    col=col_i,
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor="rgba(184,202,217,0.16)",
                    zeroline=False,
                    tickfont={"color": "#B8CAD9", "size": 11},
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikethickness=1,
                    spikecolor="rgba(184,202,217,0.30)",
                    row=row_i,
                    col=col_i,
                )

            fig.update_layout(
                height=660,
                margin={"l": 24, "r": 20, "t": 66, "b": 20},
                paper_bgcolor="rgba(9,16,26,0.68)",
                plot_bgcolor="rgba(9,16,26,0.68)",
                hovermode="x unified",
                font={"family": "Manrope, Segoe UI, sans-serif", "color": "#EAF2F8"},
                dragmode="zoom",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.04,
                    "xanchor": "left",
                    "x": 0.0,
                    "font": {"size": 11, "color": "#B8CAD9"},
                },
            )
            fig.update_annotations(font={"size": 18, "color": "#EAF2F8", "family": "Manrope, Segoe UI, sans-serif"})
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )
            st.caption("Dashed lines mark the comparison and analysis days.")
        else:
            st.info("Install Plotly for enhanced chart visuals. Falling back to the current chart style.")
        st.caption(f"Showing {today.strftime('%B %Y')} daily values.")

    api_ready = bool(st.secrets.get("OPENAI_API_KEY"))

    if "ai_last_output" not in st.session_state:
        st.session_state.ai_last_output = ""
    if "ai_last_run_at" not in st.session_state:
        st.session_state.ai_last_run_at = ""
    if "ai_qa_history" not in st.session_state:
        st.session_state.ai_qa_history = []

    st.markdown("<div class='ai-assistant-wrap'>", unsafe_allow_html=True)
    head1, head2 = st.columns([4, 1])
    with head1:
        st.markdown("### AI Assistant")
        st.caption("Ask questions about your campaigns, products, trends, and recommendations.")
    with head2:
        st.markdown(
            f"<div style='text-align:right; padding-top:6px'><span class='ai-status-pill'>{'Connected' if api_ready else 'Disconnected'}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='ai-divider'></div>", unsafe_allow_html=True)
    act1, act2 = st.columns([1, 3])
    with act1:
        gen = st.button(
            "Generate General Analysis",
            type="primary",
            key="btn_gen_ai",
            disabled=not api_ready,
            use_container_width=True,
        )
    with act2:
        st.caption("Get an overview of your store performance and recommendations")

    with st.expander("Advanced focus (optional)", expanded=False):
        focus_presets = {
            "None": "",
            "Optimize spend efficiency": "optimize spend efficiency",
            "Reduce cancellations": "reduce cancellations and improve confirmation",
            "Scale winning products": "find winners to scale with low risk",
            "Protect net profit": "protect net profit while maintaining growth",
        }
        p1, p2 = st.columns([2, 3])
        with p1:
            preset_label = st.selectbox("Focus preset", list(focus_presets.keys()), index=0, key="ai_focus_preset")
        with p2:
            focus_text = st.text_input(
                "Custom focus",
                key="ai_focus",
                placeholder="Example: prioritize campaigns with ROAS > 1.6 and high delivery rate",
            ).strip()
    selected_preset = focus_presets.get(st.session_state.get("ai_focus_preset", "None"), "")
    focus_text_val = st.session_state.get("ai_focus", "").strip()
    user_focus = focus_text_val if focus_text_val else selected_preset
    use_deeper_context = st.checkbox(
        "Use deeper data context (slower, more complete answers)",
        value=False,
        key="ai_use_deeper_context",
        help="Adds larger bounded datasets (daily series + campaign/product detail) to the AI payload.",
    )

    ai_question = st.text_area(
        "Ask anything",
        key="ai_free_question",
        height=120,
        label_visibility="collapsed",
        placeholder="Ask anything about your campaigns, products, trends, or get recommendations...",
    ).strip()

    def _set_ai_question(q: str):
        st.session_state.ai_free_question = q

    chips = [
        "What are my top performing campaigns?",
        "Analyze delivery issues",
        "Which products should I scale?",
        "Compare this week vs last week",
    ]
    chip_cols = st.columns([1, 1, 1, 1, 0.7])
    for i, q in enumerate(chips):
        with chip_cols[i]:
            st.button(q, key=f"ai_chip_{i}", on_click=_set_ai_question, args=(q,), use_container_width=True)
    with chip_cols[4]:
        ask_ai = st.button(
            "Ask AI",
            key="btn_ask_ai",
            use_container_width=True,
            disabled=not api_ready,
        )

    if not api_ready:
        st.info("Add `OPENAI_API_KEY` in Streamlit secrets to enable AI brief generation.")

    if ask_ai and api_ready and not ai_question:
        st.warning("Write a question first.")

    if (gen and api_ready) or (ask_ai and api_ready and ai_question):
            # -------- Build a richer (but still compact) analysis bundle for ChatGPT --------
            def _window_sum(start_day: pd.Timestamp, end_day: pd.Timestamp) -> dict:
                w = daily_summary[(daily_summary["day"] >= start_day) & (daily_summary["day"] <= end_day)].copy()
                if w.empty:
                    return {}
                out = {
                    "start": str(pd.Timestamp(start_day).date()),
                    "end": str(pd.Timestamp(end_day).date()),
                    "days": int(w.shape[0]),
                    "orders": _safe_int(w["orders_count"].sum()),
                    "profit": _safe_float(w["profit_disp"].sum()),
                    "spend": _safe_float(w["spend_disp"].sum()),
                    "net": _safe_float(w["net_disp"].sum()),
                }
                # simple averages (guarded)
                out["profit_per_order"] = _safe_float(out["profit"] / out["orders"]) if out["orders"] else None
                out["net_per_order"] = _safe_float(out["net"] / out["orders"]) if out["orders"] else None
                return out

            # Time windows
            last_7_start = (today - pd.Timedelta(days=6)).normalize()
            last_14_start = (today - pd.Timedelta(days=13)).normalize()
            mtd_start = today.replace(day=1).normalize()

            windows = {
                "today": _window_sum(today, today),
                "yesterday": _window_sum(yesterday, yesterday),
                "last_7d": _window_sum(last_7_start, today),
                "last_14d": _window_sum(last_14_start, today),
                "mtd": _window_sum(mtd_start, today),
            }

            # ---- Product leaderboard (7d + lifetime since first campaign day) ----
            products_top = []
            sku_to_name = build_sku_to_name_map(orders_df) if orders_df is not None else {}
            NEW_LAUNCH_DAYS = 7

            has_campaigns = campaigns_df is not None and not getattr(campaigns_df, "empty", True)
            sku_match_rate = None
            first_campaign_day_by_sku = pd.Series(dtype="datetime64[ns]")
            last_campaign_day_by_sku = pd.Series(dtype="datetime64[ns]")
            c_for_sku = pd.DataFrame(columns=["day", "sku", "spend_usd"])

            if has_campaigns and "Reporting starts" in campaigns_df.columns:
                c_all = campaigns_df.copy()
                c_all["Reporting starts"] = pd.to_datetime(c_all["Reporting starts"], errors="coerce")
                c_all = c_all.dropna(subset=["Reporting starts"])
                c_all["day"] = c_all["Reporting starts"].dt.floor("D")
                c_all = c_all[c_all["day"] <= today].copy()

                name_col_all = "Campaign name" if "Campaign name" in c_all.columns else ("Campaign" if "Campaign" in c_all.columns else None)
                spend_col_all = "Amount spent (USD)" if "Amount spent (USD)" in c_all.columns else None

                if name_col_all and spend_col_all:
                    c_all[spend_col_all] = pd.to_numeric(c_all[spend_col_all], errors="coerce").fillna(0.0)
                    extracted_all = c_all[name_col_all].apply(extract_sku_from_campaign_name)
                    sku_match_rate = float((extracted_all.notna().mean() * 100.0)) if len(extracted_all) else None

                    c_all["sku"] = extracted_all
                    c_for_sku = (
                        c_all.dropna(subset=["sku"])[["day", "sku", spend_col_all]]
                        .rename(columns={spend_col_all: "spend_usd"})
                        .copy()
                    )
                    c_for_sku["sku"] = c_for_sku["sku"].astype(str).str.strip()
                    c_for_sku = c_for_sku[c_for_sku["sku"] != ""]

                    if not c_for_sku.empty:
                        first_campaign_day_by_sku = c_for_sku.groupby("sku")["day"].min()
                        last_campaign_day_by_sku = c_for_sku.groupby("sku")["day"].max()

            analysis_start = (
                pd.Timestamp(first_campaign_day_by_sku.min()).normalize()
                if not first_campaign_day_by_sku.empty
                else last_7_start
            )

            def _clean_orders_slice(df_slice: pd.DataFrame) -> pd.DataFrame:
                if df_slice is None or df_slice.empty:
                    return pd.DataFrame()
                out = df_slice.copy()
                if "Status" in out.columns:
                    _status = out["Status"].astype(str).str.strip().str.lower()
                    out = out[~_status.str.contains("cancelled by you", na=False)].copy()
                return out

            d7 = _clean_orders_slice(ddf[(ddf["day"] >= last_7_start) & (ddf["day"] <= today)].copy())
            d_life = _clean_orders_slice(ddf[(ddf["day"] >= analysis_start) & (ddf["day"] <= today)].copy())

            lines_7 = _explode_order_lines(d7) if not d7.empty else pd.DataFrame()
            lines_life = _explode_order_lines(d_life) if not d_life.empty else pd.DataFrame()

            def _normalize_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
                if lines_df is None or lines_df.empty:
                    return pd.DataFrame()
                out = lines_df.copy()
                if "day" in out.columns:
                    out["day"] = pd.to_datetime(out["day"], errors="coerce").dt.floor("D")
                    out = out.dropna(subset=["day"])
                out["sku"] = out["sku"].astype(str).str.strip()
                out = out[out["sku"] != ""]
                return out

            lines_7 = _normalize_lines(lines_7)
            lines_life = _normalize_lines(lines_life)

            spend_usd_7d_by_day = pd.Series(dtype=float)
            spend_usd_life_by_day = pd.Series(dtype=float)
            spend_usd_lifetime_by_sku = pd.Series(dtype=float)
            if not c_for_sku.empty:
                spend_usd_7d_by_day = (
                    c_for_sku[(c_for_sku["day"] >= last_7_start) & (c_for_sku["day"] <= today)]
                    .groupby("day")["spend_usd"].sum()
                )
                spend_usd_life_by_day = (
                    c_for_sku[(c_for_sku["day"] >= analysis_start) & (c_for_sku["day"] <= today)]
                    .groupby("day")["spend_usd"].sum()
                )
                spend_usd_lifetime_by_sku = c_for_sku.groupby("sku")["spend_usd"].sum()

            def _sku_metrics(lines_df: pd.DataFrame, spend_by_day: pd.Series) -> dict:
                if lines_df is None or lines_df.empty:
                    return {
                        "profit_iqd_by_sku": pd.Series(dtype=float),
                        "spend_usd_by_sku": pd.Series(dtype=float),
                        "orders_by_sku": pd.Series(dtype=float),
                        "delivered_orders_by_sku": pd.Series(dtype=float),
                    }

                if "Status" in lines_df.columns:
                    st_lower = lines_df["Status"].astype(str).str.lower()
                    delivered_mask = st_lower.str.contains("delivered", na=False)
                else:
                    delivered_mask = pd.Series(False, index=lines_df.index)

                if "profit_iqd_alloc" in lines_df.columns:
                    profit_iqd_by_sku = lines_df[delivered_mask].groupby("sku")["profit_iqd_alloc"].sum()
                else:
                    profit_iqd_by_sku = pd.Series(dtype=float)

                orders_by_sku = lines_df.groupby("sku")["order_id"].nunique()
                delivered_orders_by_sku = lines_df[delivered_mask].groupby("sku")["order_id"].nunique()

                total_orders_by_day = lines_df.groupby("day")["order_id"].nunique()
                sku_orders_by_day = (
                    lines_df.groupby(["day", "sku"])["order_id"]
                    .nunique()
                    .rename("sku_orders")
                    .reset_index()
                )
                sku_orders_by_day["total_orders"] = sku_orders_by_day["day"].map(total_orders_by_day).fillna(0)
                sku_orders_by_day["spend_usd_day"] = sku_orders_by_day["day"].map(spend_by_day).fillna(0.0)
                sku_orders_by_day["spend_usd_alloc"] = np.where(
                    sku_orders_by_day["total_orders"] > 0,
                    sku_orders_by_day["spend_usd_day"] * (sku_orders_by_day["sku_orders"] / sku_orders_by_day["total_orders"]),
                    0.0,
                )
                spend_usd_by_sku = sku_orders_by_day.groupby("sku")["spend_usd_alloc"].sum()

                return {
                    "profit_iqd_by_sku": profit_iqd_by_sku,
                    "spend_usd_by_sku": spend_usd_by_sku,
                    "orders_by_sku": orders_by_sku,
                    "delivered_orders_by_sku": delivered_orders_by_sku,
                }

            m7 = _sku_metrics(lines_7, spend_usd_7d_by_day)
            ml = _sku_metrics(lines_life, spend_usd_life_by_day)

            all_skus = (
                set(m7["orders_by_sku"].index)
                | set(m7["spend_usd_by_sku"].index)
                | set(m7["profit_iqd_by_sku"].index)
                | set(ml["orders_by_sku"].index)
                | set(ml["spend_usd_by_sku"].index)
                | set(ml["profit_iqd_by_sku"].index)
                | set(first_campaign_day_by_sku.index)
            )

            rows = []
            for sku in all_skus:
                profit_iqd_7d = float(m7["profit_iqd_by_sku"].get(sku, 0.0))
                spend_usd_7d = float(m7["spend_usd_by_sku"].get(sku, 0.0))
                orders_7d = int(m7["orders_by_sku"].get(sku, 0) or 0)
                delivered_7d = int(m7["delivered_orders_by_sku"].get(sku, 0) or 0)

                profit_iqd_lifetime = float(ml["profit_iqd_by_sku"].get(sku, 0.0))
                spend_usd_lifetime = float(ml["spend_usd_by_sku"].get(sku, 0.0))
                orders_lifetime = int(ml["orders_by_sku"].get(sku, 0) or 0)
                delivered_lifetime = int(ml["delivered_orders_by_sku"].get(sku, 0) or 0)

                if currency == "IQD":
                    profit_7d = profit_iqd_7d
                    spend_7d = spend_usd_7d * fx
                    profit_lifetime = profit_iqd_lifetime
                    spend_lifetime = spend_usd_lifetime * fx
                else:
                    profit_7d = iqd_to_usd(profit_iqd_7d, fx)
                    spend_7d = spend_usd_7d
                    profit_lifetime = iqd_to_usd(profit_iqd_lifetime, fx)
                    spend_lifetime = spend_usd_lifetime

                net_7d = profit_7d - spend_7d
                net_lifetime = profit_lifetime - spend_lifetime

                first_day = first_campaign_day_by_sku.get(sku, pd.NaT)
                last_day = last_campaign_day_by_sku.get(sku, pd.NaT)
                age_days = None
                if pd.notna(first_day):
                    age_days = max(1, int((today - pd.Timestamp(first_day).normalize()).days) + 1)

                is_new_launch = bool(age_days is not None and age_days <= NEW_LAUNCH_DAYS)
                is_new_no_delivery_expected = bool(is_new_launch and delivered_lifetime == 0 and orders_lifetime > 0)

                rows.append({
                    "sku": sku,
                    "name": sku_to_name.get(sku) or None,
                    "orders_7d": orders_7d,
                    "delivered_orders_7d": delivered_7d,
                    "delivery_rate_7d": (delivered_7d / orders_7d) if orders_7d else None,
                    "profit_7d": profit_7d,
                    "spend_7d": spend_7d,
                    "net_7d": net_7d,
                    "orders_lifetime": orders_lifetime,
                    "delivered_orders_lifetime": delivered_lifetime,
                    "delivery_rate_lifetime": (delivered_lifetime / orders_lifetime) if orders_lifetime else None,
                    "profit_lifetime": profit_lifetime,
                    "spend_lifetime": spend_lifetime,
                    "net_lifetime": net_lifetime,
                    "campaign_spend_lifetime_usd": float(spend_usd_lifetime_by_sku.get(sku, 0.0)),
                    "first_campaign_day": str(pd.Timestamp(first_day).date()) if pd.notna(first_day) else None,
                    "last_campaign_day": str(pd.Timestamp(last_day).date()) if pd.notna(last_day) else None,
                    "campaign_age_days": age_days,
                    "is_new_launch": is_new_launch,
                    "is_new_no_delivery_expected": is_new_no_delivery_expected,
                    "new_launch_window_days": NEW_LAUNCH_DAYS,
                    "net_per_delivered_7d": (net_7d / delivered_7d) if delivered_7d else None,
                    "net_per_delivered_lifetime": (net_lifetime / delivered_lifetime) if delivered_lifetime else None,
                    "rank_weight": float(spend_usd_7d + 0.35 * spend_usd_lifetime),
                })

            rows = sorted(rows, key=lambda r: (r.get("rank_weight", 0.0), r.get("orders_7d", 0)), reverse=True)
            products_top = rows[:20]

            # ---- Campaign leaderboard (last 7d) ----
            campaigns_top = []
            if has_campaigns and "Reporting starts" in campaigns_df.columns:
                c = campaigns_df.copy()
                c["Reporting starts"] = pd.to_datetime(c["Reporting starts"], errors="coerce")
                c = c.dropna(subset=["Reporting starts"])
                c["day"] = c["Reporting starts"].dt.floor("D")
                c = c[(c["day"] >= last_7_start) & (c["day"] <= today)].copy()

                name_col = "Campaign name" if "Campaign name" in c.columns else ("Campaign" if "Campaign" in c.columns else None)
                spend_col = "Amount spent (USD)" if "Amount spent (USD)" in c.columns else None
                results_col = "Results" if "Results" in c.columns else None

                if name_col and spend_col:
                    c[spend_col] = pd.to_numeric(c[spend_col], errors="coerce").fillna(0)

                    if results_col:
                        c[results_col] = pd.to_numeric(c[results_col], errors="coerce").fillna(0)

                    # SKU extraction quality (best-effort)
                    try:
                        extracted = c[name_col].apply(extract_sku_from_campaign_name)
                        sku_match_rate = float((extracted.notna().mean() * 100.0)) if len(extracted) else None
                    except Exception:
                        sku_match_rate = None

                    agg = c.groupby(name_col).agg(
                        spend_usd=(spend_col, "sum"),
                        results=(results_col, "sum") if results_col else (spend_col, "size"),
                    ).reset_index().rename(columns={name_col: "campaign"})

                    agg["cpr_usd"] = np.where(agg["results"] > 0, agg["spend_usd"] / agg["results"], np.nan)
                    agg = agg.sort_values("spend_usd", ascending=False).head(10)

                    campaigns_top = [
                        {
                            "campaign": r["campaign"],
                            "spend_usd": float(r["spend_usd"]),
                            "results": float(r["results"]) if results_col else None,
                            "cpr_usd": None if pd.isna(r["cpr_usd"]) else float(r["cpr_usd"]),
                        }
                        for _, r in agg.iterrows()
                    ]

            detailed_context = {}
            if use_deeper_context:
                daily_detail_cols = [c for c in ["day", "orders_count", "profit_disp", "spend_disp", "net_disp"] if c in daily_summary.columns]
                daily_detail = daily_summary[daily_detail_cols].copy().sort_values("day").tail(90)
                if "day" in daily_detail.columns:
                    daily_detail["day"] = daily_detail["day"].astype(str)

                status_daily = []
                if "day" in ddf.columns and "Status" in ddf.columns:
                    ds = ddf.copy()
                    ds["day"] = pd.to_datetime(ds["day"], errors="coerce").dt.floor("D")
                    ds = ds.dropna(subset=["day"])
                    ds["status_clean"] = ds["Status"].astype(str).str.strip().str.lower()
                    id_col = get_daily_order_id_col(ds)
                    if id_col is None:
                        ds["__rowid__"] = range(len(ds))
                        id_col = "__rowid__"
                    status_daily_df = (
                        ds.groupby(["day", "status_clean"], as_index=False)[id_col]
                        .nunique()
                        .rename(columns={id_col: "orders"})
                        .sort_values(["day", "orders"], ascending=[True, False])
                    )
                    status_daily_df = status_daily_df.tail(240)
                    status_daily_df["day"] = status_daily_df["day"].astype(str)
                    status_daily = status_daily_df.to_dict("records")

                campaigns_daily_top = []
                if has_campaigns and "Reporting starts" in campaigns_df.columns:
                    cdt = campaigns_df.copy()
                    cdt["Reporting starts"] = pd.to_datetime(cdt["Reporting starts"], errors="coerce")
                    cdt = cdt.dropna(subset=["Reporting starts"])
                    cdt["day"] = cdt["Reporting starts"].dt.floor("D")
                    cdt = cdt[cdt["day"] >= (today - pd.Timedelta(days=60)).normalize()].copy()
                    cname = "Campaign name" if "Campaign name" in cdt.columns else ("Campaign" if "Campaign" in cdt.columns else None)
                    if cname and "Amount spent (USD)" in cdt.columns:
                        cdt["Amount spent (USD)"] = pd.to_numeric(cdt["Amount spent (USD)"], errors="coerce").fillna(0)
                        if "Results" in cdt.columns:
                            cdt["Results"] = pd.to_numeric(cdt["Results"], errors="coerce").fillna(0)
                        else:
                            cdt["Results"] = 0.0
                        top_names = (
                            cdt.groupby(cname, as_index=False)["Amount spent (USD)"]
                            .sum()
                            .sort_values("Amount spent (USD)", ascending=False)
                            .head(12)[cname]
                            .tolist()
                        )
                        cdt = cdt[cdt[cname].isin(top_names)]
                        cdt = (
                            cdt.groupby(["day", cname], as_index=False)[["Amount spent (USD)", "Results"]]
                            .sum()
                            .rename(columns={cname: "campaign", "Amount spent (USD)": "spend_usd", "Results": "results"})
                            .sort_values("day")
                            .tail(300)
                        )
                        cdt["day"] = cdt["day"].astype(str)
                        campaigns_daily_top = cdt.to_dict("records")

                detailed_context = {
                    "enabled": True,
                    "limits": {
                        "daily_series_rows": 90,
                        "status_daily_rows": 240,
                        "campaign_daily_rows": 300,
                        "products_rows": 50,
                    },
                    "daily_series": daily_detail.to_dict("records"),
                    "status_daily": status_daily,
                    "campaigns_daily_top": campaigns_daily_top,
                    "products_detailed": rows[:50],
                }

            data_quality = {
                "has_daily_orders": daily_orders_df is not None and not getattr(daily_orders_df, "empty", True),
                "has_campaigns": has_campaigns,
                "has_orders_catalog": orders_df is not None and not getattr(orders_df, "empty", True),
                "sku_match_rate_pct": sku_match_rate,
                "ai_deeper_context_enabled": bool(use_deeper_context),
                "notes": [
                    "Product-level ad spend is allocated as an estimate using order share by day.",
                    f"Products with campaign_age_days <= {NEW_LAUNCH_DAYS} are considered new launches.",
                ],
            }

            payload = _build_llm_payload(
                kpis=kpis,
                kpis_disp=kpis_disp,
                today_row=t,
                yesterday_row=y,
                currency=currency,
                windows=windows,
                products_top=products_top,
                campaigns_top=campaigns_top,
                detailed_context=detailed_context,
                data_quality=data_quality,
                spend_allocation_method="order_share",
            )
            payload_json = json.dumps(_json_safe(payload), ensure_ascii=False)
            if gen and api_ready:
                with st.spinner("Generating brief..."):
                    out = chatgpt_generate_store_summary(payload_json, user_focus=user_focus)
                out = clean_markdown_spacing(out)
                st.session_state.ai_last_output = out
                st.session_state.ai_last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if ask_ai and api_ready and ai_question:
                with st.spinner("Answering your question..."):
                    ans = chatgpt_answer_data_question(payload_json, ai_question)
                ans = clean_markdown_spacing(ans)
                st.session_state.ai_qa_history.append(
                    {
                        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "q": ai_question,
                        "a": ans,
                    }
                )
                st.session_state.ai_qa_history = st.session_state.ai_qa_history[-10:]

    st.markdown("#### AI Output")
    if st.session_state.ai_last_output:
        if st.session_state.ai_last_run_at:
            st.caption(f"Generated at {st.session_state.ai_last_run_at}")
        render_ai_text(st.session_state.ai_last_output)
    elif st.session_state.ai_qa_history:
        latest = st.session_state.ai_qa_history[-1]
        st.caption(f"Latest answer • {latest['at']}")
        st.markdown(f"**Q:** {latest['q']}")
        render_ai_text(latest["a"])
    else:
        st.markdown("<div class='ai-empty'>Your AI-generated insights will appear here</div>", unsafe_allow_html=True)

    tools_col, _ = st.columns([1, 6])
    with tools_col:
        clear_output = st.button("Clear", key="btn_ai_clear", use_container_width=True)
    if clear_output:
        st.session_state.ai_last_output = ""
        st.session_state.ai_qa_history = []
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (!doc) return;
          const buttons = doc.querySelectorAll('button');
          const chipTexts = new Set([
            'What are my top performing campaigns?',
            'Analyze delivery issues',
            'Which products should I scale?',
            'Compare this week vs last week'
          ]);
          buttons.forEach((btn) => {
            const t = (btn.innerText || '').trim();
            if (!t) return;
            if (t === 'Generate General Analysis') btn.classList.add('ai-btn-main');
            if (t === 'Ask AI') btn.classList.add('ai-btn-ask');
            if (t === 'Clear') btn.classList.add('ai-btn-clear');
            if (chipTexts.has(t)) btn.classList.add('ai-btn-chip');
          });
        })();
        </script>
        """,
        height=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)



# Show last upload date (from GitHub snapshot) + data source
last_saved = None
if data_source == "github" and snap is not None:
    last_saved = snap.get("generated_at")

if last_saved:
    st.caption(f"Data source: {data_source} • Last saved to GitHub: {last_saved}")
else:
    st.caption(f"Data source: {data_source}")



if one_uploaded:
    st.info("Dashboard is showing the LAST SAVED snapshot. Upload the missing file to refresh.")

# --- Floating: Quick KPIs accordion (fixed bottom-left, persistent across all tabs) ---
render_fixed_quick_kpis(daily_orders_df=daily_orders_df, orders_df=orders_df, fx=fx, currency=currency)


# --- Tabs ---
tab_dashboard, tab_ai, tab_daily, tab_orders, tab_ads, tab_campaigns, tab_product = st.tabs(
    ["📊 Dashboard", "🤖 AI Summary", "📅 Daily performance", "📦 Orders details", "📣 Ads details", "📈 Campaigns analytics", "📦 Product Deep Dive"]
)


with tab_dashboard:
    if data_source == "github" and snap is not None and snap.get("generated_at"):
        st.info(f"Showing last saved snapshot from GitHub • {snap['generated_at']}")
    elif data_source == "uploads":
        st.success("Showing uploaded files (not yet saved).")

    # Dashboard cards
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Confirmed Profit ({currency})", money_ccy(kpis_disp["confirmed_profit_disp"], currency), f"{int(kpis['confirmed_units']):,} confirmed")
    col2.metric(f"Delivered Profit ({currency})", money_ccy(kpis_disp["delivered_profit_disp"], currency), f"{int(kpis['delivered_units']):,} delivered")
    col3.metric(f"Ad Spend ({currency})", money_ccy(kpis_disp["spend_disp"], currency))

    st.markdown("### Profit After Ads")

    # helpers
    




    # compute display values
    net_disp = kpis_disp["net_profit_disp"]
    pot_disp = kpis_disp["potential_net_disp"]

    # IMPORTANT: Taager FX numbers must use 1602 only (never the user FX input)
    if net_profit_usd_taager is None:
        net_taager_disp = None
    else:
        net_taager_disp = net_profit_usd_taager * TAAGER_FX if currency == "IQD" else net_profit_usd_taager

    # Taager potential should ALWAYS be in USD using fixed 1602 rate

    # Step 1: get potential in IQD
    potential_iqd = kpis["potential_net_profit_usd"] * fx if currency == "USD" else kpis_disp["potential_net_disp"]

    # Step 2: convert to USD using fixed 1602
    pot_taager_disp = potential_iqd / TAAGER_FX if potential_iqd is not None else None



    # render 4 cards in a tight grid
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    _card("Net (Delivered − Spend)",
        money_ccy(net_disp, currency),
        "Realized profitability",
        _tone(net_disp))

    # _card("Net (Taager FX 1602)",
    #     "N/A" if net_taager_disp is None else money_ccy(net_taager_disp, currency),
    #     "Using payout FX",
    #     _tone(net_taager_disp))

    _card(
        "Potential (Confirmed − Spend)",
        money_ccy(pot_disp, currency),
        "If all confirmed deliver",
        _tone(pot_disp),
        tip=(
            "Potential Net = Confirmed Profit − Ad Spend\n"
            f"Confirmed Profit ({currency}) = {money_ccy(kpis_disp['confirmed_profit_disp'], currency)}\n"
            f"Ad Spend ({currency}) = {money_ccy(kpis_disp['spend_disp'], currency)}\n"
        )
    )


    _card(
        "Potential (Taager FX 1602)",
        "N/A" if pot_taager_disp is None else money_ccy(pot_taager_disp, "USD"),
        "Using payout FX",
        _tone(pot_taager_disp),
        tip=(
            "Taager Potential (USD) = Potential Net (IQD) ÷ 1602\n"
            "Potential Net (IQD) = (Confirmed Profit (IQD) − Ad Spend (IQD))\n"
            f"Confirmed Profit (IQD) = {money_ccy(kpis['confirmed_profit_usd'] * fx, 'IQD')}\n"
            f"Ad Spend (IQD) = {money_ccy(kpis['spend_usd'] * fx, 'IQD')}\n"
        )
    )


    st.markdown("</div>", unsafe_allow_html=True)




    # --- ROAS row (separate) ---
    roas1, roas2 = st.columns(2)
    roas1.metric("ROAS (Realized)", fmt_ratio(kpis["roas_real"]))
    roas2.metric("ROAS (Potential)", fmt_ratio(kpis["roas_potential"]))




    st.caption("Taager FX 1602 = payout rate to Payoneer (IQD → USD).")


    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmation Rate", pct(kpis["confirmation_rate"]))
    c2.metric("Delivery Rate", pct(kpis["delivery_rate"]))
    c3.metric("Return Rate", pct(kpis["return_rate"]))
    c4.metric("CPM", fmt_money_or_na(kpis["cpm"]))

    # Charts
    funnel_png, realized_png, potential_png = make_charts_bytes(kpis)

    st.subheader("Charts")
    if go is not None:
        chart_config = {
            "displayModeBar": True,
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }
        plot_bg = "rgba(9,16,26,0.70)"
        grid_clr = "rgba(184,202,217,0.16)"
        text_clr = "#EAF2F8"
        tick_clr = "#B8CAD9"

        fig_funnel = go.Figure(
            data=[
                go.Bar(
                    x=["Requested", "Confirmed", "Delivered"],
                    y=[kpis["requested_units"], kpis["confirmed_units"], kpis["delivered_units"]],
                    marker_color=["#64D2FF", "#4EE3A3", "#B58DFF"],
                    text=[f"{int(kpis['requested_units']):,}", f"{int(kpis['confirmed_units']):,}", f"{int(kpis['delivered_units']):,}"],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Units: %{y:,}<extra></extra>",
                )
            ]
        )
        fig_funnel.update_layout(
            title="Order Funnel",
            height=340,
            margin=dict(l=20, r=20, t=52, b=22),
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(family="Manrope, Segoe UI, sans-serif", color=text_clr),
            dragmode="zoom",
        )
        fig_funnel.update_yaxes(showgrid=True, gridcolor=grid_clr, tickfont=dict(color=tick_clr))
        fig_funnel.update_xaxes(tickfont=dict(color=tick_clr))

        fig_realized = go.Figure(
            data=[
                go.Bar(
                    x=["Delivered Profit", "Ad Spend", "Net Profit"],
                    y=[kpis["delivered_profit_usd"], kpis["spend_usd"], kpis["net_profit_usd"]],
                    marker_color=["#4EE3A3", "#FFA66B", "#64D2FF"],
                    text=[f"${kpis['delivered_profit_usd']:,.2f}", f"${kpis['spend_usd']:,.2f}", f"${kpis['net_profit_usd']:,.2f}"],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>USD: %{y:,.2f}<extra></extra>",
                )
            ]
        )
        fig_realized.update_layout(
            title="Realized Profit (USD)",
            height=340,
            margin=dict(l=20, r=20, t=52, b=22),
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(family="Manrope, Segoe UI, sans-serif", color=text_clr),
            dragmode="zoom",
        )
        fig_realized.update_yaxes(showgrid=True, gridcolor=grid_clr, tickfont=dict(color=tick_clr))
        fig_realized.update_xaxes(tickfont=dict(color=tick_clr))

        fig_potential = go.Figure(
            data=[
                go.Bar(
                    x=["Confirmed Profit", "Ad Spend", "Potential Net"],
                    y=[kpis["confirmed_profit_usd"], kpis["spend_usd"], kpis["potential_net_profit_usd"]],
                    marker_color=["#4EE3A3", "#FFA66B", "#B58DFF"],
                    text=[
                        f"${kpis['confirmed_profit_usd']:,.2f}",
                        f"${kpis['spend_usd']:,.2f}",
                        f"${kpis['potential_net_profit_usd']:,.2f}",
                    ],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>USD: %{y:,.2f}<extra></extra>",
                )
            ]
        )
        fig_potential.update_layout(
            title="Potential Profit from Confirmed (USD)",
            height=360,
            margin=dict(l=20, r=20, t=52, b=22),
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font=dict(family="Manrope, Segoe UI, sans-serif", color=text_clr),
            dragmode="zoom",
        )
        fig_potential.update_yaxes(showgrid=True, gridcolor=grid_clr, tickfont=dict(color=tick_clr))
        fig_potential.update_xaxes(tickfont=dict(color=tick_clr))

        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(fig_funnel, use_container_width=True, config=chart_config)
        with cc2:
            st.plotly_chart(fig_realized, use_container_width=True, config=chart_config)
        st.plotly_chart(fig_potential, use_container_width=True, config=chart_config)
    else:
        cc1, cc2 = st.columns(2)
        cc1.image(funnel_png, caption="Order Funnel", use_container_width=True)
        cc2.image(realized_png, caption="Realized Profit (USD)", use_container_width=True)
        st.image(potential_png, caption="Potential Profit from Confirmed (USD)", use_container_width=True)

    # Exports
    st.subheader("Export")
    pdf_bytes = build_pdf_bytes(kpis, fx, funnel_png, realized_png, potential_png)
    xlsx_bytes = build_excel_bytes(kpis, fx, funnel_png, realized_png, potential_png)

    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "⬇️ Download PDF Dashboard",
        data=pdf_bytes,
        file_name=f"ecommerce_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
    export_col2.download_button(
        "⬇️ Download Excel Dashboard",
        data=xlsx_bytes,
        file_name=f"ecommerce_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("Save latest snapshot")
    if st.button("💾 Save latest dashboard to GitHub"):
        try:
            if not both_uploaded:
                st.error("To save EVERYTHING (CSV + KPIs + PDF + Excel), please upload BOTH files first.")
            else:
                orders_bytes = orders_file.getvalue()
                campaigns_bytes = campaigns_file.getvalue()
                daily_bytes = daily_orders_file.getvalue() if daily_orders_file is not None else None
                save_latest_to_github(kpis, pdf_bytes, xlsx_bytes, orders_bytes, campaigns_bytes, daily_bytes)
                st.success("Saved EVERYTHING to GitHub (CSV + KPIs + PDF + Excel).")
        except Exception as e:
            st.error(str(e))





with tab_ai:
    render_ai_summary(
        kpis=kpis,
        kpis_disp=kpis_disp,
        daily_orders_df=daily_orders_df,
        campaigns_df=campaigns_df,
        orders_df=orders_df,
        fx=fx,
        currency=currency,
        last_saved=last_saved,
    )


with tab_daily:
    st.subheader("Daily performance")

    if daily_orders_df is None or getattr(daily_orders_df, "empty", True):
        st.info("Upload **Daily Orders (Taager) XLSX** to see the daily table.")
    else:
        # Month / year filter (based on daily orders dates)
        dtmp = parse_daily_orders(daily_orders_df)
        if "day" not in dtmp.columns or dtmp["day"].isna().all():
            st.warning("Couldn't read dates from the Daily Orders file (missing or invalid **Created At**).")
        else:
            available_days = pd.to_datetime(dtmp["day"].dropna().unique())
            years = sorted({d.year for d in available_days})
            # Default to latest month
            latest = pd.Timestamp(max(available_days))

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                sel_year = st.selectbox("Year", years, index=years.index(latest.year))
            with c2:
                months = list(range(1, 13))
                sel_month = st.selectbox("Month", months, index=months.index(latest.month))
            with c3:
                st.caption("Table includes **all days** in the selected month, even if there were 0 orders / 0 spend.")

            view_mode = st.radio(
                "View",
                ["Daily summary", "Product by date"],
                horizontal=True,
            )

            if view_mode == "Daily summary":
                daily_table = build_daily_table(
                    daily_df=daily_orders_df,
                    campaigns_df=campaigns_df,
                    fx_iqd_per_usd=fx,
                    currency=currency,
                    year=int(sel_year),
                    month=int(sel_month),
                )
            else:
                # Build SKU list for selector (from the selected month)
                dsel = parse_daily_orders(daily_orders_df)
                start = pd.Timestamp(year=int(sel_year), month=int(sel_month), day=1)
                end = (start + pd.offsets.MonthEnd(1)).normalize()
                dsel = dsel[(dsel["day"] >= start) & (dsel["day"] <= end)].copy()

                sku_set = set()
                if "SKUs" in dsel.columns:
                    for v in dsel["SKUs"].dropna().astype(str):
                        for p in [x.strip() for x in v.split(",")]:
                            if p:
                                sku_set.add(p)
                sku_list = sorted(sku_set)

                # Map SKU -> Product Name using Orders file (best-effort)
                sku_to_name = build_sku_to_name_map(orders_df) if "orders_df" in locals() and orders_df is not None else {}

                # Group SKUs under their product names (fallback to SKU when name is unknown)
                name_to_skus = {}
                for sku in sku_list:
                    name = sku_to_name.get(str(sku).strip())
                    label = name if name else str(sku)
                    name_to_skus.setdefault(label, []).append(str(sku).strip())

                product_options = sorted(name_to_skus.keys())

                selected_names = st.multiselect(
                    "Select product(s)",
                    options=product_options,
                    default=product_options[:1] if product_options else [],
                )

                selected_skus = []
                for n in selected_names:
                    selected_skus.extend(name_to_skus.get(n, []))

                daily_table = build_product_by_date_table(
                    daily_df=daily_orders_df,
                    campaigns_df=campaigns_df,
                    fx_iqd_per_usd=fx,
                    currency=currency,
                    year=int(sel_year),
                    month=int(sel_month),
                    selected_skus=selected_skus,
                )

            # Insights
            total_orders = int(daily_table["Orders"].sum())
            total_profit = float(daily_table["Profit"].sum())
            total_spend = float(daily_table["Ad Spend"].sum())
            total_net = float(daily_table["Net Profit"].sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Orders", f"{total_orders:,}")
            k2.metric(f"Profit ({currency})", f"{total_profit:,.2f}")
            k3.metric(f"Ad Spend ({currency})", f"{total_spend:,.2f}")
            k4.metric(f"Net Profit ({currency})", f"{total_net:,.2f}")

            # Stylish + dynamic Net Profit
            def _style_net(val):
                try:
                    v = float(val)
                except:
                    return ""
                if v > 0:
                    return "font-weight:700; color:#19a974;"  # green
                if v < 0:
                    return "font-weight:700; color:#ff4d4f;"  # red
                return "font-weight:700;"

            styled = (
                daily_table.style
                .format({
                    "Orders": "{:,.0f}",
                    "Delivered": "{:,.0f}",
                    "Cancelled": "{:,.0f}",
                    "Returned": "{:,.0f}",
                    "Ad Spend": "{:,.2f}",
                    "Profit": "{:,.2f}",
                    "Net Profit": "{:,.2f}",
                    "Avg Profit / Delivered": "{:,.2f}",
                    "Delivery Rate %": "{:,.1f}%",
                })
                .applymap(_style_net, subset=["Net Profit"])
            )

            st.dataframe(styled, use_container_width=True, height=520)

            with st.expander("More insights (status mix + totals)"):
                # Status mix per month
                mix = {
                    "Delivered %": (daily_table["Delivered"].sum() / total_orders * 100) if total_orders else 0,
                    "Cancelled %": (daily_table["Cancelled"].sum() / total_orders * 100) if total_orders else 0,
                    "Returned %": (daily_table["Returned"].sum() / total_orders * 100) if total_orders else 0,
                    "Avg profit / order": (total_profit / total_orders) if total_orders else 0,
                }
                st.write(mix)

                # Show the raw daily orders for the selected month (optional)
                start_m = pd.Timestamp(year=int(sel_year), month=int(sel_month), day=1)
                end_m = (start_m + pd.offsets.MonthEnd(1)).normalize()
                raw_m = dtmp[(dtmp["day"] >= start_m) & (dtmp["day"] <= end_m)].copy()
                st.caption(f"Raw rows in month: {len(raw_m):,}")
                st.dataframe(raw_m, use_container_width=True, height=240)

with tab_orders:
    st.subheader("Orders overview")

    if orders_df is None:
        st.info("No orders data available yet.")
    else:
        ov = orders_df.copy()
        source_mode = "orders"
        used_daily_fallback = False
        d_from = None
        d_to = None

        date_col = None
        for c in ["order_date", "created_at", "date", "تاريخ_الطلب", "Created At"]:
            if c in ov.columns:
                date_col = c
                break

        if date_col:
            ov["_order_day"] = pd.to_datetime(ov[date_col], errors="coerce").dt.floor("D")
            ov = ov.dropna(subset=["_order_day"])
            if not ov.empty:
                min_d = ov["_order_day"].min().date()
                max_d = ov["_order_day"].max().date()
                d_from, d_to = st.date_input(
                    "Date range",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    key="orders_overview_date_range",
                )
                if isinstance(d_from, tuple) and len(d_from) == 2:
                    d_from, d_to = d_from
                if d_from and d_to:
                    ov = ov[(ov["_order_day"].dt.date >= d_from) & (ov["_order_day"].dt.date <= d_to)].copy()
                st.caption(f"Showing orders from {d_from} to {d_to}.")
            else:
                st.info("No valid order dates found for date filtering.")
        else:
            source_mode = "daily_fallback"
            ddf = parse_daily_orders(daily_orders_df) if daily_orders_df is not None else None
            if ddf is not None and not ddf.empty and "day" in ddf.columns and not ddf["day"].isna().all():
                ddf = ddf.dropna(subset=["day"]).copy()
                min_d = ddf["day"].min().date()
                max_d = ddf["day"].max().date()
                d_from, d_to = st.date_input(
                    "Date range",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    key="orders_overview_date_range",
                )
                if isinstance(d_from, tuple) and len(d_from) == 2:
                    d_from, d_to = d_from
                if d_from and d_to:
                    ddf = ddf[(ddf["day"].dt.date >= d_from) & (ddf["day"].dt.date <= d_to)].copy()
                used_daily_fallback = True
                st.caption(f"Showing {d_from} to {d_to} using Daily Orders dates (Orders file has no date column).")
            else:
                ddf = None
                source_mode = "orders_no_date"
                st.caption("Date range filter unavailable (no date column in Orders and no valid Daily Orders dates).")

        if source_mode == "daily_fallback":
            if ddf is None or ddf.empty:
                st.info("No orders found for the selected date range.")
                requested = confirmed = delivered = returned = delivered_profit = 0.0
            else:
                id_col = get_daily_order_id_col(ddf)
                if id_col is None:
                    ddf["__rowid__"] = range(len(ddf))
                    id_col = "__rowid__"

                ddf["status_clean"] = ddf["Status"].astype(str).str.strip().str.lower() if "Status" in ddf.columns else ""
                delivered_mask = ddf["status_clean"].str.contains("delivered", na=False)
                returned_mask = ddf["status_clean"].str.contains("return", na=False)
                cancelled_mask = ddf["status_clean"].str.contains("cancel", na=False)
                confirmed_mask = ddf["status_clean"].str.contains("confirm", na=False) | delivered_mask | returned_mask

                requested = float(ddf[id_col].nunique())
                confirmed = float(ddf.loc[confirmed_mask, id_col].nunique())
                delivered = float(ddf.loc[delivered_mask, id_col].nunique())
                returned = float(ddf.loc[returned_mask, id_col].nunique())
                cancelled = float(ddf.loc[cancelled_mask, id_col].nunique())

                if "Order Profit" in ddf.columns:
                    ddf["Order Profit"] = pd.to_numeric(ddf["Order Profit"], errors="coerce").fillna(0)
                    delivered_profit_iqd = float(ddf.loc[delivered_mask, "Order Profit"].sum())
                else:
                    delivered_profit_iqd = 0.0
                delivered_profit = delivered_profit_iqd if currency == "IQD" else iqd_to_usd(delivered_profit_iqd, fx)
        else:
            if ov.empty:
                st.info("No orders found for the selected date range.")

            for c in ["requested_units", "confirmed_units", "delivered_units", "returned_units"]:
                if c in ov.columns:
                    ov[c] = pd.to_numeric(ov[c], errors="coerce").fillna(0)

            if currency == "IQD":
                ov["delivered_profit_disp"] = pd.to_numeric(ov.get("delivered_profit_iqd", 0), errors="coerce").fillna(0)
            else:
                ov["delivered_profit_disp"] = pd.to_numeric(ov.get("delivered_profit_usd", 0), errors="coerce").fillna(0)

            requested = float(ov.get("requested_units", pd.Series(dtype=float)).sum())
            confirmed = float(ov.get("confirmed_units", pd.Series(dtype=float)).sum())
            delivered = float(ov.get("delivered_units", pd.Series(dtype=float)).sum())
            returned = float(ov.get("returned_units", pd.Series(dtype=float)).sum())
            delivered_profit = float(ov["delivered_profit_disp"].sum())
            cancelled = max(requested - confirmed, 0.0)

        confirmation_rate = (confirmed / requested) if requested else 0.0
        delivery_rate = (delivered / confirmed) if confirmed else 0.0
        return_rate = (returned / delivered) if delivered else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Orders", f"{int(requested):,}")
        m2.metric("Confirmed", f"{int(confirmed):,}", f"{confirmation_rate*100:.1f}%")
        m3.metric("Delivered", f"{int(delivered):,}", f"{delivery_rate*100:.1f}%")
        m4.metric("Returned", f"{int(returned):,}", f"{return_rate*100:.1f}%")
        m5.metric(f"Delivered Profit ({currency})", f"{delivered_profit:,.2f}")

        if go is not None:
            c1, c2 = st.columns(2)

            with c1:
                fig_funnel = go.Figure(
                    data=[
                        go.Bar(
                            x=["Orders", "Confirmed", "Delivered", "Returned"],
                            y=[requested, confirmed, delivered, returned],
                            marker_color=["#64D2FF", "#4EE3A3", "#B58DFF", "#FFA66B"],
                            text=[f"{int(requested):,}", f"{int(confirmed):,}", f"{int(delivered):,}", f"{int(returned):,}"],
                            textposition="outside",
                            hovertemplate="<b>%{x}</b><br>Units: %{y:,}<extra></extra>",
                        )
                    ]
                )
                fig_funnel.update_layout(
                    title="Order Flow",
                    height=320,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(9,16,26,0.70)",
                    plot_bgcolor="rgba(9,16,26,0.70)",
                    font=dict(family="Manrope, Segoe UI, sans-serif", color="#EAF2F8"),
                )
                fig_funnel.update_yaxes(showgrid=True, gridcolor="rgba(184,202,217,0.16)")
                st.plotly_chart(
                    fig_funnel,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset", "displaylogo": False},
                )

            with c2:
                fig_mix = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Delivered", "Returned", "Open/Other"],
                            values=[
                                max(delivered - returned, 0),
                                returned,
                                max(requested - delivered - cancelled, 0) + cancelled,
                            ],
                            hole=0.58,
                            marker=dict(colors=["#4EE3A3", "#FFA66B", "#5D7088"]),
                            hovertemplate="<b>%{label}</b><br>%{value:,} units (%{percent})<extra></extra>",
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_mix.update_layout(
                    title="Status Mix",
                    height=320,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(9,16,26,0.70)",
                    plot_bgcolor="rgba(9,16,26,0.70)",
                    font=dict(family="Manrope, Segoe UI, sans-serif", color="#EAF2F8"),
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_mix,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset", "displaylogo": False},
                )

with tab_ads:
    st.subheader("Ads details")

    if campaigns_df is None:
        st.info("No campaigns data available yet.")
    else:
        ads_view = campaigns_df.copy()

        # Keep relevant columns only (if they exist)
        preferred_cols = [
            "Reporting starts", "Reporting ends",
            "Campaign name", "Campaign delivery",
            "Amount spent (USD)", "Impressions", "Reach",
            "Results", "Cost per results", "Result indicator",
        ]
        cols = [c for c in preferred_cols if c in ads_view.columns]

        # Add derived metrics
        if "Amount spent (USD)" in ads_view.columns and "Impressions" in ads_view.columns:
            ads_view["CPM (USD)"] = ads_view["Amount spent (USD)"] / ads_view["Impressions"].replace({0: pd.NA}) * 1000
            cols.append("CPM (USD)") if "CPM (USD)" not in cols else None

        with st.expander("Filters", expanded=False):
            campaign_search = st.text_input("Search campaign name")
            min_spend = st.number_input("Min spend (USD)", min_value=0.0, value=0.0, step=1.0)
            sort_by = st.selectbox(
                "Sort by",
                options=[c for c in ["Amount spent (USD)", "Impressions", "Reach", "Results"] if c in ads_view.columns],
            )
            desc = st.checkbox("Sort descending", value=True, key="ads_desc")

        filtered = ads_view
        if campaign_search and "Campaign name" in filtered.columns:
            filtered = filtered[filtered["Campaign name"].astype(str).str.contains(campaign_search, case=False, na=False)]

        if "Amount spent (USD)" in filtered.columns:
            filtered = filtered[filtered["Amount spent (USD)"] >= min_spend]

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=not desc)

        st.caption(f"{len(filtered):,} rows")
        st.dataframe(filtered[cols], use_container_width=True)

with tab_campaigns:
    st.subheader("Campaigns analytics")
    if campaigns_df is None:
        st.info("No campaigns data available yet.")
    else:
        campaigns_metric_explorer_sku(campaigns_df, orders_df)


with tab_product:
    st.subheader("Product Deep Dive")
    product_deep_dive(orders_df, campaigns_df, fx, currency)
