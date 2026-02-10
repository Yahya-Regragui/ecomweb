import io
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader


import base64
import json
import requests



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
    except Exception:
        return None

    return {
        "generated_at": payload.get("generated_at"),
        "kpis": payload.get("kpis"),
        "orders_csv_bytes": orders_bytes,
        "campaigns_csv_bytes": campaigns_bytes,
    }


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
                          orders_csv_bytes: bytes, campaigns_csv_bytes: bytes):
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("GITHUB_REPO", None)
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    if not token or not repo:
        raise RuntimeError("Missing GitHub secrets. Please set GITHUB_TOKEN and GITHUB_REPO in Streamlit Secrets.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # 0) Save raw inputs (latest only)
    github_put_file(token, repo, "data/latest_orders.csv", orders_csv_bytes, f"Update latest Orders CSV ({now})", branch)
    github_put_file(token, repo, "data/latest_campaigns.csv", campaigns_csv_bytes, f"Update latest Campaigns CSV ({now})", branch)

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

def money(x: float) -> str:
    return f"${x:,.2f}"

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def safe_ratio(n: float, d: float):
    return None if d == 0 else n / d

def fmt_ratio(x):
    return "N/A" if x is None else f"{x:.2f}"

def fmt_money_or_na(x):
    return "N/A" if x is None else money(x)

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

st.title("E-commerce Dashboard")
st.caption("Drop Orders CSV + Campaigns CSV → dashboard updates instantly. Export to PDF or Excel.")

with st.sidebar:
    st.subheader("Inputs")
    fx = st.number_input("FX rate (IQD per 1 USD)", min_value=1.0, value=1310.0, step=1.0)
    orders_file = st.file_uploader("Orders CSV (Arabic headers)", type=["csv"])
    campaigns_file = st.file_uploader("Campaigns CSV (Meta export)", type=["csv"])

both_uploaded = orders_file is not None and campaigns_file is not None
one_uploaded = (orders_file is not None) ^ (campaigns_file is not None)

orders_df = None
campaigns_df = None

# CASE A: both uploaded -> use uploads
if both_uploaded:
    try:
        orders_df = pd.read_csv(orders_file, encoding="utf-8-sig")
        campaigns_df = pd.read_csv(campaigns_file, encoding="utf-8-sig")
        orders_df, campaigns_df, kpis = parse_inputs(orders_df, campaigns_df, fx)
        data_source = "uploads"
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
        orders_df, campaigns_df, kpis = parse_inputs(orders_df, campaigns_df, fx)
        data_source = "github"
    except Exception as e:
        st.error(f"Loaded from GitHub but failed to parse: {e}")
        st.stop()

if one_uploaded and data_source == "github":
    st.info("Only one file uploaded → showing LAST SAVED data from GitHub. Upload the missing file to refresh.")

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

# --- Tabs ---
tab_dashboard, tab_orders, tab_ads = st.tabs(["📊 Dashboard", "📦 Orders details", "📣 Ads details"])

with tab_dashboard:
    if data_source == "github" and snap is not None and snap.get("generated_at"):
        st.info(f"Showing last saved snapshot from GitHub • {snap['generated_at']}")
    elif data_source == "uploads":
        st.success("Showing uploaded files (not yet saved).")

    # Dashboard cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Confirmed Profit (USD)", money(kpis["confirmed_profit_usd"]), f"{int(kpis['confirmed_units']):,} confirmed")
    col2.metric("Delivered Profit (USD)", money(kpis["delivered_profit_usd"]), f"{int(kpis['delivered_units']):,} delivered")
    col3.metric("Ad Spend (USD)", money(kpis["spend_usd"]))

    col4, col5, col6 = st.columns(3)
    col4.metric("Net Profit After Ads", money(kpis["net_profit_usd"]))
    col5.metric("Potential Net Profit", money(kpis["potential_net_profit_usd"]))
    col6.metric("ROAS (Realized)", fmt_ratio(kpis["roas_real"]), f"Potential: {fmt_ratio(kpis['roas_potential'])}")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmation Rate", pct(kpis["confirmation_rate"]))
    c2.metric("Delivery Rate", pct(kpis["delivery_rate"]))
    c3.metric("Return Rate", pct(kpis["return_rate"]))
    c4.metric("CPM", fmt_money_or_na(kpis["cpm"]))

    # Charts
    funnel_png, realized_png, potential_png = make_charts_bytes(kpis)

    st.subheader("Charts")
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
                save_latest_to_github(kpis, pdf_bytes, xlsx_bytes, orders_bytes, campaigns_bytes)
                st.success("Saved EVERYTHING to GitHub (CSV + KPIs + PDF + Excel).")
        except Exception as e:
            st.error(str(e))



with tab_orders:
    st.subheader("Orders details")

    if orders_df is None:
        st.info("No orders data available yet.")
    else:
        # Keep only relevant columns + add derived rates
        orders_view = orders_df.copy()

        # Add rates (safe)
        orders_view["confirmation_rate"] = orders_view["confirmed_units"] / orders_view["requested_units"].replace({0: pd.NA})
        orders_view["delivery_rate"] = orders_view["delivered_units"] / orders_view["confirmed_units"].replace({0: pd.NA})
        orders_view["return_rate"] = orders_view["returned_units"] / orders_view["delivered_units"].replace({0: pd.NA})

        # Choose relevant columns that actually exist
        preferred_cols = [
            "كود_المنتج", "اسم_المنتج",  # if present in your CSV
            "requested_units", "confirmed_units", "delivered_units", "returned_units",
            "confirmation_rate", "delivery_rate", "return_rate",
            "confirmed_profit_usd", "delivered_profit_usd",
        ]
        cols = [c for c in preferred_cols if c in orders_view.columns]

        # Simple filters
        with st.expander("Filters", expanded=False):
            search = st.text_input("Search product (code or name)")
            min_delivered = st.number_input("Min delivered units", min_value=0, value=0, step=1)
            sort_by = st.selectbox(
                "Sort by",
                options=[c for c in ["delivered_profit_usd", "confirmed_profit_usd", "delivered_units", "confirmed_units"] if c in orders_view.columns],
            )
            desc = st.checkbox("Sort descending", value=True)

        filtered = orders_view
        if search:
            # search across code/name if present
            mask = pd.Series(False, index=filtered.index)
            if "كود_المنتج" in filtered.columns:
                mask |= filtered["كود_المنتج"].astype(str).str.contains(search, case=False, na=False)
            if "اسم_المنتج" in filtered.columns:
                mask |= filtered["اسم_المنتج"].astype(str).str.contains(search, case=False, na=False)
            filtered = filtered[mask]

        if "delivered_units" in filtered.columns:
            filtered = filtered[filtered["delivered_units"] >= min_delivered]

        if sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=not desc)

        st.caption(f"{len(filtered):,} rows")
        st.dataframe(
            filtered[cols],
            use_container_width=True
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
