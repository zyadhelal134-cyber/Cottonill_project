"""
Cottonil - Executive Business Intelligence Dashboard
====================================================
Run with:   streamlit run app.py
Expected layout:
    project/
        app.py
        02_TABLES/   sales_orders.csv, branches.csv, customers.csv, ...

Design notes
------------
* Every CSV in 02_TABLES is loaded once (cached) and its column names are normalised
  (lower-case, snake_case).
* Columns are located with candidate lists + keyword fallbacks, so slightly different
  schemas still work. When something is missing the page shows a warning instead of failing.
* "Net sales" is the order amount column found in sales_orders (total_amount, net_amount, ...).
  If the orders table has no usable amount, it is rebuilt from sales_order_items.
* Partial months (first/last month not fully covered by the data) are excluded from
  MoM, YoY and peak calculations and are drawn with a hatched pattern in charts.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------------
# Theme / constants
# --------------------------------------------------------------------------------------
PRIMARY = "#6236ff"
PRIMARY_DARK = "#3d1fb3"
PRIMARY_LIGHT = "#b9a6ff"
BACKGROUND = "#f4f2fc"
INK = "#1d1740"
MUTED = "#6b6890"
POSITIVE = "#12a36b"
NEGATIVE = "#e5484d"
ACCENT = "#ff8a3d"
PALETTE = [PRIMARY, "#9b7cff", "#c9b8ff", ACCENT, "#2bb3a3", "#f2c14e", "#e5484d", "#5b6ee1"]
CURRENCY = "EGP"
DATA_DIR_NAME = "02_TABLES"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

PAGES = [
    "Executive Overview",
    "Sales Performance",
    "Branches & Geography",
    "Products & Profitability",
    "Customer Intelligence",
    "Returns & Delivery",
    "Inventory & Transfers",
    "Procurement & Imports",
    "Marketing ROI",
    "Workforce & Costs",
    "AI & RAG Assistant",
]
DEFAULT_PAGE = "Sales Performance"

# Column candidates (all lower snake_case)
DATE_C = ["order_date", "date", "order_datetime", "order_timestamp", "created_at", "invoice_date",
          "sale_date", "transaction_date", "ordered_at", "order_time"]
AMOUNT_C = ["total_amount", "net_amount", "net_sales", "total", "grand_total", "order_total",
            "total_price", "amount", "revenue", "final_amount", "total_net"]
AMOUNT_EXCLUDE = ("discount", "tax", "vat", "shipping", "count", "qty", "quantity", "cost",
                  "refund", "paid", "_id")
ORDER_ID_C = ["order_id", "sales_order_id", "order_no", "order_number", "id"]
ITEM_ORDER_ID_C = ["order_id", "sales_order_id", "order_no", "order_number"]
BRANCH_C = ["branch_id", "store_id", "branch_code", "branch", "branch_name", "store", "location_id"]
CUSTOMER_C = ["customer_id", "client_id", "customer_code", "customer"]
CHANNEL_C = ["channel", "sales_channel", "order_channel", "channel_name", "source", "order_type",
             "order_source"]
PAYMENT_C = ["payment_method", "payment_type", "payment", "payment_mode", "payment_method_name"]
STATUS_C = ["status", "order_status", "state"]
LINE_AMT_C = ["line_total", "net_line_total", "total_price", "total_amount", "line_amount",
              "subtotal", "amount"]
QTY_C = ["quantity", "qty", "units"]
PRICE_C = ["unit_price", "price", "selling_price"]
DISC_C = ["discount_amount", "discount_value"]
PRODUCT_C = ["product_id", "sku", "item_id", "product_code", "product"]


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def _norm(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def find_col(df: Optional[pd.DataFrame], candidates: Iterable[str], contains: Iterable[str] = (),
             exclude: Iterable[str] = ()) -> Optional[str]:
    """Return the first matching column: exact candidates first, then keyword matches."""
    if df is None:
        return None
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for kw in contains:
        for c in cols:
            if kw in c and not any(x in c for x in exclude):
                return c
    return None


def get_table(tables: dict, names: Iterable[str], contains: Iterable[str] = (),
              exclude: Iterable[str] = ()) -> Optional[pd.DataFrame]:
    for n in names:
        if n in tables:
            return tables[n]
    for kw in contains:
        for n, df in tables.items():
            if kw in n and not any(x in n for x in exclude):
                return df
    return None


def to_dt(series: pd.Series) -> pd.Series:
    try:
        d = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        d = pd.to_datetime(series, errors="coerce", utc=True)
    return d.dt.tz_localize(None)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_money(v: float, currency: bool = True) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    prefix = f"{CURRENCY} " if currency else ""
    a = abs(v)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{prefix}{v / div:,.2f}{suffix}"
    return f"{prefix}{v:,.0f}"


def fmt_pct(v: float, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def norm_channel(v: object) -> str:
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "null"):
        return "Unclassified"
    if re.search(r"whole|b2b|distribut|dealer|bulk|corporate", s):
        return "Wholesale"
    if re.search(r"e-?com|online|web|\bapp\b|shopify|woo|digital|social|mobile", s):
        return "E-commerce"
    if re.search(r"pos|retail|store|shop|branch|walk|showroom|offline|outlet", s):
        return "Retail POS"
    return str(v).strip().title()


def line_amount(items: pd.DataFrame) -> Optional[pd.Series]:
    lc = find_col(items, LINE_AMT_C)
    if lc:
        s = to_num(items[lc])
        if s.notna().any():
            return s.fillna(0)
    q = find_col(items, QTY_C, ("qty", "quantity"))
    p = find_col(items, PRICE_C, ("price",))
    if q and p:
        s = to_num(items[q]).fillna(0) * to_num(items[p]).fillna(0)
        d = find_col(items, DISC_C)
        if d:
            s = s - to_num(items[d]).fillna(0)
        return s
    return None


# --------------------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------------------
def resolve_data_dir() -> Optional[Path]:
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        p = base / DATA_DIR_NAME
        if p.is_dir():
            return p
    return None


def dir_signature(d: Path) -> str:
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in sorted(d.glob("*.csv")))


@st.cache_data(show_spinner="Loading tables...")
def load_tables(dir_str: str, signature: str) -> tuple[dict, list]:
    tables, errors = {}, []
    for f in sorted(Path(dir_str).glob("*.csv")):
        df = None
        for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
            try:
                df = pd.read_csv(f, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
                break
            except Exception as exc:  # noqa: BLE001 - keep dashboard alive on a bad file
                errors.append(f"{f.name}: {exc}")
                break
        if df is None:
            if not any(e.startswith(f.name) for e in errors):
                errors.append(f"{f.name}: could not decode file")
            continue
        df.columns = [_norm(c) for c in df.columns]
        tables[f.stem.lower()] = df
    return tables, errors


def empty_model(note: str) -> dict:
    fact = pd.DataFrame(columns=["order_id", "date", "amount", "branch", "channel", "payment", "customer_id"])
    fact["date"] = pd.to_datetime(fact["date"])
    return {"fact": fact, "items": pd.DataFrame(), "notes": [note], "branch_map": {}, "real_order_ids": False}


@st.cache_data(show_spinner="Preparing sales model...")
def build_model(signature: str, _tables: dict) -> dict:
    """Builds the unified sales fact table (one row per order) plus line items."""
    notes: list[str] = []
    orders = get_table(_tables, ["sales_orders", "sales_order", "orders", "sales"], ("order",),
                       ("item", "line", "return", "purchase", "transfer"))
    items = get_table(_tables, ["sales_order_items", "sales_order_item", "order_items", "sales_items"],
                      ("item",), ("purchase", "transfer", "stock", "catalog", "inventory"))
    branches = get_table(_tables, ["branches", "branch", "stores"], ("branch",), ("transfer",))
    customers = get_table(_tables, ["customers", "customer"], ("customer",), ("order", "segment"))
    if orders is None or orders.empty:
        return empty_model("No sales orders table was found in 02_TABLES.")

    o = orders.copy()
    dcol = find_col(o, DATE_C, ("date", "time"))
    if dcol is None:
        return empty_model("The sales orders table has no recognisable date column.")
    o["_date"] = to_dt(o[dcol])

    # order id
    kcol = find_col(o, ORDER_ID_C)
    real_ids = kcol is not None
    o["_oid"] = o[kcol].astype(str) if kcol else o.index.astype(str)

    # items helper (also used for amount fallback)
    items_df = pd.DataFrame()
    if items is not None and not items.empty:
        ik = find_col(items, ITEM_ORDER_ID_C)
        la = line_amount(items)
        if ik and la is not None:
            pk = find_col(items, PRODUCT_C)
            qc = find_col(items, QTY_C, ("qty", "quantity"))
            items_df = pd.DataFrame({
                "order_id": items[ik].astype(str),
                "product_id": items[pk].astype(str) if pk else "Unknown",
                "qty": to_num(items[qc]).fillna(0) if qc else 1.0,
                "amount": la,
            })

    # amount
    acol = find_col(o, AMOUNT_C, ("total", "amount", "revenue"), AMOUNT_EXCLUDE)
    amount = to_num(o[acol]) if acol else pd.Series(np.nan, index=o.index)
    if (not amount.notna().any()) and not items_df.empty and real_ids:
        by_order = items_df.groupby("order_id")["amount"].sum()
        amount = o["_oid"].map(by_order)
        notes.append("Order amount was rebuilt from sales_order_items (no usable amount column in sales_orders).")
    elif not amount.notna().any():
        return empty_model("No usable amount column was found in sales_orders or sales_order_items.")
    o["_amount"] = amount.fillna(0)

    # status filter
    scol = find_col(o, STATUS_C)
    if scol:
        bad = o[scol].astype(str).str.lower().str.contains(r"cancel|void|fail|reject|draft|abandon", regex=True)
        if bad.any():
            notes.append(f"{int(bad.sum()):,} cancelled/void orders were excluded.")
        o = o[~bad]

    # branch
    branch_map: dict = {}
    if branches is not None and not branches.empty:
        bid = find_col(branches, ["branch_id", "id", "store_id", "branch_code"])
        bname = find_col(branches, ["branch_name", "name", "store_name", "branch", "title"])
        if bid and bname:
            branch_map = dict(zip(branches[bid].astype(str), branches[bname].astype(str)))
    bcol = find_col(o, BRANCH_C)
    if bcol:
        raw = o[bcol].astype(str)
        o["_branch"] = raw.map(branch_map).fillna(raw).replace({"nan": "Unknown"})
    else:
        o["_branch"] = "All branches"
        notes.append("No branch column found in sales_orders.")

    # customer
    ccol = find_col(o, CUSTOMER_C)
    o["_cust"] = o[ccol].astype(str) if ccol else np.nan
    if ccol:
        o.loc[o[ccol].isna(), "_cust"] = np.nan

    # channel
    chcol = find_col(o, CHANNEL_C)
    if chcol:
        o["_channel"] = o[chcol].map(norm_channel)
    else:
        o["_channel"] = None
        if customers is not None and ccol:
            cid = find_col(customers, ["customer_id", "id", "customer_code"])
            ctype = find_col(customers, ["customer_type", "type", "segment", "customer_segment"])
            if cid and ctype:
                m = dict(zip(customers[cid].astype(str), customers[ctype].astype(str)))
                o["_channel"] = o["_cust"].map(m).map(lambda v: norm_channel(v) if pd.notna(v) else None)
        if o["_channel"].isna().all():
            def infer(b: str) -> str:
                s = str(b).lower()
                if re.search(r"whole|b2b", s):
                    return "Wholesale"
                if re.search(r"e-?com|online|web", s):
                    return "E-commerce"
                return "Retail POS"
            o["_channel"] = o["_branch"].map(infer)
            notes.append("No channel column found: channel was inferred from branch names (default Retail POS).")
        o["_channel"] = o["_channel"].fillna("Unclassified")

    # payment
    pcol = find_col(o, PAYMENT_C)
    if pcol:
        o["_payment"] = o[pcol].astype(str).str.strip().str.title().replace({"Nan": "Unknown", "": "Unknown"})
    else:
        pay = get_table(_tables, ["payments", "sales_payments", "order_payments"], ("payment",), ("supplier", "purchase"))
        o["_payment"] = "Unknown"
        if pay is not None and real_ids:
            pk = find_col(pay, ITEM_ORDER_ID_C)
            pm = find_col(pay, PAYMENT_C + ["method", "type"], ("method",))
            if pk and pm:
                first = pay.drop_duplicates(pk).set_index(pk)[pm].astype(str)
                first.index = first.index.astype(str)
                o["_payment"] = o["_oid"].map(first).fillna("Unknown").str.title()
        if (o["_payment"] == "Unknown").all():
            notes.append("No payment method column found.")

    fact = pd.DataFrame({
        "order_id": o["_oid"], "date": o["_date"], "amount": o["_amount"], "branch": o["_branch"],
        "channel": o["_channel"], "payment": o["_payment"], "customer_id": o["_cust"],
    })
    bad_dates = int(fact["date"].isna().sum())
    if bad_dates:
        notes.append(f"{bad_dates:,} orders with unreadable dates were dropped.")
    fact = fact.dropna(subset=["date"]).reset_index(drop=True)
    return {"fact": fact, "items": items_df, "notes": notes, "branch_map": branch_map, "real_order_ids": real_ids}


# --------------------------------------------------------------------------------------
# Monthly model
# --------------------------------------------------------------------------------------
def month_table(scope: pd.DataFrame, data_min: pd.Timestamp, data_max: pd.Timestamp) -> pd.DataFrame:
    """Monthly sales with completeness flag, same-month-last-year, YoY and MoM."""
    idx = pd.date_range(data_min.to_period("M").to_timestamp(), data_max.to_period("M").to_timestamp(), freq="MS")
    if scope.empty:
        sales = pd.Series(0.0, index=idx)
    else:
        sales = scope.set_index("date")["amount"].resample("MS").sum().reindex(idx, fill_value=0.0)
    m = pd.DataFrame({"sales": sales})
    month_end = m.index + pd.offsets.MonthEnd(0)
    m["complete"] = (data_min.normalize() <= m.index) & (data_max.normalize() >= month_end)
    prior_ok = m["complete"].shift(12, fill_value=False)
    m["prior"] = np.where(prior_ok, m["sales"].shift(12), np.nan)
    m["yoy"] = np.where(m["complete"] & (m["prior"] > 0), (m["sales"] / m["prior"] - 1) * 100, np.nan)
    prev_ok = m["complete"].shift(1, fill_value=False)
    prev = m["sales"].shift(1)
    m["mom"] = np.where(m["complete"] & prev_ok & (prev > 0), (m["sales"] / prev - 1) * 100, np.nan)
    return m


def month_label(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def inject_css() -> None:
  st.markdown(
      f"""
    <style>
    .stApp {{ background: {BACKGROUND}; color: {INK}; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{ padding-top: 4rem !important; max-width: 1500px; }}
    [data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #e6e1f7; }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem !important; }}
    h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 16px;
        border: 1px solid #e9e4fb; }}
    .brand {{ display:flex; align-items:center; gap:12px; padding: 4px 4px 14px 4px; }}
    .brand-mark {{ width:42px; height:42px; border-radius:12px; background:{PRIMARY}; color:#fff;
        display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.3rem; }}
    .brand-name {{ font-weight:800; font-size:1.15rem; color:{INK}; line-height:1.1; }}
    .brand-name sup {{ font-size:.6rem; }}
    .brand-tag {{ font-size:.68rem; letter-spacing:.14em; color:{PRIMARY}; font-weight:600; }}
    .api-pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px;
        font-size:.8rem; font-weight:600; background:#e8f8f1; color:#0b7a52; }}
    .api-pill.off {{ background:#fdecec; color:#b42327; }}
    .api-dot {{ width:8px; height:8px; border-radius:50%; background:{POSITIVE}; }}
    .api-pill.off .api-dot {{ background:{NEGATIVE}; }}
    .page-title {{ font-size:1.7rem; font-weight:800; margin:12px 0 2px 0; color:{INK}; position:relative; z-index:2; }}
    .page-sub {{ color:{MUTED}; margin:2px 0 14px 0; font-size:.95rem; }}
    .kpi {{ background:#fff; border:1px solid #e9e4fb; border-radius:16px; padding:16px 18px; height:100%; }}
    .kpi.primary {{ background:{PRIMARY}; border-color:{PRIMARY}; }}
    .kpi .l {{ font-size:.8rem; color:{MUTED}; font-weight:600; }}
    .kpi .v {{ font-size:1.65rem; font-weight:800; color:{INK}; margin-top:4px; }}
    .kpi .s {{ font-size:.8rem; color:{MUTED}; margin-top:2px; }}
    .kpi.primary .l, .kpi.primary .v, .kpi.primary .s {{ color:#fff; }}
    .kpi .d {{ display:inline-block; font-size:.78rem; font-weight:700; padding:2px 8px; border-radius:999px;
        margin-top:6px; }}
    .kpi .d.pos {{ background:#e8f8f1; color:#0b7a52; }}
    .kpi .d.neg {{ background:#fdecec; color:#b42327; }}
    .kpi .d.neu {{ background:#efeafd; color:{PRIMARY_DARK}; }}
    .card-title {{ font-weight:700; font-size:1.05rem; margin:0 0 2px 0; color:{INK}; }}
    .card-sub {{ color:{MUTED}; font-size:.82rem; margin-bottom:6px; }}
    </style>
    """,
      unsafe_allow_html=True,
  )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="page-title">{title}</p><p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)


def card_title(title: str, sub: str = "") -> None:
    st.markdown(f'<p class="card-title">{title}</p><p class="card-sub">{sub}</p>', unsafe_allow_html=True)


def kpi_card(col, label: str, value: str, sub: str = "", delta: Optional[float] = None,
             delta_text: Optional[str] = None, primary: bool = False) -> None:
    badge = ""
    if delta_text is not None:
        cls = "neu" if delta is None or pd.isna(delta) else ("pos" if delta >= 0 else "neg")
        badge = f'<div class="d {cls}">{delta_text}</div>'
    col.markdown(
        f'<div class="kpi {"primary" if primary else ""}"><div class="l">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div>{badge}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK), colorway=PALETTE,
        showlegend=legend, hovermode="x unified" if legend else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#ece8fa", zeroline=False)
    return fig


def show_fig(fig: go.Figure, key: Optional[str] = None) -> None:
    cfg = {"displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def require_sales(ctx: SimpleNamespace) -> bool:
    if ctx.fact.empty:
        st.warning("No sales data is available. Check that 02_TABLES/sales_orders.csv exists and has a date "
                   "and amount column.")
        for n in ctx.notes:
            st.caption(f"- {n}")
        return False
    if ctx.rng.empty:
        st.warning("No orders match the current filters. Widen the date range or clear the branch/channel filters.")
        return False
    return True


# --------------------------------------------------------------------------------------
# Shared analytics helpers
# --------------------------------------------------------------------------------------
def get_returns(ctx: SimpleNamespace) -> pd.DataFrame:
    t = get_table(ctx.tables, ["sales_returns", "sales_return", "returns"], ("return",),
                  ("item", "purchase", "vendor", "supplier"))
    if t is None or t.empty:
        return pd.DataFrame()
    dcol = find_col(t, ["return_date", "date", "created_at", "returned_at"] + DATE_C, ("date",))
    if dcol is None:
        return pd.DataFrame()
    acol = find_col(t, ["refund_amount", "return_amount", "total_refund", "total_amount", "amount", "credit_amount"],
                    ("refund", "amount"), ("_id",))
    rcol = find_col(t, ["reason", "return_reason", "reason_code"], ("reason",))
    qcol = find_col(t, QTY_C, ("qty", "quantity"))
    out = pd.DataFrame({
        "date": to_dt(t[dcol]),
        "amount": to_num(t[acol]).fillna(0) if acol else 0.0,
        "reason": t[rcol].astype(str).str.title() if rcol else "Not specified",
        "qty": to_num(t[qcol]).fillna(0) if qcol else 1.0,
    }).dropna(subset=["date"])
    return out[(out["date"] >= ctx.start) & (out["date"] < ctx.end + pd.Timedelta(days=1))]


def product_summary(ctx: SimpleNamespace) -> pd.DataFrame:
  it = ctx.items
  if it.empty:
    return pd.DataFrame()
  if ctx.real_order_ids:
    it = it[it["order_id"].isin(ctx.rng["order_id"])]
  if it.empty:
    return pd.DataFrame()

  g = it.groupby("product_id", as_index=False).agg(
      units=("qty", "sum"), revenue=("amount", "sum")
  )
  g["product"] = g["product_id"]
  g["category"] = "Uncategorised"
  g["cost_total"] = np.nan

  cat = get_table(
      ctx.tables,
      [
          "products_catalog",
          "products",
          "product_catalog",
          "catalog",
          "items_catalog",
          "product_dim",
          "dim_product",
      ],
      ("product", "catalog", "sku", "item"),
      ("order", "sales", "stock", "transfer", "return"),
  )

  if cat is not None and not cat.empty:
    kid = find_col(
        cat,
        [
            "product_id",
            "sku",
            "id",
            "product_code",
            "item_id",
            "item_code",
            "code",
        ],
    )
    if kid:
      c = cat.copy()
      c["_k"] = c[kid].astype(str).str.strip()
      g["product_id"] = g["product_id"].astype(str).str.strip()

      name = find_col(
          c,
          [
              "product_name",
              "name",
              "title",
              "description",
              "product_title",
              "item_name",
              "sku_name",
          ],
          ("name", "title", "desc"),
      )
      category = find_col(
          c,
          [
              "category",
              "product_category",
              "cat_name",
              "department",
              "family",
              "group",
              "line",
              "type",
          ],
          ("categ", "depart", "famil", "group", "line"),
      )
      cost = find_col(
          c,
          [
              "unit_cost",
              "cost",
              "cost_price",
              "standard_cost",
              "purchase_price",
              "cogs",
          ],
          ("cost", "cogs"),
          ("price",),
      )

      g = g.merge(
          c.drop_duplicates("_k"),
          left_on="product_id",
          right_on="_k",
          how="left",
          suffixes=("", "_cat"),
      )

      if name:
        ncol = name if name in g.columns else name + "_cat"
        prod_vals = (
            g[ncol]
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .fillna(g["product_id"])
            .astype(str)
        )
        g["product"] = prod_vals

      if category:
        cat_col = category if category in g.columns else category + "_cat"
        cat_vals = (
            g[cat_col]
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .fillna("Uncategorised")
            .astype(str)
            .str.strip()
        )
        g["category"] = cat_vals

      if cost:
        cc = cost if cost in g.columns else cost + "_cat"
        g["cost_total"] = to_num(g[cc]) * g["units"]

  g["profit"] = g["revenue"] - g["cost_total"]
  g["margin"] = np.where(
      g["revenue"] > 0, g["profit"] / g["revenue"] * 100, np.nan
  )
  return g[[
      "product_id",
      "product",
      "category",
      "units",
      "revenue",
      "cost_total",
      "profit",
      "margin",
  ]]


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_sales(ctx: SimpleNamespace) -> None:
    page_header("Sales Performance", "Net sales trend, growth and where the revenue comes from.")
    if not require_sales(ctx):
        return
    vm = ctx.view_m
    comp = vm[vm["complete"]]
    partial = vm[~vm["complete"]]
    total = ctx.rng["amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total net sales", fmt_money(total),
             f"{ctx.start:%d %b %Y} to {ctx.end:%d %b %Y}", primary=True)
    if comp.empty:
        kpi_card(c2, "Last complete month", "n/a", "No complete month in range")
        kpi_card(c3, "MoM growth", "n/a", "Needs two consecutive complete months")
        kpi_card(c4, "Peak monthly sales", "n/a", "No complete month in range")
    else:
        last = comp.iloc[-1]
        last_ts = comp.index[-1]
        prev_ts = last_ts - pd.offsets.MonthBegin(1)
        kpi_card(c2, "Last complete month", fmt_money(last["sales"]), month_label(last_ts))
        mom = last["mom"]
        kpi_card(c3, "MoM growth", fmt_pct(mom), f"{month_label(last_ts)} vs {month_label(prev_ts)}",
                 delta=mom, delta_text=("Up" if mom >= 0 else "Down") if pd.notna(mom) else "Previous month incomplete")
        pk_ts = comp["sales"].idxmax()
        kpi_card(c4, "Peak monthly sales", fmt_money(comp.loc[pk_ts, "sales"]), month_label(pk_ts))

    if not partial.empty:
        parts = []
        for ts in partial.index:
            ms, me = ts, ts + pd.offsets.MonthEnd(0)
            lo, hi = max(ctx.data_min.normalize(), ms), min(ctx.data_max.normalize(), me)
            parts.append(f"{month_label(ts)} (data covers {lo:%d %b} to {hi:%d %b})")
        st.info("Partial months are excluded from MoM growth, YoY comparisons and the peak: "
                + "; ".join(parts) + ". They appear hatched in the chart below.")

    st.write("")
    with st.container(border=True):
        card_title("Monthly net sales vs last year",
                   "Bars: net sales. Line: the same month one year earlier. Hatched bars are partial months.")
        show_yoy = st.checkbox("Show YoY % on the right axis", value=True, key="show_yoy")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=vm.index, y=vm["sales"], name="Net sales",
            marker=dict(color=[PRIMARY if c else PRIMARY_LIGHT for c in vm["complete"]],
                        pattern=dict(shape=["" if c else "/" for c in vm["complete"]], fgcolor=PRIMARY)),
            hovertemplate=f"{CURRENCY} " + "%{y:,.0f}<extra>Net sales</extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=vm.index, y=vm["prior"], name="Same month last year", mode="lines+markers",
            line=dict(color=ACCENT, width=2.5), marker=dict(size=7), connectgaps=False,
            hovertemplate=f"{CURRENCY} " + "%{y:,.0f}<extra>Last year</extra>"), secondary_y=False)
        if show_yoy:
            fig.add_trace(go.Scatter(
                x=vm.index, y=vm["yoy"], name="YoY %", mode="lines+markers",
                line=dict(color=INK, width=1.5, dash="dot"), marker=dict(size=5), connectgaps=False,
                hovertemplate="%{y:+.1f}%<extra>YoY</extra>"), secondary_y=True)
            fig.update_yaxes(title_text="YoY %", ticksuffix="%", showgrid=False, secondary_y=True)
        fig.update_yaxes(title_text=f"Net sales ({CURRENCY})", tickformat=",.2s", secondary_y=False)
        fig.update_xaxes(tickformat="%b %Y", dtick="M1" if len(vm) <= 18 else "M3")
        show_fig(style_fig(fig, 420), key="sales_main")
        if vm["prior"].notna().sum() == 0:
            st.caption("No year-over-year comparison is available yet: the data covers less than 13 complete months.")

    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        card_title("Weekday pattern by month", "Darker cells are stronger sales days.")
        h1, h2 = st.columns(2)
        metric = h1.radio("Metric", ["Average per day", "Total"], horizontal=True, key="hm_metric")
        span = h2.radio("Months shown", ["Last 12", "Last 24", "All"], horizontal=True, key="hm_span")
        days = pd.date_range(ctx.rng["date"].min().normalize(), ctx.rng["date"].max().normalize())
        daily = ctx.rng.groupby(ctx.rng["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0.0)
        d = pd.DataFrame({"sales": daily.values, "month": days.to_period("M"), "wd": days.dayofweek})
        agg = d.groupby(["wd", "month"])["sales"].mean() if metric == "Average per day" else d.groupby(["wd", "month"])["sales"].sum()
        pv = agg.unstack("month").reindex(range(7))
        if span != "All":
            pv = pv.iloc[:, -(12 if span == "Last 12" else 24):]
        hm = go.Figure(go.Heatmap(
            z=pv.values, x=[p.strftime("%b %y") for p in pv.columns], y=WEEKDAYS,
            colorscale=[[0, "#f4f2fc"], [0.5, "#a98cff"], [1, PRIMARY_DARK]], xgap=2, ygap=2,
            colorbar=dict(title=CURRENCY, tickformat=",.2s"),
            hovertemplate="%{y}, %{x}<br>" + f"{CURRENCY} " + "%{z:,.0f}<extra></extra>"))
        hm.update_yaxes(autorange="reversed")
        show_fig(style_fig(hm, 360, legend=False), key="sales_heatmap")

    with right, st.container(border=True):
        card_title("Channels and payment", "Share of net sales in the selected period.")
        view = st.radio("Breakdown", ["Channel", "Payment method", "Channel x payment"], horizontal=True,
                        key="chan_view", label_visibility="collapsed")
        r = ctx.rng
        if view == "Channel":
            g = r.groupby("channel", as_index=False)["amount"].sum().sort_values("amount")
            g["share"] = g["amount"] / g["amount"].sum() * 100
            fig2 = px.bar(g, x="amount", y="channel", orientation="h", text=g["share"].map(lambda v: f"{v:.1f}%"),
                          color_discrete_sequence=[PRIMARY])
        elif view == "Payment method":
            g = r.groupby("payment", as_index=False)["amount"].sum().sort_values("amount")
            g["share"] = g["amount"] / g["amount"].sum() * 100
            fig2 = px.bar(g, x="amount", y="payment", orientation="h", text=g["share"].map(lambda v: f"{v:.1f}%"),
                          color_discrete_sequence=[PRIMARY])
        else:
            g = r.groupby(["channel", "payment"], as_index=False)["amount"].sum()
            order = g.groupby("channel")["amount"].sum().sort_values().index.tolist()
            fig2 = px.bar(g, x="amount", y="channel", color="payment", orientation="h",
                          category_orders={"channel": order}, color_discrete_sequence=PALETTE)
        fig2.update_traces(hovertemplate=f"{CURRENCY} " + "%{x:,.0f}<extra></extra>")
        fig2.update_xaxes(title=None, tickformat=",.2s")
        fig2.update_yaxes(title=None)
        show_fig(style_fig(fig2, 360, legend=(view == "Channel x payment")), key="sales_channels")

    with st.expander("Monthly data table"):
        t = vm.reset_index(names="month").copy()
        t["month"] = t["month"].dt.strftime("%b %Y")
        t["status"] = np.where(t["complete"], "Complete", "Partial")
        t = t[["month", "status", "sales", "prior", "yoy", "mom"]]
        t.columns = ["Month", "Status", f"Net sales ({CURRENCY})", "Same month last year", "YoY %", "MoM %"]
        show_df(t.round(1))


def page_overview(ctx: SimpleNamespace) -> None:
    page_header("Executive Overview", "The business at a glance for the selected period.")
    if not require_sales(ctx):
        return
    r = ctx.rng
    orders = len(r)
    aov = r["amount"].sum() / orders if orders else np.nan
    cust = r["customer_id"].nunique() if r["customer_id"].notna().any() else None
    ret = get_returns(ctx)
    ret_rate = (len(ret) / orders * 100) if orders and not ret.empty else None
    c = st.columns(5)
    kpi_card(c[0], "Net sales", fmt_money(r["amount"].sum()), f"{ctx.start:%d %b %Y} to {ctx.end:%d %b %Y}", primary=True)
    kpi_card(c[1], "Orders", f"{orders:,}", "Completed orders")
    kpi_card(c[2], "Average order value", fmt_money(aov), "Net sales per order")
    kpi_card(c[3], "Active customers", f"{cust:,}" if cust is not None else "n/a", "Unique buyers")
    kpi_card(c[4], "Return rate", fmt_pct(ret_rate, signed=False) if ret_rate is not None else "n/a", "Returns per order")
    st.write("")
    a, b = st.columns([3, 2])
    with a, st.container(border=True):
        card_title("Net sales by month", "Hatched bars are partial months.")
        vm = ctx.view_m
        fig = go.Figure(go.Bar(x=vm.index, y=vm["sales"],
                               marker=dict(color=[PRIMARY if x else PRIMARY_LIGHT for x in vm["complete"]],
                                           pattern=dict(shape=["" if x else "/" for x in vm["complete"]], fgcolor=PRIMARY)),
                               hovertemplate=f"{CURRENCY} " + "%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(tickformat=",.2s")
        fig.update_xaxes(tickformat="%b %Y")
        show_fig(style_fig(fig, 340, legend=False), key="ov_trend")
    with b, st.container(border=True):
        card_title("Top branches", "By net sales.")
        g = r.groupby("branch", as_index=False)["amount"].sum().nlargest(8, "amount").sort_values("amount")
        fig = px.bar(g, x="amount", y="branch", orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title=None, tickformat=",.2s")
        fig.update_yaxes(title=None)
        show_fig(style_fig(fig, 340, legend=False), key="ov_branches")
    d, e = st.columns(2)
    with d, st.container(border=True):
        card_title("Sales by channel")
        g = r.groupby("channel", as_index=False)["amount"].sum()
        fig = px.pie(g, names="channel", values="amount", hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent+label")
        show_fig(style_fig(fig, 320, legend=False), key="ov_channel")
    with e, st.container(border=True):
        card_title("Top products", "By revenue.")
        ps = product_summary(ctx)
        if ps.empty:
            st.info("Product revenue needs sales_order_items with an order id and an amount or quantity x price.")
        else:
            g = ps.nlargest(8, "revenue").sort_values("revenue")
            fig = px.bar(g, x="revenue", y="product", orientation="h", color_discrete_sequence=[PRIMARY])
            fig.update_xaxes(title=None, tickformat=",.2s")
            fig.update_yaxes(title=None)
            show_fig(style_fig(fig, 320, legend=False), key="ov_products")


def page_branches(ctx: SimpleNamespace) -> None:
    page_header("Branches & Geography", "How each branch and region contributes to sales.")
    if not require_sales(ctx):
        return
    r = ctx.rng
    g = r.groupby("branch").agg(sales=("amount", "sum"), orders=("amount", "size")).reset_index()
    g["aov"] = g["sales"] / g["orders"]
    g["share"] = g["sales"] / g["sales"].sum() * 100
    a, b = st.columns([2, 3])
    with a, st.container(border=True):
        card_title("Net sales by branch")
        fig = px.bar(g.sort_values("sales"), x="sales", y="branch", orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title=None, tickformat=",.2s")
        fig.update_yaxes(title=None)
        show_fig(style_fig(fig, max(320, 28 * len(g)), legend=False), key="br_bar")
    with b, st.container(border=True):
        card_title("Monthly trend for the top 6 branches")
        top = g.nlargest(6, "sales")["branch"]
        t = r[r["branch"].isin(top)].copy()
        t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
        t = t.groupby(["month", "branch"], as_index=False)["amount"].sum()
        fig = px.line(t, x="month", y="amount", color="branch", markers=True, color_discrete_sequence=PALETTE)
        fig.update_yaxes(title=None, tickformat=",.2s")
        fig.update_xaxes(title=None, tickformat="%b %Y")
        show_fig(style_fig(fig, 360), key="br_lines")

    # Geography
    br = get_table(ctx.tables, ["branches", "branch", "stores"], ("branch",), ("transfer",))
    with st.container(border=True):
        card_title("Geography")
        if br is None or br.empty:
            st.info("No branches table found, so regional views are unavailable.")
        else:
            bid = find_col(br, ["branch_id", "id", "store_id", "branch_code"])
            bname = find_col(br, ["branch_name", "name", "store_name", "branch"])
            city = find_col(br, ["city", "governorate", "region", "area", "state", "district"],
                            ("city", "govern", "region", "area"))
            lat = find_col(br, ["latitude", "lat"], ("lat",))
            lon = find_col(br, ["longitude", "lng", "lon", "long"], ("lon", "lng"))
            b2 = br.copy()
            b2["_name"] = b2[bname].astype(str) if bname else (b2[bid].astype(str) if bid else "")
            m = g.merge(b2, left_on="branch", right_on="_name", how="left")
            if m[city].isna().all() if city else True:
                if city is None:
                    st.info("The branches table has no city/region column.")
            else:
                gc = m.groupby(city, as_index=False)["sales"].sum().sort_values("sales", ascending=False)
                fig = px.bar(gc.head(15), x=city, y="sales", color_discrete_sequence=[PRIMARY])
                fig.update_yaxes(title=None, tickformat=",.2s")
                fig.update_xaxes(title=None)
                show_fig(style_fig(fig, 320, legend=False), key="br_city")
            if lat and lon:
                mm = m.dropna(subset=[lat, lon]).copy()
                mm[lat], mm[lon] = to_num(mm[lat]), to_num(mm[lon])
                mm = mm.dropna(subset=[lat, lon])
                if not mm.empty:
                    kw = dict(lat=lat, lon=lon, size="sales", hover_name="branch", zoom=5,
                              color_discrete_sequence=[PRIMARY], size_max=40)
                    if hasattr(px, "scatter_map"):
                        fmap = px.scatter_map(mm, map_style="carto-positron", **kw)
                    else:
                        fmap = px.scatter_mapbox(mm, mapbox_style="carto-positron", **kw)
                    show_fig(style_fig(fmap, 420, legend=False), key="br_map")
    with st.expander("Branch table"):
        t = g.sort_values("sales", ascending=False).rename(columns={
            "branch": "Branch", "sales": f"Net sales ({CURRENCY})", "orders": "Orders",
            "aov": "Avg order value", "share": "Share %"})
        show_df(t.round(1))


def page_products(ctx: SimpleNamespace) -> None:
    page_header("Products & Profitability", "Best sellers, categories and margins.")
    if not require_sales(ctx):
        return
    ps = product_summary(ctx)
    if ps.empty:
        st.warning("Product analysis needs sales_order_items with an order id plus an amount or quantity x price.")
        return
    has_cost = ps["cost_total"].notna().any()
    c = st.columns(4)
    kpi_card(c[0], "Product revenue", fmt_money(ps["revenue"].sum()), "From order lines", primary=True)
    kpi_card(c[1], "Units sold", f"{ps['units'].sum():,.0f}", f"{len(ps):,} products")
    if has_cost:
        profit = ps["profit"].sum()
        kpi_card(c[2], "Gross profit", fmt_money(profit), "Revenue minus unit cost x units")
        kpi_card(c[3], "Gross margin", fmt_pct(profit / ps["revenue"].sum() * 100, signed=False), "On products with a cost")
    else:
        kpi_card(c[2], "Gross profit", "n/a", "No cost column in the catalog")
        kpi_card(c[3], "Gross margin", "n/a", "Add unit_cost to products_catalog")
    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        card_title("Top products")
        opts = ["revenue", "units"] + (["profit"] if has_cost else [])
        by = st.radio("Rank by", opts, horizontal=True, key="prod_by")
        g = ps.nlargest(15, by).sort_values(by)
        fig = px.bar(g, x=by, y="product", orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title=None, tickformat=",.2s")
        fig.update_yaxes(title=None)
        show_fig(style_fig(fig, 460, legend=False), key="prod_top")
    with b, st.container(border=True):
        card_title("Revenue by category")
        g = ps.groupby("category", as_index=False)["revenue"].sum()
        fig = px.treemap(g, path=["category"], values="revenue", color="revenue",
                         color_continuous_scale=["#e6defd", PRIMARY_DARK])
        fig.update_coloraxes(showscale=False)
        show_fig(style_fig(fig, 490, legend=False), key="prod_cat")
    if has_cost:
        with st.container(border=True):
            card_title("Revenue vs margin", "Each bubble is a product; size is units sold.")
            s = ps.dropna(subset=["margin"])
            fig = px.scatter(s, x="revenue", y="margin", size=s["units"].clip(lower=1), hover_name="product",
                             color="category", color_discrete_sequence=PALETTE)
            fig.update_xaxes(title=f"Revenue ({CURRENCY})", tickformat=",.2s")
            fig.update_yaxes(title="Gross margin %")
            show_fig(style_fig(fig, 420), key="prod_scatter")
    with st.expander("Product table"):
        show_df(ps.drop(columns=["product_id"]).sort_values("revenue", ascending=False).round(1))


def page_customers(ctx: SimpleNamespace) -> None:
    page_header("Customer Intelligence", "Who buys, how often, and how many are new.")
    if not require_sales(ctx):
        return
    if not ctx.fact["customer_id"].notna().any():
        st.warning("The sales orders table has no customer id column, so customer analytics are unavailable.")
        return
    r = ctx.rng[ctx.rng["customer_id"].notna()]
    first = ctx.fact.dropna(subset=["customer_id"]).groupby("customer_id")["date"].min()
    spend = r.groupby("customer_id").agg(spend=("amount", "sum"), orders=("amount", "size")).reset_index()
    repeat = (spend["orders"] > 1).mean() * 100 if len(spend) else np.nan
    new_c = int((first[spend["customer_id"]] >= ctx.start).sum()) if len(spend) else 0
    c = st.columns(4)
    kpi_card(c[0], "Active customers", f"{len(spend):,}", "Bought in the period", primary=True)
    kpi_card(c[1], "New customers", f"{new_c:,}", "First order in the period")
    kpi_card(c[2], "Repeat rate", fmt_pct(repeat, signed=False), "Customers with 2+ orders")
    kpi_card(c[3], "Spend per customer", fmt_money(spend["spend"].mean() if len(spend) else np.nan), "Average")
    st.write("")
    a, b = st.columns([3, 2])
    with a, st.container(border=True):
        card_title("New vs returning customers by month")
        t = r.copy()
        t["month"] = t["date"].dt.to_period("M").dt.to_timestamp()
        act = t.groupby("month")["customer_id"].nunique()
        f_month = first.dt.to_period("M").dt.to_timestamp()
        new = f_month.value_counts().reindex(act.index, fill_value=0)
        d = pd.DataFrame({"New": new.clip(upper=act), "Returning": (act - new.clip(upper=act))}).reset_index()
        d = d.melt("month", var_name="type", value_name="customers")
        fig = px.bar(d, x="month", y="customers", color="type",
                     color_discrete_map={"New": ACCENT, "Returning": PRIMARY})
        fig.update_xaxes(title=None, tickformat="%b %Y")
        fig.update_yaxes(title=None)
        show_fig(style_fig(fig, 360), key="cu_new")
    with b, st.container(border=True):
        card_title("Orders per customer")
        fig = px.histogram(spend, x="orders", nbins=min(20, max(int(spend["orders"].max()), 2)),
                           color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(title="Orders in period")
        fig.update_yaxes(title="Customers")
        show_fig(style_fig(fig, 360, legend=False), key="cu_hist")
    cust = get_table(ctx.tables, ["customers", "customer"], ("customer",), ("order", "segment"))
    top = spend.nlargest(10, "spend").copy()
    if cust is not None and not cust.empty:
        cid = find_col(cust, ["customer_id", "id", "customer_code"])
        nm = find_col(cust, ["customer_name", "name", "full_name"], ("name",))
        if cid and nm:
            top["name"] = top["customer_id"].map(dict(zip(cust[cid].astype(str), cust[nm].astype(str))))
        seg = find_col(cust, ["segment", "customer_segment", "tier", "customer_type", "type"], ("segment", "tier"))
        if cid and seg:
            smap = dict(zip(cust[cid].astype(str), cust[seg].astype(str)))
            sg = spend.assign(segment=spend["customer_id"].map(smap).fillna("Unknown")).groupby("segment", as_index=False)["spend"].sum()
            with st.container(border=True):
                card_title("Spend by customer segment")
                fig = px.pie(sg, names="segment", values="spend", hole=0.55, color_discrete_sequence=PALETTE)
                show_fig(style_fig(fig, 320, legend=True), key="cu_seg")
    with st.container(border=True):
        card_title("Top 10 customers by spend")
        cols = [x for x in ["customer_id", "name", "orders", "spend"] if x in top.columns]
        show_df(top[cols].round(0))


def page_returns(ctx: SimpleNamespace) -> None:
    page_header("Returns & Delivery", "Return volumes, reasons and delivery speed.")
    ret = get_returns(ctx)
    orders = len(ctx.rng)
    if ret.empty:
        st.warning("No usable sales_returns table found (needs a date column). Showing delivery metrics only.")
    else:
        c = st.columns(4)
        kpi_card(c[0], "Returns", f"{len(ret):,}", "Return records", primary=True)
        kpi_card(c[1], "Returned units", f"{ret['qty'].sum():,.0f}", "Quantity")
        kpi_card(c[2], "Refund value", fmt_money(ret["amount"].sum()) if ret["amount"].abs().sum() else "n/a", "Total refunded")
        kpi_card(c[3], "Return rate", fmt_pct(len(ret) / orders * 100, signed=False) if orders else "n/a", "Returns per order")
        st.write("")
        a, b = st.columns([3, 2])
        with a, st.container(border=True):
            card_title("Returns by month")
            t = ret.assign(month=ret["date"].dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False).agg(
                returns=("qty", "size"), value=("amount", "sum"))
            fig = px.bar(t, x="month", y="returns", color_discrete_sequence=[PRIMARY])
            fig.update_xaxes(title=None, tickformat="%b %Y")
            fig.update_yaxes(title=None)
            show_fig(style_fig(fig, 340, legend=False), key="rt_month")
        with b, st.container(border=True):
            card_title("Return reasons")
            g = ret.groupby("reason", as_index=False).size().nlargest(10, "size").sort_values("size")
            fig = px.bar(g, x="size", y="reason", orientation="h", color_discrete_sequence=[ACCENT])
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show_fig(style_fig(fig, 340, legend=False), key="rt_reason")

    # Delivery metrics from the raw orders table
    raw = get_table(ctx.tables, ["sales_orders", "sales_order", "orders", "sales"], ("order",),
                    ("item", "line", "return", "purchase", "transfer"))
    with st.container(border=True):
        card_title("Delivery")
        if raw is None:
            st.info("No sales orders table available.")
            return
        od = find_col(raw, DATE_C, ("date",))
        dd = find_col(raw, ["delivery_date", "delivered_at", "delivered_date", "shipped_date", "delivery_time"],
                      ("deliver",), ("status", "fee", "cost", "address"))
        ds = find_col(raw, ["delivery_status", "shipping_status", "fulfillment_status"], ("deliver", "shipping"))
        if od and dd:
            t = pd.DataFrame({"o": to_dt(raw[od]), "d": to_dt(raw[dd])}).dropna()
            t = t[(t["o"] >= ctx.start) & (t["o"] < ctx.end + pd.Timedelta(days=1))]
            t["days"] = (t["d"] - t["o"]).dt.total_seconds() / 86400
            t = t[(t["days"] >= 0) & (t["days"] <= 60)]
            if not t.empty:
                k1, k2 = st.columns(2)
                kpi_card(k1, "Average delivery time", f"{t['days'].mean():.1f} days", f"{len(t):,} delivered orders")
                kpi_card(k2, "90th percentile", f"{t['days'].quantile(.9):.1f} days", "Slowest 10% start here")
                fig = px.histogram(t, x="days", nbins=30, color_discrete_sequence=[PRIMARY])
                fig.update_xaxes(title="Days from order to delivery")
                fig.update_yaxes(title="Orders")
                show_fig(style_fig(fig, 300, legend=False), key="dl_hist")
        if ds:
            g = raw[ds].astype(str).str.title().value_counts().reset_index()
            g.columns = ["status", "orders"]
            fig = px.pie(g, names="status", values="orders", hole=0.55, color_discrete_sequence=PALETTE)
            show_fig(style_fig(fig, 300), key="dl_status")
        if not (od and dd) and not ds:
            st.info("No delivery date or delivery status column found in the orders table.")


def page_inventory(ctx: SimpleNamespace) -> None:
    page_header("Inventory & Transfers", "Stock levels over time and stock moving between branches.")
    stock = get_table(ctx.tables, ["monthly_stock", "stock_monthly", "inventory_monthly", "stock"], ("stock",),
                      ("transfer", "movement"))
    if stock is None or stock.empty:
        st.warning("No monthly_stock table found in 02_TABLES.")
    else:
        dcol = find_col(stock, ["month", "stock_month", "period", "snapshot_date", "year_month"] + DATE_C,
                        ("month", "date", "period"))
        qcol = find_col(stock, ["closing_stock", "closing_qty", "ending_stock", "on_hand", "stock_qty", "quantity", "qty", "stock"],
                        ("stock", "qty", "quantity"), ("_id",))
        vcol = find_col(stock, ["stock_value", "inventory_value", "closing_value", "value"], ("value",))
        bcol = find_col(stock, BRANCH_C)
        pcol = find_col(stock, PRODUCT_C)
        if not (dcol and (qcol or vcol)):
            st.warning("monthly_stock needs a month/date column and a quantity or value column.")
        else:
            s = stock.copy()
            s["_m"] = to_dt(s[dcol]).dt.to_period("M").dt.to_timestamp()
            s = s.dropna(subset=["_m"])
            measure = st.radio("Measure", [x for x in [("Units" if qcol else None), ("Value" if vcol else None)] if x],
                               horizontal=True, key="inv_measure")
            mcol = qcol if measure == "Units" else vcol
            s["_v"] = to_num(s[mcol]).fillna(0)
            last = s["_m"].max()
            c = st.columns(3)
            kpi_card(c[0], f"Latest {measure.lower()} on hand", f"{s.loc[s['_m'] == last, '_v'].sum():,.0f}", month_label(last), primary=True)
            kpi_card(c[1], "Snapshots", f"{s['_m'].nunique():,}", "Months of history")
            kpi_card(c[2], "Records", f"{len(s):,}", "Stock rows")
            st.write("")
            a, b = st.columns([3, 2])
            with a, st.container(border=True):
                card_title("Stock over time")
                t = s.groupby("_m", as_index=False)["_v"].sum()
                fig = px.area(t, x="_m", y="_v", color_discrete_sequence=[PRIMARY])
                fig.update_xaxes(title=None, tickformat="%b %Y")
                fig.update_yaxes(title=None, tickformat=",.2s")
                show_fig(style_fig(fig, 340, legend=False), key="inv_trend")
            with b, st.container(border=True):
                dim = bcol or pcol
                card_title(f"Latest stock by {'branch' if bcol else 'product'}")
                if dim:
                    cur = s[s["_m"] == last].copy()
                    cur["_d"] = cur[dim].astype(str)
                    if bcol:
                        cur["_d"] = cur["_d"].map(ctx.branch_map).fillna(cur["_d"])
                    g = cur.groupby("_d", as_index=False)["_v"].sum().nlargest(12, "_v").sort_values("_v")
                    fig = px.bar(g, x="_v", y="_d", orientation="h", color_discrete_sequence=[PRIMARY])
                    fig.update_xaxes(title=None, tickformat=",.2s")
                    fig.update_yaxes(title=None)
                    show_fig(style_fig(fig, 340, legend=False), key="inv_dim")
                else:
                    st.info("No branch or product column in monthly_stock.")

    tr = get_table(ctx.tables, ["stock_transfers", "inventory_transfers", "branch_transfers", "transfers"], ("transfer",))
    with st.container(border=True):
        card_title("Transfers between branches")
        if tr is None or tr.empty:
            st.info("No transfers table found.")
            return
        dcol = find_col(tr, ["transfer_date", "date", "created_at", "shipped_date"] + DATE_C, ("date",))
        qcol = find_col(tr, QTY_C, ("qty", "quantity"))
        fcol = find_col(tr, ["from_branch_id", "source_branch_id", "from_branch", "source_branch", "from_location"], ("from", "source"))
        tcol = find_col(tr, ["to_branch_id", "destination_branch_id", "to_branch", "destination_branch", "to_location"], ("to_branch", "dest"))
        t = tr.copy()
        t["_q"] = to_num(t[qcol]).fillna(0) if qcol else 1.0
        k = st.columns(2)
        kpi_card(k[0], "Transfer records", f"{len(t):,}", "Rows in transfers")
        kpi_card(k[1], "Units moved", f"{t['_q'].sum():,.0f}", "Total quantity" if qcol else "Rows counted (no quantity column)")
        if dcol:
            t["_m"] = to_dt(t[dcol]).dt.to_period("M").dt.to_timestamp()
            g = t.dropna(subset=["_m"]).groupby("_m", as_index=False)["_q"].sum()
            fig = px.bar(g, x="_m", y="_q", color_discrete_sequence=[PRIMARY])
            fig.update_xaxes(title=None, tickformat="%b %Y")
            fig.update_yaxes(title=None)
            show_fig(style_fig(fig, 300, legend=False), key="tr_trend")
        if fcol and tcol:
            t["_r"] = (t[fcol].astype(str).map(ctx.branch_map).fillna(t[fcol].astype(str)) + " to "
                       + t[tcol].astype(str).map(ctx.branch_map).fillna(t[tcol].astype(str)))
            g = t.groupby("_r", as_index=False)["_q"].sum().nlargest(10, "_q").sort_values("_q")
            fig = px.bar(g, x="_q", y="_r", orientation="h", color_discrete_sequence=[ACCENT])
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            card_title("Busiest routes")
            show_fig(style_fig(fig, 340, legend=False), key="tr_routes")


def page_domain(ctx: SimpleNamespace, title: str, subtitle: str, keywords: tuple, roi: bool = False) -> None:
    """Data-driven page for domains without a fixed schema: explores every table whose name matches."""
    page_header(title, subtitle)
    found = [(n, df) for n, df in ctx.tables.items() if any(k in n for k in keywords) and not df.empty]
    if not found:
        st.warning(f"No tables matching {', '.join(keywords)} were found in {DATA_DIR_NAME}. "
                   "Add a CSV whose file name contains one of these words and it will appear here.")
        st.caption("Available tables: " + (", ".join(sorted(ctx.tables)) or "none"))
        return
    c = st.columns(3)
    kpi_card(c[0], "Matching tables", f"{len(found)}", ", ".join(n for n, _ in found)[:60], primary=True)
    kpi_card(c[1], "Total rows", f"{sum(len(d) for _, d in found):,}", "Across matching tables")
    kpi_card(c[2], "Columns", f"{sum(d.shape[1] for _, d in found):,}", "Across matching tables")
    st.write("")
    for name, df in found:
        with st.container(border=True):
            card_title(name, f"{len(df):,} rows, {df.shape[1]} columns")
            nums = [c for c in df.select_dtypes("number").columns if c != "id" and not c.endswith("_id")]
            dcol = find_col(df, DATE_C + ["month", "period", "invoice_date", "payment_date", "start_date"], ("date", "month", "period"))
            if roi:
                sp = find_col(df, ["spend", "cost", "budget", "amount_spent"], ("spend", "cost", "budget"))
                rv = find_col(df, ["revenue", "sales", "attributed_revenue", "conversion_value"], ("revenue", "sales"))
                if sp and rv:
                    tsp, trv = to_num(df[sp]).sum(), to_num(df[rv]).sum()
                    k = st.columns(3)
                    kpi_card(k[0], "Spend", fmt_money(tsp), sp)
                    kpi_card(k[1], "Revenue", fmt_money(trv), rv)
                    kpi_card(k[2], "ROI", fmt_pct((trv - tsp) / tsp * 100 if tsp else np.nan), "(Revenue - spend) / spend")
                    lab = find_col(df, ["campaign", "campaign_name", "name", "channel", "platform"], ("campaign", "channel"))
                    if lab:
                        g = df.assign(_s=to_num(df[sp]), _r=to_num(df[rv])).groupby(lab, as_index=False)[["_s", "_r"]].sum()
                        g["roi"] = np.where(g["_s"] > 0, (g["_r"] - g["_s"]) / g["_s"] * 100, np.nan)
                        g = g.dropna(subset=["roi"]).nlargest(15, "roi").sort_values("roi")
                        fig = px.bar(g, x="roi", y=lab, orientation="h", color_discrete_sequence=[PRIMARY])
                        fig.update_xaxes(title="ROI %")
                        fig.update_yaxes(title=None)
                        show_fig(style_fig(fig, 340, legend=False), key=f"roi_{name}")
                else:
                    st.caption("ROI needs both a spend/cost column and a revenue/sales column in the same table.")
            if nums:
                measure = st.selectbox("Measure", nums, key=f"{name}_measure")
                v1, v2 = st.columns(2)
                if dcol:
                    t = df.assign(_m=to_dt(df[dcol]).dt.to_period("M").dt.to_timestamp(), _v=to_num(df[measure]))
                    g = t.dropna(subset=["_m"]).groupby("_m", as_index=False)["_v"].sum()
                    if not g.empty:
                        fig = px.bar(g, x="_m", y="_v", color_discrete_sequence=[PRIMARY])
                        fig.update_xaxes(title=None, tickformat="%b %Y")
                        fig.update_yaxes(title=f"Sum of {measure}", tickformat=",.2s")
                        with v1:
                            show_fig(style_fig(fig, 320, legend=False), key=f"{name}_trend")
                cats = [c for c in df.select_dtypes(exclude="number").columns
                        if 2 <= df[c].nunique() <= 40 and c != dcol and not c.endswith("_id")]
                if cats:
                    cat = st.selectbox("Group by", cats, key=f"{name}_cat")
                    g = df.assign(_v=to_num(df[measure])).groupby(cat, as_index=False)["_v"].sum().nlargest(15, "_v").sort_values("_v")
                    fig = px.bar(g, x="_v", y=cat, orientation="h", color_discrete_sequence=[ACCENT])
                    fig.update_xaxes(title=f"Sum of {measure}", tickformat=",.2s")
                    fig.update_yaxes(title=None)
                    with v2:
                        show_fig(style_fig(fig, 320, legend=False), key=f"{name}_cat_chart")
            with st.expander("Preview rows"):
                show_df(df.head(200))


# --------------------------------------------------------------------------------------
# AI & RAG assistant (local retrieval engine over the loaded data)
# --------------------------------------------------------------------------------------
def answer_question(q: str, ctx: SimpleNamespace) -> str:
    ql = q.lower()
    has_sales = not ctx.fact.empty and not ctx.rng.empty
    r = ctx.rng
    vm = ctx.view_m
    comp = vm[vm["complete"]] if has_sales else pd.DataFrame()
    out: list[str] = []
    if has_sales and re.search(r"branch|store|location", ql):
        g = r.groupby("branch")["amount"].sum().nlargest(5)
        out.append("**Top branches by net sales:**\n" + "\n".join(f"- {b}: {fmt_money(v)}" for b, v in g.items()))
    if has_sales and not comp.empty and re.search(r"peak|best month|highest|record", ql):
        ts = comp["sales"].idxmax()
        out.append(f"**Peak complete month:** {month_label(ts)} with {fmt_money(comp.loc[ts, 'sales'])}.")
    if has_sales and not comp.empty and re.search(r"mom|growth|last month|month over month|trend|yoy|last year", ql):
        last = comp.iloc[-1]
        txt = f"**Last complete month ({month_label(comp.index[-1])}):** {fmt_money(last['sales'])}, MoM {fmt_pct(last['mom'])}"
        txt += f", YoY {fmt_pct(last['yoy'])}." if pd.notna(last["yoy"]) else "; no YoY comparison available."
        out.append(txt)
    if has_sales and re.search(r"channel|pos|wholesale|e-?commerce|online", ql):
        g = r.groupby("channel")["amount"].sum().sort_values(ascending=False)
        tot = g.sum()
        out.append("**Net sales by channel:**\n" + "\n".join(f"- {k}: {fmt_money(v)} ({v / tot * 100:.1f}%)" for k, v in g.items()))
    if has_sales and re.search(r"payment|cash|card|visa|method", ql):
        g = r.groupby("payment")["amount"].sum().sort_values(ascending=False)
        tot = g.sum()
        out.append("**Net sales by payment method:**\n" + "\n".join(f"- {k}: {fmt_money(v)} ({v / tot * 100:.1f}%)" for k, v in g.head(6).items()))
    if has_sales and re.search(r"product|sku|best sell|top sell|category", ql):
        ps = product_summary(ctx)
        if not ps.empty:
            out.append("**Top products by revenue:**\n" + "\n".join(
                f"- {x.product}: {fmt_money(x.revenue)} ({x.units:,.0f} units)" for x in ps.nlargest(5, "revenue").itertuples()))
    if re.search(r"return|refund", ql):
        ret = get_returns(ctx)
        if not ret.empty:
            out.append(f"**Returns:** {len(ret):,} records, {ret['qty'].sum():,.0f} units"
                       + (f", {fmt_money(ret['amount'].sum())} refunded." if ret["amount"].abs().sum() else "."))
    if has_sales and re.search(r"customer|buyer|client", ql) and r["customer_id"].notna().any():
        out.append(f"**Customers:** {r['customer_id'].nunique():,} active customers, "
                   f"average order value {fmt_money(r['amount'].mean())}.")
    if has_sales and re.search(r"total|sales|revenue|overall", ql) and not out:
        out.append(f"**Net sales for the selected period:** {fmt_money(r['amount'].sum())} across {len(r):,} orders.")

    # Retrieval: match question words to table and column names
    tokens = {t for t in re.findall(r"[a-z0-9]+", ql) if len(t) > 2}
    hits = []
    for name, df in ctx.tables.items():
        score = sum(t in name for t in tokens) * 3 + sum(any(t in c for t in tokens) for c in df.columns)
        if score:
            hits.append((score, name, df))
    hits.sort(key=lambda x: -x[0])
    if re.search(r"table|column|schema|data|file|available", ql) and not hits:
        hits = [(1, n, d) for n, d in ctx.tables.items()]
    if hits and (not out or re.search(r"table|column|schema|data|file|available|where", ql)):
        out.append("**Relevant tables:**\n" + "\n".join(
            f"- `{n}` ({len(d):,} rows): {', '.join(d.columns[:10])}{'...' if d.shape[1] > 10 else ''}" for _, n, d in hits[:5]))
    if not out:
        out.append("I could not match that to the loaded data. Try asking about top branches, the peak month, "
                   "growth, channels, payment methods, top products, returns, customers, or which tables are available.")
    return "\n\n".join(out)


def page_assistant(ctx: SimpleNamespace) -> None:
    page_header("AI & RAG Assistant", "Ask questions about your data. Answers are retrieved from the tables loaded in 02_TABLES.")
    st.caption("This assistant runs locally: it retrieves facts from the filtered data and the table catalog. "
               "It does not send your data to an external service.")
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! Ask me about branches, growth, channels, products, returns or customers.")]
    suggestions = ["Which branches sell the most?", "What was the peak month?", "How did last month grow?",
                   "Show sales by channel", "Which tables have customer data?"]
    cols = st.columns(len(suggestions))
    clicked = None
    for col, s in zip(cols, suggestions):
        if col.button(s, key=f"sg_{s}"):
            clicked = s
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)
    prompt = st.chat_input("Ask about your Cottonil data") or clicked
    if prompt:
        st.session_state.chat.append(("user", prompt))
        st.session_state.chat.append(("assistant", answer_question(prompt, ctx)))
        st.rerun()
    if len(st.session_state.chat) > 1 and st.button("Clear conversation"):
        st.session_state.pop("chat")
        st.rerun()
    with st.expander("Table catalog"):
        cat = pd.DataFrame([{"table": n, "rows": len(d), "columns": ", ".join(d.columns[:12])} for n, d in ctx.tables.items()])
        if cat.empty:
            st.info("No tables loaded.")
        else:
            show_df(cat)


# --------------------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Cottonil | Executive BI", page_icon="🟣", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()

    data_dir = resolve_data_dir()
    tables, errors, sig = {}, [], ""
    if data_dir is not None:
        sig = dir_signature(data_dir)
        tables, errors = load_tables(str(data_dir), sig)
    model = build_model(sig, tables) if tables else empty_model(
        f"Folder '{DATA_DIR_NAME}' was not found next to app.py." if data_dir is None else "No CSV files were found.")
    fact = model["fact"]

    # ---- Sidebar -------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">C</div><div>'
                    '<div class="brand-name">COTTONIL<sup>®</sup></div>'
                    '<div class="brand-tag">WEAR IT BETTER</div></div></div>', unsafe_allow_html=True)
        connected = bool(tables)
        st.markdown(f'<span class="api-pill {"" if connected else "off"}"><span class="api-dot"></span>'
                    f'{"API connected" if connected else "API disconnected"}</span>', unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigate", PAGES, index=PAGES.index(DEFAULT_PAGE), key="page", label_visibility="collapsed")
        st.divider()

        if fact.empty:
            start = end = pd.Timestamp.today().normalize()
            data_min = data_max = start
            sel_branches, sel_channels = [], []
            scope = fact
        else:
            st.markdown("**Filters**")
            data_min, data_max = fact["date"].min(), fact["date"].max()
            picked = st.date_input("Date range", value=(data_min.date(), data_max.date()),
                                   min_value=data_min.date(), max_value=data_max.date())
            if isinstance(picked, (tuple, list)):
                s_d = picked[0] if len(picked) else data_min.date()
                e_d = picked[1] if len(picked) > 1 else s_d
            else:
                s_d = e_d = picked
            start, end = pd.Timestamp(s_d), pd.Timestamp(e_d)
            b_opts = sorted(fact["branch"].unique())
            c_opts = sorted(fact["channel"].unique())
            sel_branches = st.multiselect("Branches", b_opts, placeholder="All branches")
            sel_channels = st.multiselect("Channels", c_opts, placeholder="All channels")
            scope = fact
            if sel_branches:
                scope = scope[scope["branch"].isin(sel_branches)]
            if sel_channels:
                scope = scope[scope["channel"].isin(sel_channels)]
        if data_dir is not None:
            st.caption(f"Data folder: {DATA_DIR_NAME} ({len(tables)} tables)")
        notes = list(model["notes"]) + [f"Skipped {e}" for e in errors]
        if notes:
            with st.expander("Data notes"):
                for n in notes:
                    st.caption(f"- {n}")
        if st.button("Reload data"):
            st.cache_data.clear()
            st.rerun()

    # ---- Context -------------------------------------------------------------------
    if fact.empty:
        rng = fact
        m = pd.DataFrame(columns=["sales", "complete", "prior", "yoy", "mom"])
        view_m = m
    else:
        rng = scope[(scope["date"] >= start) & (scope["date"] < end + pd.Timedelta(days=1))]
        m = month_table(scope, data_min, data_max)
        view_m = m.loc[start.to_period("M").to_timestamp():end.to_period("M").to_timestamp()]
    ctx = SimpleNamespace(tables=tables, fact=fact, items=model["items"], notes=model["notes"],
                          branch_map=model["branch_map"], real_order_ids=model["real_order_ids"],
                          scope=scope, rng=rng, m=m, view_m=view_m, start=start, end=end,
                          data_min=data_min, data_max=data_max)

    if not tables:
        st.error(f"No data could be loaded. Place your CSV files in a folder named '{DATA_DIR_NAME}' next to app.py.")

    routes = {
        "Executive Overview": page_overview,
        "Sales Performance": page_sales,
        "Branches & Geography": page_branches,
        "Products & Profitability": page_products,
        "Customer Intelligence": page_customers,
        "Returns & Delivery": page_returns,
        "Inventory & Transfers": page_inventory,
        "Procurement & Imports": lambda c: page_domain(
            c, "Procurement & Imports", "Purchasing, suppliers and inbound shipments.",
            ("purchase", "import", "supplier", "procure", "vendor", "shipment")),
        "Marketing ROI": lambda c: page_domain(
            c, "Marketing ROI", "Campaign spend and the revenue it brings back.",
            ("marketing", "campaign", "ads", "promo", "coupon"), roi=True),
        "Workforce & Costs": lambda c: page_domain(
            c, "Workforce & Costs", "Headcount, payroll and operating expenses.",
            ("employee", "payroll", "salar", "staff", "expense", "cost", "attendance", "hr_")),
        "AI & RAG Assistant": page_assistant,
    }
    try:
        routes[page](ctx)
    except Exception as exc:  # noqa: BLE001 - one broken page must not take the app down
        st.error(f"This page could not be rendered: {exc}")
        st.caption("Check the column names in the related CSV files, or open 'Data notes' in the sidebar.")


if __name__ == "__main__":
    main()