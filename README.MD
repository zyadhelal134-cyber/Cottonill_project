# Cottonil — Executive Business Intelligence Dashboard

An interactive Business Intelligence and analytics web application for **Cottonil**, built with **Python**, **Streamlit**, **Pandas**, **NumPy**, and **Plotly** to transform operational CSV tables into clear, decision-ready insights.

## Key Features

- **Multi-Page BI Navigation:** Switch across *Executive Overview*, *Sales Performance*, *Branches & Geography*, *Products & Profitability*, *Customer Intelligence*, *Returns & Delivery*, *Inventory & Transfers*, *Procurement & Imports*, *Marketing ROI*, *Workforce & Costs*, and a local *AI & RAG Assistant*.
- **Sales & Revenue Analytics:** Tracks net sales trends, month-over-month (MoM) and year-over-year (YoY) comparisons, weekday sales patterns, and channel/payment breakdowns.
- **Smart Schema Detection & Fallbacks:** Automatically normalizes column names to `snake_case`, matches common column candidates/keywords, and rebuilds order amounts from line items when needed.
- **Partial Month Handling:** Flags incomplete months and excludes them from MoM, YoY, and peak calculations so growth metrics remain accurate[cite: 3, 6].
- **Interactive Filtering:** Filter insights dynamically by date range, branch, and sales channel from the sidebar[cite: 3, 6].

## Project Structure

```text
project/
├── app.py              # Main Streamlit dashboard application
├── README.md           # Project documentation
└── 02_TABLES/          # Local directory containing CSV data tables
    ├── sales_orders.csv
    ├── sales_order_items.csv
    ├── branches.csv
    ├── customers.csv
    ├── products_catalog.csv
    ├── sales_returns.csv
    └── monthly_stock.csv