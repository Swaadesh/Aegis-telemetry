"""
Project Aegis - Telemetry & Token Tracking Dashboard
Data Analytics Specialist workstream (Swaadesh U.S)

Reads from the three tables populated either by mock_data.py (sample
data) or ingest_logs.py (live, real data piped from Narendra's server)
and visualizes: request volume, token usage, latency, Spend Guard
lock/unlock activity, and out-of-scope -> web-search fallback activity.

Usage:
    streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st

from storage import (
    init_db,
    fetch_all_execution_logs,
    fetch_all_lock_events,
    fetch_all_websearch_events,
)

st.set_page_config(page_title="Aegis Telemetry Dashboard", layout="wide")
init_db()


@st.cache_data(ttl=15)
def load_data():
    logs = pd.DataFrame(fetch_all_execution_logs())
    locks = pd.DataFrame(fetch_all_lock_events())
    websearch = pd.DataFrame(fetch_all_websearch_events())

    for df in (logs, locks, websearch):
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not logs.empty:
        logs["date"] = logs["timestamp"].dt.date
    return logs, locks, websearch


logs, locks, websearch = load_data()

st.title("Project Aegis — Telemetry & Usage Dashboard")
st.caption(
    "Reads execution_logs / lock_events / websearch_events. Populate with "
    "`python mock_data.py` for sample data, or pipe Narendra's live server "
    "output through `ingest_logs.py` for real data — this page doesn't "
    "change either way."
)

if logs.empty:
    st.warning(
        "No data yet. Run `python mock_data.py` for sample data, or start "
        "`ingest_logs.py` reading from the real server's stdout, then refresh."
    )
    st.stop()

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
st.sidebar.header("Filters")
roles = sorted(logs["role"].dropna().unique().tolist())
selected_roles = st.sidebar.multiselect("Role", roles, default=roles)

date_min, date_max = logs["date"].min(), logs["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max))

f = logs[logs["role"].isin(selected_roles)]
if len(date_range) == 2:
    f = f[(f["date"] >= date_range[0]) & (f["date"] <= date_range[1])]

# ---------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_requests = len(f)
total_tokens = int(f["total_tokens"].fillna(0).sum())
avg_latency = int(f["latency_ms"].mean()) if total_requests else 0
locked_pct = (f["status"] == "locked").mean() * 100 if total_requests else 0
out_of_scope_pct = (f["status"] == "out_of_scope").mean() * 100 if total_requests else 0

col1.metric("Total Requests", f"{total_requests:,}")
col2.metric("Total Tokens", f"{total_tokens:,}")
col3.metric("Avg Latency", f"{avg_latency:,} ms")
col4.metric("Locked (Spend Guard)", f"{locked_pct:.1f}%")
col5.metric("Out of Scope", f"{out_of_scope_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------
# Token usage over time + by role
# ---------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Token Usage Over Time")
    daily = f.groupby("date")["total_tokens"].sum(min_count=1).reset_index()
    st.line_chart(daily.set_index("date"))

with right:
    st.subheader("Token Usage by Role")
    by_role = f.groupby("role")["total_tokens"].sum(min_count=1).sort_values(ascending=False)
    st.bar_chart(by_role)

# ---------------------------------------------------------------------
# Status breakdown + retrieval quality
# ---------------------------------------------------------------------
left2, right2 = st.columns(2)

with left2:
    st.subheader("Requests by Status")
    st.bar_chart(f["status"].value_counts())

with right2:
    st.subheader("Retrieval Quality (search_score)")
    scored = f[f["search_score"].notna()]
    if scored.empty:
        st.info("No scored retrievals in range.")
    else:
        st.bar_chart(scored.groupby("date")["search_score"].mean())

st.divider()

# ---------------------------------------------------------------------
# Spend Guard — lock/unlock activity
# ---------------------------------------------------------------------
st.subheader("Spend Guard — Lock & Unlock Activity")

locked_requests = f[f["status"] == "locked"]

if locked_requests.empty:
    st.info("No Spend Guard locks in the selected range.")
else:
    # resolve each locked request_id to its final lock_events outcome
    final_outcomes = {}
    if not locks.empty:
        for req_id, group in locks.groupby("request_id"):
            final_outcomes[req_id] = group.sort_values("timestamp").iloc[-1]["event"]

    resolved = sum(1 for v in final_outcomes.values() if v == "unlocked")
    still_blocked = sum(1 for v in final_outcomes.values() if v == "blocked")
    failed_attempts = int((locks["event"] == "attempt_failed").sum()) if not locks.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locks Triggered", len(locked_requests))
    c2.metric("Resolved (Unlocked)", resolved)
    c3.metric("Still Blocked", still_blocked)
    c4.metric("Failed Attempts", failed_attempts)

    st.markdown("**Locks by Role**")
    st.bar_chart(locked_requests["role"].value_counts())

    with st.expander("Raw lock events"):
        st.dataframe(locks.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------
# Out-of-scope -> web search fallback
# ---------------------------------------------------------------------
st.subheader("Out-of-Scope → Web Search Fallback")

oos = f[f["status"] == "out_of_scope"]
if oos.empty:
    st.info("No out-of-scope requests in the selected range.")
else:
    confirmed = int((websearch["event"] == "confirmed").sum()) if not websearch.empty else 0
    declined = int((websearch["event"] == "declined").sum()) if not websearch.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Out-of-Scope Requests", len(oos))
    c2.metric("Fell Back to Web Search", confirmed)
    c3.metric("Declined", declined)

# ---------------------------------------------------------------------
# Raw log explorer
# ---------------------------------------------------------------------
with st.expander("Raw execution logs"):
    st.dataframe(f.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
