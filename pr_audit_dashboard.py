import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Testim PR Audit Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 0 6px 0;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }

    .metric-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17,153,142,0.3); }
    .metric-blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 4px 15px rgba(79,172,254,0.3); }
    .metric-orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 4px 15px rgba(245,87,108,0.3); }
    .metric-purple { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); box-shadow: 0 4px 15px rgba(161,140,209,0.3); }

    .user-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 14px 20px;
        color: white;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .user-header .name {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .user-header .badge {
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 6px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Action styling ─────────────────────────────────────────────────────────────
ACTION_CONFIG = {
    "pull-request.submitted": {"label": "Submitted", "color": "#4facfe", "icon": "📤", "short": "Submitted"},
    "pull-request.requested-changes": {"label": "Changes Requested", "color": "#f5576c", "icon": "🔄", "short": "Changes Req."},
    "pull-request.approved": {"label": "Approved", "color": "#38ef7d", "icon": "✅", "short": "Approved"},
    "pull-request.closed": {"label": "Closed / Merged", "color": "#a18cd1", "icon": "🔒", "short": "Closed/Merged"},
}

ACTION_COLOR_MAP = {k: v["color"] for k, v in ACTION_CONFIG.items()}
ACTION_ORDER = list(ACTION_CONFIG.keys())


def get_display_name(email: str) -> str:
    """Convert email to a readable display name."""
    local = email.split("@")[0]
    parts = local.replace("_", ".").replace("-", ".").split(".")
    return " ".join(p.capitalize() for p in parts if p)


def parse_extra_data(extra_str: str) -> dict:
    """Parse the Extra Data JSON field safely."""
    try:
        return json.loads(extra_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_pr_info(extra: dict) -> dict:
    """Extract useful PR metadata from Extra Data."""
    title = extra.get("title") or extra.get("prTitle", "")
    pr_id = extra.get("prId", "")
    source = extra.get("sourceBranch", "")
    target = extra.get("targetBranch", "")
    status = extra.get("status", "")
    tests_added = tests_changed = tests_deleted = 0
    shared_added = shared_changed = shared_deleted = 0
    if "revisionData" in extra:
        rd = extra["revisionData"]
        if "tests" in rd:
            tests_added = len(rd["tests"].get("added", []))
            tests_changed = len(rd["tests"].get("changed", []))
            tests_deleted = len(rd["tests"].get("deleted", []))
        if "sharedSteps" in rd:
            shared_added = len(rd["sharedSteps"].get("added", []))
            shared_changed = len(rd["sharedSteps"].get("changed", []))
            shared_deleted = len(rd["sharedSteps"].get("deleted", []))
    return {
        "pr_title": title,
        "pr_id": pr_id,
        "source_branch": source,
        "target_branch": target,
        "pr_status": status,
        "tests_added": tests_added,
        "tests_changed": tests_changed,
        "tests_deleted": tests_deleted,
        "shared_added": shared_added,
        "shared_changed": shared_changed,
        "shared_deleted": shared_deleted,
    }


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    """Load and process the CSV file."""
    df = pd.read_csv(uploaded_file)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df["Date"] = df["Time"].dt.date
    df["Hour"] = df["Time"].dt.hour
    df["Day Name"] = df["Time"].dt.strftime("%a")
    df["Display Name"] = df["User Email"].apply(get_display_name)

    extra_info = df["Extra Data"].apply(parse_extra_data).apply(extract_pr_info)
    extra_df = pd.DataFrame(extra_info.tolist())
    df = pd.concat([df, extra_df], axis=1)

    df["Action Label"] = df["Action"].map(lambda a: ACTION_CONFIG.get(a, {}).get("label", a))
    return df


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Testim PR Audit Dashboard")
st.markdown("Upload the Testim audit log CSV to visualize pull request activity across the team.")

uploaded_file = st.file_uploader("Drop your Testim audit CSV here", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("👆 Upload a Testim audit log CSV file to get started.")
    st.stop()

df = load_data(uploaded_file)

# ─── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    all_users = sorted(df["Display Name"].unique())
    selected_users = st.multiselect("Team Members", all_users, default=all_users)

    all_actions = [a for a in ACTION_ORDER if a in df["Action"].unique()]
    selected_actions = st.multiselect(
        "Action Types",
        all_actions,
        default=all_actions,
        format_func=lambda a: f'{ACTION_CONFIG[a]["icon"]} {ACTION_CONFIG[a]["label"]}',
    )

    date_range = st.date_input(
        "Date Range",
        value=(df["Date"].min(), df["Date"].max()),
        min_value=df["Date"].min(),
        max_value=df["Date"].max(),
    )

# Apply filters
mask = (
    df["Display Name"].isin(selected_users)
    & df["Action"].isin(selected_actions)
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ─── KPI Cards ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

total_events = len(filtered)
unique_users = filtered["Display Name"].nunique()
submitted = len(filtered[filtered["Action"] == "pull-request.submitted"])
merged = len(filtered[(filtered["Action"] == "pull-request.closed") & (filtered["pr_status"] == "merged")])
reviews = len(filtered[filtered["Action"].isin(["pull-request.requested-changes", "pull-request.approved"])])

with k1:
    st.markdown(f'<div class="metric-card"><h3>Total Events</h3><p class="value">{total_events}</p></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card metric-blue"><h3>Team Members</h3><p class="value">{unique_users}</p></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card metric-green"><h3>PRs Submitted</h3><p class="value">{submitted}</p></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card metric-orange"><h3>PRs Merged</h3><p class="value">{merged}</p></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="metric-card metric-purple"><h3>Reviews Done</h3><p class="value">{reviews}</p></div>', unsafe_allow_html=True)

st.markdown("")

# ─── Row 1: Activity Timeline + Action Breakdown ────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<p class="section-title">Daily Activity Timeline</p>', unsafe_allow_html=True)

    daily = (
        filtered.groupby(["Date", "Action"])
        .size()
        .reset_index(name="Count")
    )
    daily["Action Label"] = daily["Action"].map(lambda a: ACTION_CONFIG.get(a, {}).get("label", a))

    fig_timeline = px.bar(
        daily,
        x="Date",
        y="Count",
        color="Action Label",
        color_discrete_map={v["label"]: v["color"] for v in ACTION_CONFIG.values()},
        barmode="stack",
    )
    fig_timeline.update_layout(
        xaxis_title="",
        yaxis_title="Events",
        legend_title="",
        height=380,
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick="D1", tickformat="%a %d/%m"),
    )
    fig_timeline.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_right:
    st.markdown('<p class="section-title">Action Breakdown</p>', unsafe_allow_html=True)

    action_counts = filtered["Action"].value_counts().reset_index()
    action_counts.columns = ["Action", "Count"]
    action_counts["Label"] = action_counts["Action"].map(lambda a: ACTION_CONFIG.get(a, {}).get("label", a))
    action_counts["Color"] = action_counts["Action"].map(lambda a: ACTION_CONFIG.get(a, {}).get("color", "#ccc"))

    fig_donut = go.Figure(
        go.Pie(
            labels=action_counts["Label"],
            values=action_counts["Count"],
            hole=0.55,
            marker_colors=action_counts["Color"].tolist(),
            textinfo="label+value",
            textposition="outside",
            textfont_size=12,
        )
    )
    fig_donut.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=[
            dict(text=f"<b>{total_events}</b><br>events", x=0.5, y=0.5, font_size=18, showarrow=False)
        ],
    )
    st.plotly_chart(fig_donut, use_container_width=True)


# ─── Row 2: Per-person bar chart ────────────────────────────────────────────────
st.markdown('<p class="section-title">Activity per Team Member</p>', unsafe_allow_html=True)

user_action = (
    filtered.groupby(["Display Name", "Action"])
    .size()
    .reset_index(name="Count")
)
user_action["Action Label"] = user_action["Action"].map(lambda a: ACTION_CONFIG.get(a, {}).get("label", a))

# Sort users by total activity descending
user_totals = user_action.groupby("Display Name")["Count"].sum().sort_values(ascending=True)
user_order = user_totals.index.tolist()

fig_per_user = px.bar(
    user_action,
    y="Display Name",
    x="Count",
    color="Action Label",
    color_discrete_map={v["label"]: v["color"] for v in ACTION_CONFIG.values()},
    orientation="h",
    barmode="stack",
    category_orders={"Display Name": user_order},
)
fig_per_user.update_layout(
    xaxis_title="Number of Events",
    yaxis_title="",
    legend_title="",
    height=max(320, len(user_order) * 42),
    margin=dict(l=20, r=20, t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)",
)
fig_per_user.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
st.plotly_chart(fig_per_user, use_container_width=True)


# ─── Row 3: Activity Heatmap (Day x Person) ─────────────────────────────────────
st.markdown('<p class="section-title">Activity Heatmap (Day x Person)</p>', unsafe_allow_html=True)

heatmap_data = (
    filtered.groupby(["Display Name", "Date"])
    .size()
    .reset_index(name="Count")
)
pivot = heatmap_data.pivot(index="Display Name", columns="Date", values="Count").fillna(0)
pivot = pivot.loc[user_order[::-1]]  # most active on top

date_labels = [d.strftime("%a %d/%m") for d in pivot.columns]

fig_heat = go.Figure(
    go.Heatmap(
        z=pivot.values,
        x=date_labels,
        y=pivot.index.tolist(),
        colorscale=[
            [0, "#f0f2f6"],
            [0.25, "#c3dafe"],
            [0.5, "#7f9cf5"],
            [0.75, "#5a67d8"],
            [1, "#3730a3"],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Events: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Events", thickness=15, len=0.6),
    )
)
fig_heat.update_layout(
    height=max(300, len(user_order) * 38),
    margin=dict(l=20, r=20, t=10, b=40),
    xaxis=dict(side="top"),
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_heat, use_container_width=True)


# ─── Row 4: Event Timeline (Gantt-style) ────────────────────────────────────────
st.markdown('<p class="section-title">Event Timeline</p>', unsafe_allow_html=True)
st.caption("Each dot represents a PR action. Hover for details.")

timeline_df = filtered.copy()
timeline_df["Time_local"] = timeline_df["Time"].dt.tz_convert(None)

fig_scatter = px.scatter(
    timeline_df,
    x="Time_local",
    y="Display Name",
    color="Action Label",
    color_discrete_map={v["label"]: v["color"] for v in ACTION_CONFIG.values()},
    hover_data={"pr_title": True, "pr_status": True, "Time_local": False},
    category_orders={"Display Name": user_order[::-1]},
)
fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1, color="white")), opacity=0.9)
fig_scatter.update_layout(
    xaxis_title="",
    yaxis_title="",
    legend_title="",
    height=max(350, len(user_order) * 45),
    margin=dict(l=20, r=20, t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickformat="%a %d/%m %H:%M"),
)
fig_scatter.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
fig_scatter.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
st.plotly_chart(fig_scatter, use_container_width=True)


# ─── Row 5: Individual Detail Cards ─────────────────────────────────────────────
st.markdown('<p class="section-title">Individual Breakdown</p>', unsafe_allow_html=True)

for user in sorted(filtered["Display Name"].unique()):
    user_df = filtered[filtered["Display Name"] == user].sort_values("Time", ascending=False)
    email = user_df["User Email"].iloc[0]
    total = len(user_df)

    sub = user_df["Action"].value_counts()
    badges = " ".join(
        f'<span class="badge">{ACTION_CONFIG.get(a, {}).get("icon", "")} '
        f'{ACTION_CONFIG.get(a, {}).get("short", a)}: {c}</span>'
        for a, c in sub.items()
    )

    st.markdown(
        f'<div class="user-header">'
        f'<span class="name">{user}</span>'
        f'<span class="badge" style="opacity:0.7">{email}</span>'
        f'<span class="badge" style="background:rgba(79,172,254,0.3)">{total} events</span>'
        f"{badges}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"Details for {user}", expanded=False):
        detail_left, detail_right = st.columns([2, 1])

        with detail_left:
            # Daily breakdown mini chart
            user_daily = user_df.groupby(["Date", "Action"]).size().reset_index(name="Count")
            user_daily["Action Label"] = user_daily["Action"].map(
                lambda a: ACTION_CONFIG.get(a, {}).get("label", a)
            )
            fig_user = px.bar(
                user_daily,
                x="Date",
                y="Count",
                color="Action Label",
                color_discrete_map={v["label"]: v["color"] for v in ACTION_CONFIG.values()},
                barmode="stack",
            )
            fig_user.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=30),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
                xaxis=dict(dtick="D1", tickformat="%a %d/%m"),
                yaxis_title="",
                xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_user.update_yaxes(gridcolor="rgba(0,0,0,0.06)", dtick=1)
            st.plotly_chart(fig_user, use_container_width=True)

        with detail_right:
            # PR summary stats
            submitted_count = len(user_df[user_df["Action"] == "pull-request.submitted"])
            review_count = len(user_df[user_df["Action"].isin(["pull-request.requested-changes", "pull-request.approved"])])
            merge_count = len(user_df[(user_df["Action"] == "pull-request.closed") & (user_df["pr_status"] == "merged")])
            total_tests = user_df["tests_added"].sum() + user_df["tests_changed"].sum()

            st.metric("PRs Submitted", submitted_count)
            st.metric("Reviews Given", review_count)
            st.metric("PRs Merged", merge_count)
            st.metric("Tests Touched", int(total_tests))

        # PR activity log table
        table_df = user_df[["Time", "Action Label", "pr_title", "pr_status", "tests_added", "tests_changed"]].copy()
        table_df.columns = ["Timestamp", "Action", "PR Title", "Status", "Tests Added", "Tests Changed"]
        table_df["Timestamp"] = table_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(table_df, use_container_width=True, hide_index=True)


# ─── Row 6: PR Titles Summary ───────────────────────────────────────────────────
st.markdown('<p class="section-title">PR Summary Table</p>', unsafe_allow_html=True)

pr_summary = (
    filtered[filtered["pr_title"] != ""]
    .groupby("pr_title")
    .agg(
        Events=("Action", "count"),
        Submitter=("Display Name", "first"),
        Last_Action=("Action Label", "last"),
        Last_Status=("pr_status", "last"),
        Tests_Added=("tests_added", "max"),
        Tests_Changed=("tests_changed", "max"),
        Last_Update=("Time", "max"),
    )
    .sort_values("Last_Update", ascending=False)
    .reset_index()
)
pr_summary.columns = ["PR Title", "Events", "Submitter", "Last Action", "Status", "Tests+", "Tests~", "Last Update"]
pr_summary["Last Update"] = pr_summary["Last Update"].dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(pr_summary, use_container_width=True, hide_index=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Dashboard generated from {len(df)} audit events | Date range: {df['Date'].min()} to {df['Date'].max()}")
