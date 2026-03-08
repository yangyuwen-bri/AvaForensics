"""Streamlit MVP for AvaForensics."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from avaforensics import build_protocol_view, get_leaderboard, load_app_state, refresh_protocol_live


st.set_page_config(
    page_title="AvaForensics MVP",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Training the baseline model from local Avalanche data...")
def _load_state() -> Dict[str, object]:
    try:
        return load_app_state("avax_data")
    except TypeError:
        return load_app_state()


def _load_leaderboard(state: Dict[str, object], band_filter: str, top_n: int) -> pd.DataFrame:
    try:
        if band_filter == "All":
            leaderboard = get_leaderboard(state)
        elif band_filter == "Healthy":
            leaderboard = get_leaderboard(state, alive_only=True)
        elif band_filter == "High Risk":
            leaderboard = get_leaderboard(state, alive_only=False)
        else:
            leaderboard = get_leaderboard(state)
    except TypeError:
        leaderboard = get_leaderboard(
            state,
            risk_band=None if band_filter == "All" else band_filter,
            top_n=top_n,
        )
        return leaderboard

    if isinstance(leaderboard, pd.DataFrame):
        if band_filter == "Watchlist" and "Risk Band" in leaderboard.columns:
            leaderboard = leaderboard[leaderboard["Risk Band"] == "Watchlist"]
        if top_n:
            leaderboard = leaderboard.head(top_n)
    return leaderboard


def _protocol_frame(state: Dict[str, object]) -> pd.DataFrame:
    if "scored_protocols" in state:
        return state["scored_protocols"].copy()
    if "leaderboard" in state and isinstance(state["leaderboard"], pd.DataFrame):
        frame = state["leaderboard"].copy()
        rename_map = {
            "Protocol": "name",
            "Current Label": "label",
            "Risk Band": "risk_band",
            "Health Score": "health_score",
        }
        return frame.rename(columns=rename_map)
    raise KeyError("State does not expose a protocol dataframe.")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.5rem 1.75rem;
                border-radius: 20px;
                background:
                    radial-gradient(circle at top left, rgba(96, 165, 250, 0.24), transparent 34%),
                    linear-gradient(135deg, #081425 0%, #0d2239 50%, #112d4f 100%);
                border: 1px solid rgba(125, 211, 252, 0.18);
                color: #f8fafc;
                margin-bottom: 1.25rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.4rem;
                line-height: 1.1;
            }
            .hero p {
                margin: 0.6rem 0 0 0;
                color: rgba(248, 250, 252, 0.82);
                font-size: 1rem;
            }
            .score-card {
                padding: 1.1rem 1.2rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.78));
                border: 1px solid rgba(148, 163, 184, 0.18);
                min-height: 172px;
            }
            .score-label {
                color: #94a3b8;
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .score-value {
                margin-top: 0.2rem;
                font-size: 3rem;
                font-weight: 700;
                line-height: 1;
            }
            .score-band {
                display: inline-block;
                margin-top: 0.8rem;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 600;
            }
            .signal-card {
                padding: 1rem 1.05rem;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.62);
                border: 1px solid rgba(148, 163, 184, 0.16);
                min-height: 220px;
            }
            .signal-name {
                color: #e2e8f0;
                font-weight: 600;
                font-size: 1.02rem;
            }
            .signal-value {
                margin-top: 0.35rem;
                font-size: 1.7rem;
                font-weight: 700;
            }
            .signal-risk {
                margin-top: 0.3rem;
                color: #f8fafc;
                font-size: 0.92rem;
            }
            .section-note {
                color: #64748b;
                font-size: 0.92rem;
            }
            .mini-card {
                padding: 0.9rem 1rem;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.54);
                border: 1px solid rgba(148, 163, 184, 0.14);
                min-height: 110px;
            }
            .mini-card .label {
                color: #94a3b8;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .mini-card .value {
                color: #e2e8f0;
                margin-top: 0.3rem;
                font-size: 1.1rem;
                font-weight: 600;
            }
            .empty-state {
                padding: 1.35rem 1.4rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.68));
                border: 1px solid rgba(148, 163, 184, 0.16);
                margin-bottom: 1rem;
            }
            .empty-state h3 {
                margin: 0 0 0.45rem 0;
                color: #f8fafc;
            }
            .empty-state p {
                margin: 0;
                color: #cbd5e1;
            }
            .action-card {
                padding: 1rem 1.05rem;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.62);
                border: 1px solid rgba(148, 163, 184, 0.14);
                min-height: 150px;
            }
            .action-card .title {
                color: #f8fafc;
                font-size: 1rem;
                font-weight: 600;
            }
            .action-card .body {
                margin-top: 0.5rem;
                color: #cbd5e1;
                font-size: 0.93rem;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _band_style(band: str) -> Dict[str, str]:
    if band == "Healthy":
        return {"background": "rgba(34, 197, 94, 0.16)", "color": "#86efac"}
    if band == "Watchlist":
        return {"background": "rgba(245, 158, 11, 0.16)", "color": "#fcd34d"}
    return {"background": "rgba(248, 113, 113, 0.16)", "color": "#fca5a5"}


def _render_hero(overview: Dict[str, float]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>AvaForensics MVP</h1>
            <p>
                Avalanche-native protocol health scoring built from real TVL time-series data,
                with price divergence and on-chain enrichment where available.
            </p>
            <p style="margin-top: 0.9rem;">
                {overview['protocols_analyzed']} protocols analyzed |
                baseline AUC {overview['baseline_auc']:.3f} |
                price coverage {overview['price_coverage']} |
                on-chain coverage {overview['onchain_coverage']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_score(protocol: Dict[str, object]) -> None:
    style = _band_style(str(protocol["risk_band"]))
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">Health Score</div>
            <div class="score-value">{protocol['health_score']:.1f}</div>
            <div class="score-band" style="background:{style['background']}; color:{style['color']};">
                {protocol['risk_band']}
            </div>
            <div style="margin-top: 1rem; color: #cbd5e1; font-size: 0.95rem;">
                Dead probability: {protocol['dead_probability'] * 100:.1f}%<br/>
                Category: {protocol['category']}<br/>
                Current label: {str(protocol['label']).title()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _history_chart(history: pd.DataFrame, protocol_name: str) -> go.Figure:
    figure = go.Figure()
    if history.empty:
        figure.add_annotation(
            text="No TVL history available for this protocol.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16},
        )
        figure.update_layout(height=420)
        return figure

    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["tvl"],
            mode="lines",
            line={"color": "#60a5fa", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.14)",
            name="TVL",
        )
    )
    if not history.empty:
        start_date = history["date"].min()
        cutoff_date = start_date + pd.Timedelta(days=90)
        figure.add_vrect(
            x0=start_date,
            x1=cutoff_date,
            fillcolor="rgba(250, 204, 21, 0.10)",
            line_width=0,
            annotation_text="First 90 days drive the score",
            annotation_position="top left",
        )

    figure.update_layout(
        title=f"{protocol_name} TVL History",
        xaxis_title="Date",
        yaxis_title="TVL (USD)",
        template="plotly_dark",
        height=420,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return figure


def _render_signal_cards(signals: List[Dict[str, object]]) -> None:
    columns = st.columns(len(signals))
    for column, signal in zip(columns, signals):
        with column:
            risk_tone = "#fca5a5" if signal["risk_score"] >= 66 else "#fcd34d" if signal["risk_score"] >= 40 else "#86efac"
            st.markdown(
                f"""
                <div class="signal-card">
                    <div class="signal-name">{signal['label']}</div>
                    <div class="signal-value">{signal['value']}</div>
                    <div class="signal-risk" style="color:{risk_tone};">
                        Risk pressure: {signal['risk_score']:.1f}/100
                    </div>
                    <div style="margin-top: 0.7rem; color:#cbd5e1; font-size:0.92rem;">
                        Alive median: {signal['alive_median']}<br/>
                        Dead median: {signal['dead_median']}
                    </div>
                    <div style="margin-top: 0.75rem; color:#94a3b8; font-size:0.92rem;">
                        {signal['description']}
                    </div>
                    <div style="margin-top: 0.75rem; color:#e2e8f0; font-size:0.9rem;">
                        {signal['narrative']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_supporting_metrics(items: List[Dict[str, str]]) -> None:
    rows = [items[index:index + 3] for index in range(0, len(items), 3)]
    for row in rows:
        columns = st.columns(len(row))
        for column, item in zip(columns, row):
            with column:
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <div class="label">{item['label']}</div>
                        <div class="value">{item['value']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_comparison(comparison: Dict[str, object] | None) -> None:
    st.subheader("Nearest Opposite-Side Peer")
    if not comparison:
        st.info("No comparison peer was available for this protocol.")
        return

    style = _band_style("Healthy" if comparison["label"] == "alive" else "High Risk")
    st.markdown(
        f"""
        <div class="mini-card" style="min-height: 140px;">
            <div class="label">Reference Case</div>
            <div class="value">{comparison['name']}</div>
            <div style="margin-top: 0.5rem; color:#cbd5e1;">
                Slug: {comparison['slug']}<br/>
                Current label: {str(comparison['label']).title()}<br/>
                Health score: {comparison['health_score']:.1f}<br/>
                Peak TVL: {comparison['peak_tvl']}<br/>
                Current TVL: {comparison['current_tvl']}
            </div>
            <div class="score-band" style="margin-top:0.8rem; background:{style['background']}; color:{style['color']};">
                Contrast Case
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_refresh(live_data: Dict[str, object]) -> None:
    if not live_data.get("available"):
        st.warning(live_data.get("reason", "Live refresh is currently unavailable."))
        return

    baseline = live_data["baseline"]
    live_metrics = live_data["live_metrics"]
    glacier = live_data["glacier"]

    st.subheader("Live Refresh")
    source_parts = ["DeFiLlama live TVL"]
    if live_data["sources"].get("glacier_live"):
        source_parts.append("Avalanche Glacier Data API")
    st.caption("Sources: " + " + ".join(source_parts))

    top_metrics = st.columns(4)
    top_metrics[0].metric(
        "Refreshed Health Score",
        "N/A" if pd.isna(baseline["refreshed_health_score"]) else f"{baseline['refreshed_health_score']:.1f}",
        baseline["risk_band"],
    )
    top_metrics[1].metric(
        "Live Monitor Score",
        "N/A" if pd.isna(live_metrics["monitor_score"]) else f"{live_metrics['monitor_score']:.1f}",
        live_metrics["monitor_band"],
    )
    top_metrics[2].metric(
        "Live Current TVL",
        "N/A" if pd.isna(live_metrics["current_tvl"]) else f"${live_metrics['current_tvl']:,.0f}",
        "N/A" if pd.isna(live_metrics["current_tvl_delta_pct"]) else f"{live_metrics['current_tvl_delta_pct'] * 100:+.1f}% vs local snapshot",
    )
    top_metrics[3].metric(
        "Last Live Data Point",
        str(pd.to_datetime(baseline["last_live_date"]).date()) if baseline.get("last_live_date") is not None else "N/A",
        f"{baseline['data_points']} points",
    )

    signal_columns = st.columns(max(1, len(live_data["live_signals"])))
    for column, signal in zip(signal_columns, live_data["live_signals"]):
        with column:
            risk_tone = "#fca5a5" if signal["risk_score"] >= 66 else "#fcd34d" if signal["risk_score"] >= 40 else "#86efac"
            st.markdown(
                f"""
                <div class="signal-card" style="min-height: 205px;">
                    <div class="signal-name">{signal['label']}</div>
                    <div class="signal-value">{signal['value']}</div>
                    <div class="signal-risk" style="color:{risk_tone};">
                        Live pressure: {signal['risk_score']:.1f}/100
                    </div>
                    <div style="margin-top: 0.7rem; color:#cbd5e1; font-size:0.92rem;">
                        Alive median: {signal['alive_median']}<br/>
                        Dead median: {signal['dead_median']}
                    </div>
                    <div style="margin-top: 0.75rem; color:#94a3b8; font-size:0.92rem;">
                        {signal['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    glacier_left, glacier_right = st.columns([1.15, 1], gap="large")
    with glacier_left:
        st.markdown("#### Avalanche On-Chain Snapshot")
        if glacier.get("available"):
            glacier_metrics = [
                {"label": "Contract Address", "value": glacier.get("address", "N/A")},
                {"label": "Asset", "value": " ".join(part for part in [glacier.get("name"), f"({glacier.get('symbol')})" if glacier.get("symbol") else ""] if part) or "N/A"},
                {"label": "ERC Type", "value": glacier.get("erc_type") or "N/A"},
                {"label": "Recent Token Transfers", "value": f"{int(glacier.get('recent_transfer_count') or 0):,}"},
                {"label": "Recent Transactions", "value": f"{int(glacier.get('recent_transaction_count') or 0):,}"},
                {"label": "Active Wallets", "value": f"{int(glacier.get('recent_active_wallets') or 0):,}"},
                {"label": "Last Activity", "value": str(pd.to_datetime(glacier['last_activity_at']).strftime('%Y-%m-%d %H:%M UTC')) if glacier.get('last_activity_at') is not None else "N/A"},
                {"label": "Native Balance", "value": f"{float(glacier.get('native_balance_avax') or 0):,.2f} AVAX"},
            ]
            _render_supporting_metrics(glacier_metrics)
        else:
            st.info(glacier.get("reason", "Avalanche Glacier data is unavailable for this protocol."))

    with glacier_right:
        st.markdown("#### Live TVL Context")
        context_items = [
            {"label": "Peak TVL", "value": "N/A" if pd.isna(live_metrics["peak_tvl"]) else f"${live_metrics['peak_tvl']:,.0f}"},
            {"label": "Drawdown From Peak", "value": "N/A" if pd.isna(live_metrics["drawdown_from_peak"]) else f"{live_metrics['drawdown_from_peak'] * 100:.1f}%"},
            {"label": "30-Day TVL Change", "value": "N/A" if pd.isna(live_metrics["tvl_30d_change"]) else f"{live_metrics['tvl_30d_change'] * 100:+.1f}%"},
            {"label": "90-Day TVL Change", "value": "N/A" if pd.isna(live_metrics["tvl_90d_change"]) else f"{live_metrics['tvl_90d_change'] * 100:+.1f}%"},
            {"label": "Lifespan", "value": f"{int(live_metrics['lifespan_days'])} days" if pd.notna(live_metrics["lifespan_days"]) else "N/A"},
            {"label": "Decline Count", "value": f"{int(live_metrics['consecutive_decline_days'])}" if pd.notna(live_metrics["consecutive_decline_days"]) else "N/A"},
        ]
        _render_supporting_metrics(context_items)

    with st.expander("See all live monitoring signals"):
        signal_frame = pd.DataFrame(live_data["all_live_signals"])
        st.dataframe(
            signal_frame[["label", "value", "risk_score", "alive_median", "dead_median", "narrative"]],
            use_container_width=True,
            hide_index=True,
        )


def _render_empty_protocol_state(protocols: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Start With a Protocol</h3>
            <p>
                AvaForensics should open as a product home, not as a pre-selected protocol report.
                Choose a protocol in the sidebar to unlock health score, TVL history, top risk signals,
                and live Avalanche monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(3)
    card_specs = [
        (
            "Explore a protocol",
            "Search by name or slug to inspect one protocol in depth. This is the main workflow for health scoring and explanation.",
        ),
        (
            "Scan the ecosystem",
            "Use the Leaderboard tab without selecting a protocol to see which Avalanche names look healthier or weaker.",
        ),
        (
            "Run live monitoring",
            "After selecting a protocol, use Live Refresh to pull the latest TVL context and Avalanche on-chain snapshot.",
        ),
    ]
    for column, (title, body) in zip(cards, card_specs):
        with column:
            st.markdown(
                f"""
                <div class="action-card">
                    <div class="title">{title}</div>
                    <div class="body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    examples = st.columns(2)
    available_slugs = set(protocols["slug"])
    with examples[0]:
        if st.button("Try Healthy Example: Benqi Lending", use_container_width=True, disabled="benqi-lending" not in available_slugs):
            st.session_state["selected_protocol_slug"] = "benqi-lending"
            st.rerun()
    with examples[1]:
        if st.button("Try Failed Example: Blizz Finance", use_container_width=True, disabled="blizz-finance" not in available_slugs):
            st.session_state["selected_protocol_slug"] = "blizz-finance"
            st.rerun()

    st.info("Tip: The Leaderboard and Method tabs work even before you choose a protocol.")


def main() -> None:
    _inject_styles()

    state = _load_state()
    overview = state["overview"]
    protocols = _protocol_frame(state).sort_values(["health_score", "name"], ascending=[False, True])

    _render_hero(overview)

    with st.sidebar:
        st.header("Protocol Explorer")
        search = st.text_input("Search by protocol name or slug")
        filtered = protocols
        if search:
            query = search.strip().lower()
            filtered = protocols[
                protocols["name"].str.lower().str.contains(query)
                | protocols["slug"].str.lower().str.contains(query)
            ]
            if filtered.empty:
                st.warning("No matches found. Showing all protocols instead.")
                filtered = protocols
        filtered = filtered.reset_index(drop=True)
        protocol_options = [""] + filtered["slug"].tolist()
        if st.session_state.get("selected_protocol_slug", "") not in protocol_options:
            st.session_state["selected_protocol_slug"] = ""
        selected_slug = st.selectbox(
            "Select a protocol",
            options=protocol_options,
            key="selected_protocol_slug",
            format_func=lambda slug: "Choose a protocol..." if slug == "" else f"{filtered.loc[filtered['slug'] == slug, 'name'].iloc[0]} ({slug})",
        )

        st.caption("The MVP retrains the baseline RandomForest from local research data on load.")
        st.metric("Protocols scored", overview["protocols_analyzed"])
        st.metric("High-risk protocols", overview["high_risk_count"])
        st.metric("Healthy protocols", overview["healthy_count"])

    live_cache = st.session_state.setdefault("live_refresh_cache", {})

    top_metrics = st.columns(4)
    top_metrics[0].metric("Protocols analyzed", overview["protocols_analyzed"])
    top_metrics[1].metric("Baseline AUC", f"{overview['baseline_auc']:.3f}")
    top_metrics[2].metric("Price coverage", f"{overview['price_coverage']} protocols")
    top_metrics[3].metric("On-chain coverage", f"{overview['onchain_coverage']} protocols")

    tabs = st.tabs(["Protocol View", "Leaderboard", "Method"])

    with tabs[0]:
        sidebar_refresh = False
        with st.sidebar:
            sidebar_refresh = st.button("Live Refresh Selected Protocol", use_container_width=True, disabled=not selected_slug)
            if sidebar_refresh and selected_slug:
                with st.spinner("Fetching live TVL and Avalanche on-chain data..."):
                    live_cache[selected_slug] = refresh_protocol_live(state, selected_slug)

        if not selected_slug:
            _render_empty_protocol_state(protocols)
        else:
            protocol_view = build_protocol_view(state, selected_slug)
            live_data = live_cache.get(selected_slug)
            left, right = st.columns([0.85, 1.45], gap="large")
            with left:
                _render_score(protocol_view["protocol"])
                st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
                _render_comparison(protocol_view["comparison"])

            with right:
                chart_history = live_data["history"] if live_data and live_data.get("available") else protocol_view["history"]
                figure = _history_chart(chart_history, protocol_view["protocol"]["name"])
                st.plotly_chart(figure, use_container_width=True)
                if live_data and live_data.get("available"):
                    st.caption("Chart is using live TVL history from DeFiLlama. The highlighted first 90-day window still drives the baseline model.")
                else:
                    st.caption("The highlighted first 90-day window is the basis for baseline prediction features.")

            st.subheader("Top Risk Signals")
            st.caption("These signals are computed from the protocol's early TVL curve and benchmarked against alive vs dead medians.")
            _render_signal_cards(protocol_view["risk_signals"])

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            if live_data:
                _render_live_refresh(live_data)
            else:
                st.info("Run Live Refresh to pull the latest TVL snapshot and Avalanche Glacier activity for this protocol.")

            st.subheader("Supporting Context")
            _render_supporting_metrics(protocol_view["supporting_metrics"])

            with st.expander("See all tracked signals"):
                all_signals = pd.DataFrame(protocol_view["all_signals"])
                st.dataframe(
                    all_signals[["label", "value", "risk_score", "alive_median", "dead_median", "narrative"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[1]:
        filter_columns = st.columns([0.9, 0.9, 2.2])
        with filter_columns[0]:
            band_filter = st.selectbox("Risk band", ["All", "Healthy", "Watchlist", "High Risk"])
        with filter_columns[1]:
            top_n = st.selectbox("Rows", [10, 25, 50, 100], index=1)
        with filter_columns[2]:
            st.markdown(
                "<div class='section-note'>Leaderboard is ranked by model-derived health score. "
                "Current label is the latest known alive/dead tag from the local dataset.</div>",
                unsafe_allow_html=True,
            )

        leaderboard = _load_leaderboard(state, band_filter=band_filter, top_n=top_n)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("How This MVP Works")
        st.markdown(
            """
            - The app loads `422` Avalanche protocols from the local research dataset.
            - It retrains the baseline RandomForest on the first `90` days of TVL-derived features.
            - Health score is calculated as `100 * (1 - dead_probability)`.
            - Risk signals are benchmarked against the median behavior of alive vs dead projects.
            - Price divergence and Avalanche on-chain data appear as enrichment only where coverage exists.
            """
        )

        method_metrics = st.columns(4)
        method_metrics[0].metric("Alive protocols", overview["alive_protocols"])
        method_metrics[1].metric("Dead protocols", overview["dead_protocols"])
        method_metrics[2].metric("Accuracy", f"{overview['baseline_accuracy']:.3f}")
        method_metrics[3].metric("Watchlist", overview["watchlist_count"])

        st.markdown(
            """
            The MVP is intentionally narrow for Stage 2:

            - One product surface: protocol health explorer.
            - One primary model: TVL early-warning baseline.
            - Two optional enrichments: price/TVL divergence and Avalanche on-chain activity.
            """
        )


if __name__ == "__main__":
    main()
