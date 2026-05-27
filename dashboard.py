"""Streamlit demo dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_project_ids, group_by_project, load_promise_dataset
from src.experiments import OUTPUTS_DIR
from src.prompts import PROMPT_DIR
from src.rq_results import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RUNS_DIR,
    PILOT_PROJECT_IDS,
    aggregate_rq_results,
    save_rq_results,
)

from src.pilot_prompt_sensitivity import (
    DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH,
    PILOT_VARIANT_DIRS,
    aggregate_pilot_prompt_sensitivity,
    save_pilot_prompt_sensitivity,
)



DATA_DIR = Path("data")
PILOT_V2_DIR = OUTPUTS_DIR / "Pilot-V2"
PILOT_V3_DIR = OUTPUTS_DIR / "Pilot-V3"

TARGET_CATEGORIES = ["performance", "security", "maintainability"]

def main() -> None:
    st.set_page_config(page_title="NFR-LLM Pipeline", layout="wide")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("NFR-LLM Pipeline")
        st.caption("Controls")

        csv_files = sorted(DATA_DIR.glob("*.csv"))

        if not csv_files:
            st.warning("No CSV files found in `data/`. Place the dataset there.")
            return

        selected_file = st.selectbox(
            "Dataset file",
            csv_files,
            format_func=lambda p: p.name,
        )

        st.divider()
        page = st.radio(
            "Page",
            [
                "Data Explorer",
                "Evaluation",
                "Pilot Analysis",
                "Prompt Templates",
                "RQ Results",
            ],
        )

    # ---------------- Load Data ----------------
    requirements = load_promise_dataset(selected_file)
    grouped = group_by_project(requirements)
    project_ids = get_project_ids(requirements)

    # ---------------- Sidebar: page-specific controls ----------------
    selected_project: str | None = None
    selected_result: Path | None = None
    selected_projects: list[str] = []

    if page == "Data Explorer":
        with st.sidebar:
            st.divider()
            st.markdown("### Filters")
            selected_project = st.selectbox(
                "Project",
                ["All"] + project_ids,
            )

    elif page == "Evaluation":
        with st.sidebar:
            st.divider()
            st.markdown("### Filters")

            # Available experiment folders
            run_dirs = sorted(
                [
                    p for p in OUTPUTS_DIR.iterdir()
                    if p.is_dir()
                    and (
                        p.name.startswith("Pilot-")
                        or p.name.startswith("run-")
                    )
                ]
            )

            if not run_dirs:
                st.info("No experiment folders found in outputs/")
            else:
                selected_run = st.selectbox(
                    "Experiment Run",
                    run_dirs,
                    format_func=lambda p: p.name,
                )

                # All JSON result files inside selected folder
                result_files = sorted(
                    selected_run.glob("*.json"),
                    reverse=True,
                )

                if result_files:
                    selected_result = st.selectbox(
                        "Project Result",
                        result_files,
                        format_func=lambda p: p.name,
                    )
                else:
                    st.info(f"No JSON result files found in {selected_run.name}")
    elif page == "RQ Results":
        with st.sidebar:
            st.divider()
            st.markdown("### Filters")
            default_selection = [
                pid for pid in project_ids if pid not in PILOT_PROJECT_IDS
            ]
            selected_projects = st.multiselect(
                "Select projects to include",
                options=project_ids,
                default=default_selection,
            )

    # ---------------- Main: route to active page ----------------
    if page == "Data Explorer":
        _render_data_explorer(requirements, grouped, project_ids, selected_project)
    elif page == "Evaluation":
        _render_evaluation(selected_result)
    elif page == "Pilot Analysis":
        _render_pilot_analysis()
    elif page == "Prompt Templates":
        _render_prompt_templates()
    elif page == "RQ Results":
        _render_rq_results(selected_projects, Path(str(selected_file)))


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

def _render_data_explorer(
    requirements: list,
    grouped: dict,
    project_ids: list[str],
    selected_project: str | None,
) -> None:
    st.header("PROMISE NFR Dataset")
    st.caption("Data inspection view")

    total_all = len(requirements)
    total_nfr_all = sum(1 for r in requirements if r.is_nfr)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total requirements", total_all)
    col2.metric("Projects", len(project_ids))
    col3.metric("NFRs", total_nfr_all)

    # ---------------- Project Filtering ----------------
    if selected_project == "All" or selected_project is None:
        display_reqs = requirements
    else:
        display_reqs = grouped[selected_project]

        proj_total = len(display_reqs)
        proj_nfr = sum(1 for r in display_reqs if r.is_nfr)
        proj_fr = proj_total - proj_nfr

        st.markdown(f"**Project {selected_project}**")

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Requirements in project", proj_total)
        pcol2.metric("NFRs in project", proj_nfr)
        pcol3.metric("FRs in project", proj_fr)

    # ---------------- Table ----------------
    df = pd.DataFrame(
        [
            {
                "ID": r.req_id,
                "Project": r.project_id,
                "Text": r.text,
                "Is NFR": r.is_nfr,
                "Category": r.category,
            }
            for r in display_reqs
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_evaluation(selected_result: Path | None) -> None:
    st.header("Evaluation Results")

    if selected_result is None:
        st.info(
            "No result file selected"
        )
        return

    data = json.loads(selected_result.read_text(encoding="utf-8"))
    step = data.get("step", "")
    eval_data = data.get("evaluation", {})

    # Compact metadata line
    project_id = data.get("project_id", "?")
    variant_info = data.get("variant_info", {})
    prompt_variant = variant_info.get("prompt_variant", "default")
    is_pilot = data.get("is_pilot", False)
    pilot_ids = variant_info.get("pilot_ids", [])

    pilot_tag = " `PILOT`" if is_pilot else ""
    st.caption(
        f"Selected: `{selected_result.name}`| "
        f"Project **{project_id}** | Step `{step}` | Variant `{prompt_variant}` | {pilot_tag}"
    )


    # Pick the right slices depending on step type
    if step == "identify":
        id_section, cls_section = eval_data, None
    elif step == "classify":
        id_section, cls_section = None, eval_data
    elif step == "pipeline":
        id_section = eval_data.get("identification", {})
        cls_section = eval_data.get("classification", {})
    else:
        id_section, cls_section = None, None

    if id_section:
        st.subheader("Identification")
        _render_identification(id_section)

    if cls_section:
        if id_section:
            st.divider()
        st.subheader("Classification")
        _render_classification(cls_section)

    # Pipeline-only: end-to-end error breakdown
    error_breakdown = eval_data.get("error_breakdown") if step == "pipeline" else None
    if error_breakdown:
        st.divider()
        st.subheader("Pipeline Error Breakdown")
        _render_error_breakdown(error_breakdown)

    if id_section is None and cls_section is None:
        st.info("No identification or classification data found in this file.")


def _render_identification(id_section: dict) -> None:
    cm = id_section.get("confusion_matrix", {})
    metrics = id_section.get("metrics", {})

    # Counts from confusion matrix
    tp = cm.get("NFR", {}).get("NFR", 0)  # correct NFR
    fn = cm.get("NFR", {}).get("FR", 0)   # missed NFR
    fp = cm.get("FR", {}).get("NFR", 0)   # false alarm
    tn = cm.get("FR", {}).get("FR", 0)    # correct FR
    total = tp + fn + fp + tn

    # summary
    if total > 0:
        correct = tp + tn
        wrong = fp + fn
        st.markdown(
            f"**Out of {total} requirements:**\n"
            f"- **{correct}** classified correctly ({_pct(correct, total)})\n"
            f"- **{wrong}** classified incorrectly ({_pct(wrong, total)})"
        )

    if id_section.get("no_fr_in_project"):
        st.warning(
            "This project has **0 functional requirements**. FR metrics are 0.0."
        )

   # Stacked bar
    if (tp + fn) > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Found",
            x=["Actual NFR"],
            y=[tp],
            marker_color="#2ecc71",
            text=[f"{tp} found"],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Missed",
            x=["Actual NFR"],
            y=[fn],
            marker_color="#e74c3c",
            text=[f"{fn} missed" if fn > 0 else ""],
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            title="Identification: What happened to the NFRs?",
            yaxis_title="Count",
            height=350,
            font=dict(size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        if fp > 0:
            st.caption(
                f"Plus **{fp} false alarms** - FRs the model wrongly called NFR."
            )

    # P/R/F1 cards for the NFR class
    if metrics:
        nfr_m = metrics.get("NFR", {})

        st.markdown("**NFR Detection**")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Precision", f"{nfr_m.get('precision', 0):.1%}",
            help="Of all requirements the model called NFR, how many were actually NFR?",
        )
        col2.metric(
            "Recall", f"{nfr_m.get('recall', 0):.1%}",
            help="Of all actual NFRs, how many did the model find?",
        )
        col3.metric(
            "F1", f"{nfr_m.get('f1', 0):.1%}",
            help="Harmonic mean of precision and recall.",
        )


def _render_classification(cls_section: dict) -> None:
    cm = cls_section.get("confusion_matrix", {})
    metrics = cls_section.get("metrics", {})

    if not metrics:
        st.info("No classification metrics found.")
        return

    # summary
    if cm:
        _render_classification_summary(cm)

    # Grouped bar chart: P/R/F1 per target category
    categories = ["performance", "security", "maintainability"]
    rows = []
    for cat in categories:
        m = metrics.get(cat, {})
        rows.append({"Category": cat.title(), "Metric": "Precision", "Score": m.get("precision", 0)})
        rows.append({"Category": cat.title(), "Metric": "Recall", "Score": m.get("recall", 0)})
        rows.append({"Category": cat.title(), "Metric": "F1", "Score": m.get("f1", 0)})

    df = pd.DataFrame(rows)
    color_map = {"Precision": "#3498db", "Recall": "#2ecc71", "F1": "#f39c12"}

    fig = px.bar(
        df, x="Category", y="Score", color="Metric",
        barmode="group",
        color_discrete_map=color_map,
        title="Classification Performance per Category",
        text=df["Score"].apply(lambda x: f"{x:.0%}"),
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        height=400,
        font=dict(size=14),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    if cm:
        with st.expander("Show confusion matrix"):
            _render_confusion_matrix(cm)

    # Residual "other" class analysis
    other = cls_section.get("other_analysis", {})
    if other:
        _render_other_analysis(other)


def _render_other_analysis(other: dict) -> None:
    """Residual 'other' class analysis"""
    st.markdown("#### Residual 'other' Class Analysis")

    true_count = other.get("true_other_count", 0)
    pred_count = other.get("predicted_other_count", 0)

    col1, col2 = st.columns(2)
    col1.metric(
        "Actually 'other'", true_count,
        help="NFRs in dataset that don't belong to performance/security/maintainability.",
    )
    col2.metric(
        "Model predicted 'other'", pred_count,
        delta=pred_count - true_count,
        delta_color="inverse",
        help="Negative delta = model under-predicts 'other' (assigns target categories instead).",
    )

    confused_as_other = other.get("target_confused_as_other", {})
    other_as_target = other.get("other_confused_as_target", {})

    if confused_as_other or other_as_target:
        col1, col2 = st.columns(2)
        with col1:
            if confused_as_other:
                st.markdown("**Target categories lost to 'other':**")
                for cat, count in confused_as_other.items():
                    st.markdown(f"- {cat}: **{count}** wrongly called 'other'")
            else:
                st.success("No target categories were misclassified as 'other'.")
        with col2:
            if other_as_target:
                st.markdown("**'Other' NFRs mistaken for target categories:**")
                for cat, count in other_as_target.items():
                    st.markdown(f"- {count}× wrongly called **{cat}**")
            else:
                st.success("No 'other' NFRs leaked into target categories.")


def _render_error_breakdown(breakdown: dict) -> None:
    """Pipeline-level error breakdown with a funnel chart."""
    total = breakdown.get("total_gold_nfrs", 0)
    found = breakdown.get("correctly_identified", 0)
    missed = breakdown.get("missed_in_identification", 0)
    fp_count = breakdown.get("false_positive_count", 0)

    # Funnel: gold NFRs --> found in step 1 --> missed
    fig = go.Figure(go.Funnel(
        y=["Gold NFRs in dataset", "Found by model (Step 1)", "Missed by model"],
        x=[total, found, missed],
        textinfo="value+percent initial",
        marker=dict(color=["#3498db", "#2ecc71", "#e74c3c"]),
    ))
    fig.update_layout(
        title="Pipeline flow: what happened to the NFRs?",
        height=300,
        font=dict(size=14),
    )
    st.plotly_chart(fig, use_container_width=True)

    if missed > 0:
        missed_ids = breakdown.get("missed_nfr_ids", [])
        st.error(
            f"**{missed} NFRs were missed in identification** and never reached "
            f"classification: {', '.join(missed_ids)}"
        )

    if fp_count > 0:
        fps = breakdown.get("false_positives", [])
        st.warning(
            f"**{fp_count} FRs were falsely identified as NFR** and sent to classification:"
        )
        fp_df = pd.DataFrame(fps)
        if not fp_df.empty:
            fp_df.columns = ["Requirement", "True Type", "Model Classified As"]
            st.dataframe(fp_df, use_container_width=True, hide_index=True)


def _render_confusion_matrix(cm: dict) -> None:
    """Row-normalized confusion matrix with count + row % per cell."""
    if not cm:
        st.info("No confusion matrix data available.")
        return
    
    preferred_order = [
        "performance",
        "security",
        "maintainability",
        "other",
        "_missed",
        "_unclassified",
        "F",
    ]
    labels = [lab for lab in preferred_order if lab in cm]
    labels += [lab for lab in cm.keys() if lab not in labels]

    # Raw counts and row totals
    raw_matrix = [[cm[t].get(p, 0) for p in labels] for t in labels]
    row_totals = [sum(row) for row in raw_matrix]

    # Row-normalized matrix with annotation text "count\n(pct)"
    norm_matrix = []
    annotation_text = []
    for i, row in enumerate(raw_matrix):
        total = row_totals[i]
        norm_row = []
        ann_row = []
        for count in row:
            pct = (count / total) if total > 0 else 0.0
            norm_row.append(pct)
            ann_row.append(f"{count}<br>({pct:.1%})")
        norm_matrix.append(norm_row)
        annotation_text.append(ann_row)

    y_labels = [f"{lab} (n={tot})" for lab, tot in zip(labels, row_totals)]

    st.info(
        "**How to read this matrix:**\n\n"
        "- Each **row** = the real category.\n"
        "- Each **column** = what the model predicted.\n"
        "- Each cell shows **count + row percentage**.\n"
        "- The **diagonal** = correct predictions.\n"
        "- Off-diagonal = mistakes.\n\n"
    )

    fig = ff.create_annotated_heatmap(
        z=norm_matrix,
        x=labels,
        y=y_labels,
        annotation_text=annotation_text,
        colorscale="Blues",
        showscale=True,
        hoverinfo="z",
    )

    for ann in fig.layout.annotations:
        ann.font.size = 13

    fig.update_layout(
        title="Row-normalized confusion matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=520,
        font=dict(size=13),
        coloraxis_colorbar=dict(title="Row %"),
    )

    for trace in fig.data:
        trace.hovertemplate = (
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Row share: %{z:.1%}<extra></extra>"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Tip: dark cells outside the diagonal show where the model most often makes mistakes."
    )

def _pilot_pair_key(path: Path) -> str:
    """Strip timestamp suffix so V2/V3 files match on `{step}_{project}`."""
    parts = path.stem.split("_")
    # Files look like step_project_YYYYMMDD_HHMMSS, drop the trailing two parts
    return "_".join(parts[:-2]) if len(parts) >= 4 else path.stem


def _render_pilot_analysis() -> None:
    st.header("Pilot Analysis")

    data_path = DATA_DIR / "nfr.csv"

    if not PILOT_V2_DIR.exists() or not PILOT_V3_DIR.exists():
        st.info("Need both `outputs/Pilot-V2/` and `outputs/Pilot-V3/` folders.")
        return

    V2_files = {_pilot_pair_key(p): p for p in PILOT_V2_DIR.glob("*.json")}
    V3_files = {_pilot_pair_key(p): p for p in PILOT_V3_DIR.glob("*.json")}
    pair_keys = sorted(set(V2_files) & set(V3_files))

    if not pair_keys:
        st.info("No matching `{step}_{project}` files across Pilot-V2 and Pilot-V3.")
        return

    selected_key = st.selectbox("Result", pair_keys, key="pilot_pair_key")
    file_a = V2_files[selected_key]
    file_b = V3_files[selected_key]

    data_a = json.loads(file_a.read_text(encoding="utf-8"))
    data_b = json.loads(file_b.read_text(encoding="utf-8"))

    step = data_a.get("step", "")
    project_id = data_a.get("project_id", "?")
    st.caption(f"Project **{project_id}** | Step `{step}` — comparing **Prompt V2** vs **Prompt V3**")
    st.divider()

    eval_a = data_a.get("evaluation", {})
    eval_b = data_b.get("evaluation", {})

    if step == "identify":
        id_a, id_b = eval_a, eval_b
        cls_a, cls_b = None, None
    elif step == "classify":
        id_a, id_b = None, None
        cls_a, cls_b = eval_a, eval_b
    elif step == "pipeline":
        id_a = eval_a.get("identification", {})
        id_b = eval_b.get("identification", {})
        cls_a = eval_a.get("classification", {})
        cls_b = eval_b.get("classification", {})
    else:
        st.info("Unknown step type; nothing to compare.")
        return

    if id_a and id_b:
        st.subheader("Identification — prompt effect on NFR detection")
        _render_metric_diff(
            id_a.get("metrics", {}).get("NFR", {}),
            id_b.get("metrics", {}).get("NFR", {}),
            label="NFR",
        )

    if cls_a and cls_b:
        if id_a:
            st.divider()
        st.subheader("Classification — prompt effect per category")
        _render_classification_diff(cls_a, cls_b)

    if step == "pipeline":
        eb_a = eval_a.get("error_breakdown", {})
        eb_b = eval_b.get("error_breakdown", {})
        if eb_a and eb_b:
            st.divider()
            st.subheader("Pipeline error breakdown")
            _render_error_breakdown_diff(eb_a, eb_b)

    if "pilot_prompt_sensitivity" not in st.session_state:
        cached = _load_json_if_exists(DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH)
        if cached:
            st.session_state["pilot_prompt_sensitivity"] = cached

    if st.button("Compute pilot prompt sensitivity", type="primary"):
        with st.spinner("Computing pilot prompt-sensitivity results..."):
            try:
                results = aggregate_pilot_prompt_sensitivity(
                    outputs_dir=OUTPUTS_DIR,
                    data_path=data_path,
                )
                save_pilot_prompt_sensitivity(
                    results,
                    DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH,
                )
            except Exception as exc:
                st.error(f"Pilot prompt-sensitivity aggregation failed: {exc}")
            else:
                st.session_state["pilot_prompt_sensitivity"] = results
                st.success(
                    "Saved pilot prompt-sensitivity results to "
                    f"`{DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH}`"
                )

    results = st.session_state.get("pilot_prompt_sensitivity")
    if not results:
        st.info("Click **Compute pilot prompt sensitivity**")
        return

    missing = results.get("missing_files", [])
    if missing:
        with st.expander(f"⚠ Missing pilot files ({len(missing)})", expanded=False):
            st.dataframe(pd.DataFrame(missing), use_container_width=True, hide_index=True)

    pooled = results.get("pooled_projects_6_9", {})
    variants = [v for v in PILOT_VARIANT_DIRS.keys() if v in pooled]

    st.subheader("Pooled pilot prompt-sensitivity results")
    
    pooled_rows = []
    for variant in variants:
        row = pooled.get(variant, {})
        pooled_rows.append(
            {
                "Prompt": variant,
                "NFR F1": float(row.get("nfr_f1", 0.0)),
                "Gold macro-F1": float(row.get("gold_macro_f1", 0.0)),
                "End-to-end macro-F1": float(row.get("end_to_end_macro_f1", 0.0)),
            }
        )

    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        st.dataframe(
            pooled_df.style.format(
                {
                    "NFR F1": "{:.3f}",
                    "Gold macro-F1": "{:.3f}",
                    "End-to-end macro-F1": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No pooled pilot results found.")

    st.subheader("Project-level gold classification F1")

    project_level = results.get("project_level_gold_classification_f1", {})
    project_rows = []

    for project_id in ["6", "9"]:
        variant_data = project_level.get(project_id, {})
        for variant in variants:
            entry = variant_data.get(variant, {})
            f1s = entry.get("f1", {})
            support = entry.get("support", {})
            project_rows.append(
                {
                    "Project": project_id,
                    "Prompt": variant,
                    "Performance F1": float(f1s.get("performance", 0.0)),
                    "Security F1": float(f1s.get("security", 0.0)),
                    "Maintainability F1": float(f1s.get("maintainability", 0.0)),
                    "Performance n": int(support.get("performance", 0)),
                    "Security n": int(support.get("security", 0)),
                    "Maintainability n": int(support.get("maintainability", 0)),
                }
            )

    if project_rows:
        project_df = pd.DataFrame(project_rows)
        st.dataframe(
            project_df.style.format(
                {
                    "Performance F1": "{:.3f}",
                    "Security F1": "{:.3f}",
                    "Maintainability F1": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No project-level pilot results found.")

def _render_metric_diff(metrics_V2: dict, metrics_V3: dict, label: str = "") -> None:
    """Three side-by-side cards (precision/recall/F1) showing V3 - V2 delta."""
    cols = st.columns(3)
    for col, key, name in zip(cols, ["precision", "recall", "f1"], ["Precision", "Recall", "F1"]):
        V2_val = metrics_V2.get(key, 0.0)
        V3_val = metrics_V3.get(key, 0.0)
        delta = V3_val - V2_val
        col.metric(
            f"{label + ' ' if label else ''}{name}",
            f"{V2_val:.1%} → {V3_val:.1%}",
            delta=f"{delta:+.1%}",
        )


def _render_classification_diff(cls_V2: dict, cls_V3: dict) -> None:
    """Per-category P/R/F1 metric cards (V2 → V3) for each target category."""

    metrics_V2 = cls_V2.get("metrics", {})

    metrics_V3 = cls_V3.get("metrics", {})

    categories = ["performance", "security", "maintainability"]

    for cat in categories:
        st.markdown(f"**{cat.title()}**")
        _render_metric_diff(metrics_V2.get(cat, {}), metrics_V3.get(cat, {}))

def _render_error_breakdown_diff(eb_V2: dict, eb_V3: dict) -> None:
    """Compare missed NFRs and false positives between V2 and V3 as metric cards."""
    items = [
        ("Gold NFRs",                "total_gold_nfrs",         False),
        ("Correctly identified",     "correctly_identified",    False),
        ("Missed in identification", "missed_in_identification", True),
        ("False positives",          "false_positive_count",    True),
        ]
    cols = st.columns(len(items))
    for col, (label, key, is_error) in zip(cols, items):  
        V2_val = eb_V2.get(key, 0)
        V3_val = eb_V3.get(key, 0)
        delta = V3_val - V2_val
        col.metric(
            label,
            f"{V3_val}",
            delta=f"{delta:+}" if delta != 0 else "0",
            delta_color="inverse" if is_error else "normal",
            help=f"V2: {V2_val} → V3: {V3_val}",
        )


def _render_classification_summary(cm: dict) -> None:
    """Plain-language summary mirroring the identification summary."""
    # Skip ("F" = false positives from step 1, "_missed", "_unclassified")
    gold_categories = [
        c for c in cm.keys() if c not in {"F", "_missed", "_unclassified"}
    ]

    total = 0
    correct = 0
    per_category: list[dict] = []

    for cat in gold_categories:
        row = cm.get(cat, {})
        row_total = sum(row.values())
        if row_total == 0:
            continue
        row_correct = row.get(cat, 0)
        total += row_total
        correct += row_correct
        per_category.append({
            "category": cat,
            "gold": row_total,
            "correct": row_correct,
            "wrong": row_total - row_correct,
            "row": row,
        })

    if total == 0:
        return

    wrong = total - correct

    st.markdown(
        f"**Out of {total} NFRs that reached classification:**\n"
        f"- **{correct}** classified correctly ({_pct(correct, total)})\n"
        f"- **{wrong}** classified incorrectly ({_pct(wrong, total)})"
    )

    st.markdown("**Per category:**")
    for entry in per_category:
        cat = entry["category"]
        gold = entry["gold"]
        right = entry["correct"]
        bad = entry["wrong"]
        pct = _pct(right, gold)

        mistakes = {k: v for k, v in entry["row"].items() if k != cat and v > 0}
        if bad == 0:
            detail = "all correct"
        else:
            parts = []
            for k, v in mistakes.items():
                if k == "_missed":
                    parts.append(f"{v} missed in step 1")
                elif k == "_unclassified":
                    parts.append(f"{v} got no category")
                else:
                    parts.append(f"{v} wrongly called **{k}**")
            detail = ", ".join(parts)

        st.markdown(
            f"- **{cat.title()}** - {gold} in dataset: "
            f"{right} correct ({pct}), {bad} wrong ({detail})"
        )

def _render_prompt_templates() -> None:
    st.header("Prompt Templates")
    st.markdown(
        "This page shows the actual prompt templates used in the experiments. "
        "The goal is to provide transparency and help understand how the models were prompted."
    )
    templates: dict[str, str] = {}
    if not PROMPT_DIR.exists():
        st.warning(f"Prompts directory not found: {PROMPT_DIR}")
        return

    for p in sorted(PROMPT_DIR.iterdir()):
        if p.is_file():
            try:
                templates[p.stem] = p.read_text(encoding="utf-8")
            except Exception:
                templates[p.stem] = "(failed to read file)"

    if not templates:
        st.info("No prompt templates found in the prompts directory.")
        return

    for name, content in templates.items():
        # Show each template inside a collapsed expander
        with st.expander(name, expanded=False):
            st.code(content, language="text")

def _pct(n: int, total: int) -> str:
    return f"{n / total:.0%}" if total > 0 else "0%"


# ----------------------------------------------------------------------
# RQ Results page
# ----------------------------------------------------------------------

def _fmt_stats(stats: dict | None) -> str:
    if not stats:
        return "—"
    mean = stats.get("mean", 0.0)
    std = stats.get("std", 0.0)
    return f"{mean:.3f} ± {std:.3f}"


def _stats_to_row(label: str, prf: dict) -> dict[str, str]:
    return {
        "": label,
        "Precision": _fmt_stats(prf.get("precision")),
        "Recall": _fmt_stats(prf.get("recall")),
        "F1": _fmt_stats(prf.get("f1")),
    }


def _f1_bar_chart_with_error_bars(
    target_metrics: dict[str, dict],
    title: str,
) -> go.Figure:
    categories = [cat for cat in TARGET_CATEGORIES if cat in target_metrics]
    means = [target_metrics[cat]["f1"]["mean"] for cat in categories]
    stds = [target_metrics[cat]["f1"]["std"] for cat in categories]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[c.title() for c in categories],
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
                marker_color="#313030",
            )
        ]
    )
    for cat, mean, std in zip(categories, means, stds):
        fig.add_annotation(
            x=cat.title(),
            y=mean,
            text=f"<b>{mean:.3f}</b> ± {std:.3f}",
            showarrow=False,
            xshift=22,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            borderpad=3,
        )
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1.15], title="F1"),
        height=380,
        font=dict(size=14),
        showlegend=False,
    )
    return fig


def _render_rq_confusion_matrix(cm: dict, title: str = "Confusion matrix") -> None:
    if not cm:
        st.info("No confusion matrix data available.")
        return

    preferred_order = [
        "NFR",
        "FR",
        "performance",
        "security",
        "maintainability",
        "other",
        "_missed",
        "_unclassified",
        "F",
    ]
    labels = [lab for lab in preferred_order if lab in cm]
    labels += [lab for lab in cm.keys() if lab not in labels]
    cols = list(labels)
    for row in cm.values():
        for col in row.keys():
            if col not in cols:
                cols.append(col)

    raw_matrix = [[float(cm.get(t, {}).get(p, 0)) for p in cols] for t in labels]
    row_totals = [sum(row) for row in raw_matrix]

    norm_matrix: list[list[float]] = []
    annotation_text: list[list[str]] = []
    for i, row in enumerate(raw_matrix):
        total = row_totals[i]
        norm_row: list[float] = []
        ann_row: list[str] = []
        for count in row:
            pct = (count / total) if total > 0 else 0.0
            norm_row.append(pct)
            ann_row.append(f"{count:.1f}<br>({pct:.1%})")
        norm_matrix.append(norm_row)
        annotation_text.append(ann_row)

    y_labels = [f"{lab} (n={tot:.1f})" for lab, tot in zip(labels, row_totals)]

    fig = ff.create_annotated_heatmap(
        z=norm_matrix,
        x=cols,
        y=y_labels,
        annotation_text=annotation_text,
        colorscale="Greys",
        showscale=True,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=520,
        font=dict(size=13),
    )
    for trace in fig.data:
        trace.hovertemplate = (
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Row share: %{z:.1%}<extra></extra>"
        )

    st.plotly_chart(fig, use_container_width=True)


def _render_rq_results(selected_projects: list[str], data_path: Path) -> None:
    """Final results organized by research question."""
    st.header("Final Results by Research Question")

    runs_dir = DEFAULT_RUNS_DIR

    if st.button("Compute RQ results", type="primary"):
        if not selected_projects:
            st.error("Select at least one project before computing.")
            return
        with st.spinner("Aggregating dataset-level metrics across runs…"):
            try:
                results = aggregate_rq_results(
                    runs_dir=runs_dir,
                    data_path=data_path,
                    selected_project_ids=list(selected_projects),
                )
                save_rq_results(results, DEFAULT_OUTPUT_PATH)
            except Exception as exc:
                st.error(f"Aggregation failed: {exc}")
                return
        st.session_state["rq_results"] = results

    results = st.session_state.get("rq_results")
    if not results:
        return

    missing = results["config"].get("missing_files", [])
    if missing:
        with st.expander(
            f"⚠ Missing files in some runs ({len(missing)}).",
            expanded=False,
        ):
            st.dataframe(pd.DataFrame(missing), use_container_width=True, hide_index=True)

    # ---- Tabs ----
    tabs = st.tabs(
        [
            "RQa Identification",
            "RQb Classification",
            "RQc Pipeline",
        ]
    )

    with tabs[0]:
        _render_rq_a_tab(results)
    with tabs[1]:
        _render_rq_b_tab(results)
    with tabs[2]:
        _render_rq_c_tab(results)


def _render_rq_a_tab(results: dict) -> None:
    section = results["rqa_identification"]
    st.subheader(section["title"])

    nfr_metrics = section["nfr_metrics"]
    fr_metrics = section["fr_metrics_supplementary"]

    st.markdown("#### NFR identification (primary focus)")
    cols = st.columns(3)
    cols[0].metric("Precision (NFR)", _fmt_stats(nfr_metrics["precision"]))
    cols[1].metric("Recall (NFR)", _fmt_stats(nfr_metrics["recall"]))
    cols[2].metric("F1 (NFR)", _fmt_stats(nfr_metrics["f1"]))

    table_rows = [
        _stats_to_row("NFR", nfr_metrics),
        _stats_to_row("FR (supplementary)", fr_metrics),
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### FR / NFR confusion matrix")
    _render_rq_confusion_matrix(
        section["mean_confusion_matrix"],
        title="Identification confusion matrix (mean counts)",
    )


def _render_rq_b_tab(results: dict) -> None:
    section = results["rqb_classification_gold"]
    st.subheader(section["title"])

    target_metrics = section["target_category_metrics"]

    table_rows = [
        _stats_to_row(cat.title(), target_metrics[cat]) for cat in TARGET_CATEGORIES
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        _f1_bar_chart_with_error_bars(
            target_metrics, "Classification F1 with sample SD error bars"
        ),
        use_container_width=True,
    )

    st.markdown("#### Classification confusion matrix")
    _render_rq_confusion_matrix(
        section["mean_confusion_matrix"],
        title="Classification confusion matrix (mean counts)",
    )


def _render_rq_c_tab(results: dict) -> None:
    section = results["rqc_end_to_end_pipeline"]
    st.subheader(section["title"])

        # ---- Step 1: identification inside the pipeline ----
    id_block = section.get("identification_metrics", {})
    if id_block:
        st.markdown("#### Step 1 — Identification (within the pipeline)")
        nfr_m = id_block.get("nfr", {})
        fr_m = id_block.get("fr_supplementary", {})
        cols = st.columns(3)
        cols[0].metric("Precision (NFR)", _fmt_stats(nfr_m.get("precision")))
        cols[1].metric("Recall (NFR)", _fmt_stats(nfr_m.get("recall")))
        cols[2].metric("F1 (NFR)", _fmt_stats(nfr_m.get("f1")))
        st.dataframe(
            pd.DataFrame(
                [
                    _stats_to_row("NFR", nfr_m),
                    _stats_to_row("FR (supplementary)", fr_m),
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        id_cm = id_block.get("mean_confusion_matrix", {})
        if id_cm:
            _render_rq_confusion_matrix(
                id_cm, title="Pipeline identification confusion matrix (mean counts)"
            )

    # ---- Step 2: classification inside the pipeline ----
    st.markdown("#### Step 2 — Classification (within the pipeline)")
    target_metrics = section["target_category_metrics"]
    table_rows = [
        _stats_to_row(cat.title(), target_metrics[cat]) for cat in TARGET_CATEGORIES
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        _f1_bar_chart_with_error_bars(
            target_metrics, "Pipeline F1 with sample SD error bars"
        ),
        use_container_width=True,
    )

    st.markdown("#### Pipeline classification confusion matrix")
    _render_rq_confusion_matrix(
        section["mean_confusion_matrix"],
        title="Pipeline confusion matrix (mean counts)",
    )

    st.markdown("#### Error propagation")
    error_prop = section["error_propagation"]
    keys_order = [
        ("missed_in_identification", "Missed in identification"),
        ("false_positive_count", "False positives from identification"),
        ("unclassified_after_identification", "Unclassified after identification"),
        (
            "classification_errors_after_correct_identification",
            "Classification errors after correct identification",
        ),
    ]
    cols = st.columns(len(keys_order))
    for col, (key, label) in zip(cols, keys_order):
        col.metric(label, _fmt_stats(error_prop.get(key)))

    bar_df = pd.DataFrame(
        [
            {
                "Stage": label,
                "Mean": error_prop[key]["mean"],
                "Std": error_prop[key]["std"],
            }
            for key, label in keys_order
        ]
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=bar_df["Stage"],
                y=bar_df["Mean"],
                error_y=dict(type="data", array=bar_df["Std"], visible=True),
                marker_color="#000000",
            )
        ]
    )
    for stage, mean, std in zip(bar_df["Stage"], bar_df["Mean"], bar_df["Std"]):
        fig.add_annotation(
            x=stage,
            y=mean,
            text=f"<b>{mean:.1f}</b> ± {std:.1f}",
            showarrow=False,
            xshift=22,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            borderpad=3,
        )
    fig.update_layout(
        title="Error propagation across pipeline stages",
        yaxis_title="Count (mean across runs)",
        height=380,
        font=dict(size=14),
    )
    st.plotly_chart(fig, use_container_width=True)


def _load_json_if_exists(path: Path) -> dict | None:
    """Load a JSON file if it exists; otherwise return None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

if __name__ == "__main__":
    main()