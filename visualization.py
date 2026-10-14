from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_drift_dashboard(
    report: Dict[str, Any],
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    output_path: str = "drift_dashboard.png",
) -> str:
    """
    Renders a comprehensive, publication-grade multi-panel drift visualization dashboard.
    Panels:
    1. Overall PSI Scores by Feature (color-coded by severity with threshold lines)
    2. Distribution Shift Comparison (KDE/Hist for highest-drift continuous feature)
    3. Kolmogorov-Smirnov ECDF comparison highlighting maximum vertical distance D
    4. Quantile Binned PSI breakdown (Expected Qi vs Actual Pi)
    5. Categorical Feature Shift (Reference vs Production proportions)
    """
    summary = report["feature_drift_summary"]
    df_summary = pd.DataFrame(summary)

    fig = plt.figure(figsize=(19, 12))
    fig.patch.set_facecolor("#0f172a")  # Slate-900 dark theme
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    plt.rcParams["text.color"] = "#e2e8f0"
    plt.rcParams["axes.labelcolor"] = "#e2e8f0"
    plt.rcParams["xtick.color"] = "#94a3b8"
    plt.rcParams["ytick.color"] = "#94a3b8"
    plt.rcParams["axes.edgecolor"] = "#334155"

    severity_colors = {
        "STABLE": "#10b981",         # Emerald green
        "WARNING": "#f59e0b",        # Amber
        "CRITICAL_DRIFT": "#ef4444", # Crimson
    }

    # =========================================================================
    # PANEL 1: Feature PSI Scores Bar Chart
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1e293b")

    features = df_summary["feature_name"].tolist()
    psi_scores = df_summary["psi_score"].tolist()
    severities = [str(s) for s in df_summary["severity"].tolist()]
    bar_colors = [severity_colors.get(s, "#38bdf8") for s in severities]

    y_pos = np.arange(len(features))
    bars = ax1.barh(y_pos, psi_scores, color=bar_colors, height=0.6, edgecolor="#334155")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features, fontsize=11, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Population Stability Index (PSI)", fontsize=11)
    ax1.set_title("1. Feature PSI Scores & Alert Severity", fontsize=13, fontweight="bold", pad=12)

    # Threshold guidelines
    ax1.axvline(0.10, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.8, label="Warning (0.10)")
    ax1.axvline(0.25, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.8, label="Critical (0.25)")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9, loc="lower right")

    # Add numeric labels to bars
    for bar, score in zip(bars, psi_scores):
        ax1.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#f8fafc",
        )
    ax1.set_xlim(0, max(max(psi_scores) * 1.25, 0.35))
    ax1.grid(axis="x", color="#334155", linestyle=":", alpha=0.6)

    # =========================================================================
    # PANEL 2: Distribution Comparison for Critical Feature (e.g. annual_income)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1e293b")

    # Pick the numeric feature with highest PSI
    numeric_rows = df_summary[df_summary["feature_type"] == "numeric"]
    crit_feat = (
        numeric_rows.sort_values(by="psi_score", ascending=False).iloc[0]["feature_name"]
        if not numeric_rows.empty
        else "annual_income"
    )

    ref_vals = ref_df[crit_feat].dropna().values
    curr_vals = curr_df[crit_feat].dropna().values

    # Determine common range
    p1 = min(np.percentile(ref_vals, 0.5), np.percentile(curr_vals, 0.5))
    p99 = max(np.percentile(ref_vals, 99.5), np.percentile(curr_vals, 99.5))
    bins = np.linspace(p1, p99, 35)

    ax2.hist(ref_vals, bins=bins, density=True, alpha=0.45, color="#38bdf8", label="Reference Baseline", edgecolor="#0284c7")
    ax2.hist(curr_vals, bins=bins, density=True, alpha=0.45, color="#f43f5e", label="Production Batch", edgecolor="#e11d48")

    ax2.set_title(f"2. Distribution Shift: '{crit_feat}'", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel(crit_feat.replace("_", " ").title(), fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    ax2.grid(color="#334155", linestyle=":", alpha=0.6)

    # =========================================================================
    # PANEL 3: Kolmogorov-Smirnov ECDF with Distance D
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1e293b")

    ref_sorted = np.sort(ref_vals)
    curr_sorted = np.sort(curr_vals)
    ref_ecdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    curr_ecdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)

    ax3.plot(ref_sorted, ref_ecdf, color="#38bdf8", linewidth=2.2, label="Ref ECDF $F_{ref}(x)$")
    ax3.plot(curr_sorted, curr_ecdf, color="#f43f5e", linewidth=2.2, label="Curr ECDF $F_{curr}(x)$")

    # Find the maximum vertical divergence point for KS visualization
    all_x = np.sort(np.unique(np.concatenate([ref_sorted, curr_sorted])))
    r_cdf = np.searchsorted(ref_sorted, all_x, side="right") / len(ref_sorted)
    c_cdf = np.searchsorted(curr_sorted, all_x, side="right") / len(curr_sorted)
    diff = np.abs(r_cdf - c_cdf)
    max_idx = np.argmax(diff)
    max_x = all_x[max_idx]
    max_d = diff[max_idx]

    # Draw vertical arrow showing D
    y_ref = r_cdf[max_idx]
    y_curr = c_cdf[max_idx]
    ax3.vlines(x=max_x, ymin=min(y_ref, y_curr), ymax=max(y_ref, y_curr), color="#fbbf24", linewidth=2.5, linestyle="-")
    ax3.plot([max_x], [y_ref], "o", color="#38bdf8", markersize=6)
    ax3.plot([max_x], [y_curr], "o", color="#f43f5e", markersize=6)
    ax3.annotate(
        f"KS $D$ = {max_d:.3f}",
        xy=(max_x, (y_ref + y_curr) / 2),
        xytext=(max_x * 0.75, (y_ref + y_curr) / 2 + 0.1),
        arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.5),
        fontsize=10,
        fontweight="bold",
        color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor="#fbbf24", alpha=0.9),
    )

    ax3.set_title("3. Two-Sample KS Test (ECDF Divergence)", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel(crit_feat.replace("_", " ").title(), fontsize=11)
    ax3.set_ylabel("Cumulative Probability", fontsize=11)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9, loc="lower right")
    ax3.grid(color="#334155", linestyle=":", alpha=0.6)

    # =========================================================================
    # PANEL 4: Quantile-Bin PSI Breakdown (Expected Qi vs Actual Pi)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#1e293b")

    num_bins = 10
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.unique(np.percentile(ref_vals, quantiles))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_vals, bins=bin_edges)

    q_props = ref_counts / len(ref_vals)
    p_props = curr_counts / len(curr_vals)
    bin_psi = (p_props - q_props) * np.log(np.where(p_props == 0, 1e-4, p_props) / np.where(q_props == 0, 1e-4, q_props))

    indices = np.arange(len(q_props))
    width = 0.35

    ax4.bar(indices - width / 2, q_props * 100, width=width, label="Reference Expected % ($Q_i$)", color="#38bdf8", edgecolor="#0284c7")
    ax4.bar(indices + width / 2, p_props * 100, width=width, label="Production Actual % ($P_i$)", color="#f43f5e", edgecolor="#e11d48")

    # Secondary axis for bin PSI contribution
    ax4_psi = ax4.twinx()
    ax4_psi.plot(indices, bin_psi, color="#fbbf24", marker="o", linewidth=2.0, label="Bin PSI Contribution")
    ax4_psi.set_ylabel("Bin PSI Contribution: $(P_i - Q_i) \\ln(P_i / Q_i)$", color="#fbbf24", fontsize=10)
    ax4_psi.tick_params(axis="y", labelcolor="#fbbf24")

    ax4.set_xticks(indices)
    ax4.set_xticklabels([f"Bin {i+1}\n({quantiles[i]:.0f}%-{quantiles[i+1]:.0f}%)" for i in range(len(q_props))], fontsize=9)
    ax4.set_xlabel("Reference Quantile Buckets", fontsize=11)
    ax4.set_ylabel("Population Share (%)", fontsize=11)
    ax4.set_title(f"4. Quantile-Binned PSI Decomposition for '{crit_feat}'", fontsize=13, fontweight="bold", pad=12)

    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_psi.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, facecolor="#0f172a", edgecolor="#334155", fontsize=9, loc="upper left")
    ax4.grid(axis="y", color="#334155", linestyle=":", alpha=0.6)

    # =========================================================================
    # PANEL 5: Categorical Distribution Shift
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1e293b")

    cat_rows = df_summary[df_summary["feature_type"] == "categorical"]
    cat_feat = cat_rows.iloc[0]["feature_name"] if not cat_rows.empty else "subscription_tier"

    cats = sorted(set(ref_df[cat_feat].dropna().unique()).union(set(curr_df[cat_feat].dropna().unique())))
    ref_cat_props = ref_df[cat_feat].value_counts(normalize=True).reindex(cats, fill_value=0.0).values * 100
    curr_cat_props = curr_df[cat_feat].value_counts(normalize=True).reindex(cats, fill_value=0.0).values * 100

    cat_indices = np.arange(len(cats))
    c_width = 0.35
    ax5.bar(cat_indices - c_width / 2, ref_cat_props, width=c_width, label="Reference %", color="#38bdf8", edgecolor="#0284c7")
    ax5.bar(cat_indices + c_width / 2, curr_cat_props, width=c_width, label="Production %", color="#f43f5e", edgecolor="#e11d48")

    ax5.set_xticks(cat_indices)
    ax5.set_xticklabels([c.title() for c in cats], fontsize=10, fontweight="bold")
    ax5.set_xlabel(cat_feat.replace("_", " ").title(), fontsize=11)
    ax5.set_ylabel("Share (%)", fontsize=11)
    ax5.set_title(f"5. Categorical Shift: '{cat_feat}'", fontsize=13, fontweight="bold", pad=12)
    ax5.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9)
    ax5.grid(axis="y", color="#334155", linestyle=":", alpha=0.6)

    # Main Super Title
    retrain_str = "RETRAINING TRIGGERED: CRITICAL DRIFT" if report["retraining_triggered"] else "SYSTEM STABLE: NO RETRAINING NEEDED"
    title_color = "#ef4444" if report["retraining_triggered"] else "#10b981"
    fig.suptitle(
        f"Data Pipeline Statistical Drift Report | {retrain_str} | Valid: {report['valid_record_count']} | Quarantined: {report['quarantined_record_count']}",
        fontsize=16,
        fontweight="bold",
        color=title_color,
        y=0.975,
    )
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.06, right=0.94)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    return output_path



def generate_html_report(
    report: Dict[str, Any],
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    output_path: str = "drift_report.html",
) -> str:
    """
    Generates an interactive, modern, standalone HTML monitoring dashboard
    incorporating metric cards, interactive charts, and DLQ quarantine logs.
    """
    summary = report["feature_drift_summary"]
    dlq_samples = report.get("sample_dlq_errors", [])
    retrain_flag = report["retraining_triggered"]

    status_badge = (
        '<span style="background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; padding: 4px 12px; border-radius: 9999px; font-weight: bold;">CRITICAL DRIFT DETECTED - RETRAINING TRIGGERED</span>'
        if retrain_flag
        else '<span style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; padding: 4px 12px; border-radius: 9999px; font-weight: bold;">ALL FEATURES STABLE</span>'
    )

    table_rows = ""
    for r in summary:
        sev = str(r["severity"])
        if sev == "CRITICAL_DRIFT":
            badge = '<span style="background:#ef4444; color:#fff; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;">CRITICAL</span>'
        elif sev == "WARNING":
            badge = '<span style="background:#f59e0b; color:#fff; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;">WARNING</span>'
        else:
            badge = '<span style="background:#10b981; color:#fff; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;">STABLE</span>'

        ks_stat = f"{r['ks_statistic']:.4f}" if r["ks_statistic"] is not None else "—"
        ks_p = f"{r['ks_p_value']:.6f}" if r["ks_p_value"] is not None else "—"
        psi = f"{r['psi_score']:.4f}"

        table_rows += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 12px; font-weight: bold; color: #f8fafc;">{r['feature_name']}</td>
            <td style="padding: 12px; color: #94a3b8;">{r['feature_type']}</td>
            <td style="padding: 12px; font-family: monospace;">{ks_stat}</td>
            <td style="padding: 12px; font-family: monospace;">{ks_p}</td>
            <td style="padding: 12px; font-family: monospace; font-weight: bold; color: {'#ef4444' if r['psi_score'] >= 0.25 else ('#f59e0b' if r['psi_score'] >= 0.10 else '#10b981')};">{psi}</td>
            <td style="padding: 12px;">{badge}</td>
            <td style="padding: 12px; color: {'#ef4444' if r['drift_detected'] else '#10b981'}; font-weight: bold;">{'YES' if r['drift_detected'] else 'NO'}</td>
        </tr>
        """

    dlq_html = ""
    if dlq_samples:
        for idx, item in enumerate(dlq_samples):
            payload_str = json.dumps(item.get("raw_payload", {}), indent=2, default=str)
            errors_str = json.dumps(item.get("error_details", []), indent=2, default=str)
            dlq_html += f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                <div style="font-weight: bold; color: #f43f5e; margin-bottom: 6px;">Quarantined Record #{idx+1} (Isolated by Schema Gatekeeper)</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Raw Payload</div>
                        <pre style="background: #0f172a; padding: 10px; border-radius: 6px; font-size: 12px; color: #cbd5e1; overflow-x: auto;">{payload_str}</pre>
                    </div>
                    <div>
                        <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Pydantic Core Validation Errors</div>
                        <pre style="background: #0f172a; padding: 10px; border-radius: 6px; font-size: 12px; color: #fda4af; overflow-x: auto;">{errors_str}</pre>
                    </div>
                </div>
            </div>
            """
    else:
        dlq_html = '<p style="color: #10b981;">No records quarantined. 100% valid batch.</p>'

    feature_labels = [r["feature_name"] for r in summary]
    feature_psi = [r["psi_score"] for r in summary]
    feature_colors = [
        ("#ef4444" if r["psi_score"] >= 0.25 else ("#f59e0b" if r["psi_score"] >= 0.10 else "#10b981"))
        for r in summary
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Health & Drift Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            background-color: #0b1120;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .card-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 6px; }}
        .card-value {{ font-size: 24px; font-weight: bold; color: #f8fafc; }}
        .section {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 22px;
            margin-bottom: 26px;
        }}
        .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 16px; color: #f8fafc; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #0f172a; padding: 12px; color: #94a3b8; font-size: 12px; text-transform: uppercase; border-bottom: 2px solid #334155; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 26px; color: #f8fafc;">Data Drift & Pipeline Diagnostic Report</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">Timestamp: {report['batch_timestamp']}</p>
            </div>
            <div>{status_badge}</div>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-label">Valid Records</div>
                <div class="card-value" style="color: #38bdf8;">{report['valid_record_count']}</div>
            </div>
            <div class="card">
                <div class="card-label">Quarantined (DLQ)</div>
                <div class="card-value" style="color: {'#ef4444' if report['quarantined_record_count'] > 0 else '#10b981'};">{report['quarantined_record_count']}</div>
            </div>
            <div class="card">
                <div class="card-label">Retraining Triggered</div>
                <div class="card-value" style="color: {'#ef4444' if retrain_flag else '#10b981'};">{'YES' if retrain_flag else 'NO'}</div>
            </div>
            <div class="card">
                <div class="card-label">Monitored Features</div>
                <div class="card-value">{len(summary)}</div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Feature Population Stability Index (PSI) Overview</div>
            <div style="height: 280px;">
                <canvas id="psiChart"></canvas>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Statistical Feature Drift Summary</div>
            <table>
                <thead>
                    <tr>
                        <th>Feature Name</th>
                        <th>Type</th>
                        <th>KS Statistic (D)</th>
                        <th>KS p-value</th>
                        <th>PSI Score</th>
                        <th>Severity</th>
                        <th>Drift Detected</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="section-title">Dead-Letter Queue (DLQ) Sample Payloads</div>
            {dlq_html}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('psiChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(feature_labels)},
                datasets: [{{
                    label: 'Population Stability Index (PSI)',
                    data: {json.dumps(feature_psi)},
                    backgroundColor: {json.dumps(feature_colors)},
                    borderRadius: 4,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#e2e8f0', font: {{ weight: 'bold' }} }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return ' PSI: ' + context.raw.toFixed(4);
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
