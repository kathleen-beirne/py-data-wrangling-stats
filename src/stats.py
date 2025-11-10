from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

REPORTS = Path("reports")
FIGS = REPORTS / "figures"
REPORTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)

def descriptives(df: pd.DataFrame, report_path: str = "reports/summary.md") -> str:
    """Compute basic descriptives and save a small markdown report."""
    lines = []
    lines.append("# Descriptive statistics\n")
    lines.append(f"Rows: {len(df)}\n")

    # by group means
    if "group" in df.columns:
        desc = df.groupby("group")[["age","score1","score2","rt_ms","score1_z"]].mean(numeric_only=True)
        lines.append("## Means by group\n")
        lines.append(desc.round(2).to_markdown())
        lines.append("\n")

    # Distribution plot
    if "score1" in df.columns:
        plt.figure()
        df["score1"].hist(bins=30)
        plt.title("Score1 distribution")
        fig_path = FIGS / "score1_hist.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

        lines.append(f"![score1 hist]({fig_path.as_posix()})\n")

    Path(report_path).write_text("\n".join(lines))
    return report_path

def glm(df: pd.DataFrame, report_path: str = "reports/summary.md") -> str:
    """
    Fit a simple GLM/OLS: score1 ~ age + C(group)
    Append the summary table to the report.
    """
    if not {"score1","age","group"}.issubset(df.columns):
        raise ValueError("Dataframe missing required columns for GLM.")
    model = smf.ols("score1 ~ age + C(group)", data=df).fit()
    summary_txt = model.summary().as_text()

    existing = Path(report_path).read_text() if Path(report_path).exists() else ""
    header = "\n# GLM: score1 ~ age + C(group)\n"
    Path(report_path).write_text(existing + header + "```\n" + summary_txt + "\n```\n")
    return report_path
