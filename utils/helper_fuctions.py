import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal
from pydantic import Field, create_model
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


def plot_bar_grid(data: dict, ncols: int = 3, figsize_per_plot: tuple = (5, 4), horizontal: bool = False,
                  show_values: bool = True, title: str = None):
    """Plot multiple bar charts in a grid."""

    if not data:
        raise ValueError("data cannot be empty.")
    if ncols < 1:
        raise ValueError("ncols must be at least 1.")

    n = len(data)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_plot[0] * ncols,
            figsize_per_plot[1] * nrows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (chart_title, values) in zip(axes.flat, data.items()):

        if isinstance(values, pd.Series):
            categories = values.index.tolist()
            counts = values.to_numpy()
        else:
            categories = list(values.keys())
            counts = list(values.values())

        if horizontal:
            bars = ax.barh(categories, counts)
            ax.invert_yaxis()
            ax.set_xlabel("Value")
            ax.bar_label(bars, padding=3) if show_values else None
        else:
            bars = ax.bar(categories, counts)
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=90)
            if show_values:
                ax.bar_label(bars, padding=3)

        ax.set_title(chart_title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.show()

    return fig, axes


CHART_DIR = Path("charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)

def plot_benchmark(
    df,
    metric,
    title,
    ylabel,
    dataset=None,
    filename=None
):
    data = df.copy()

    if dataset is not None:
        data = data[data["dataset"] == dataset]

    data = (
        data.groupby("model", as_index=False)[metric]
        .mean()
        .sort_values(metric)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    bar_width = 0.58
    values = data[metric].to_numpy()
    models = data["model"].to_list()

    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, max(values) * 1.22)

    # Rounded bars
    for i, value in enumerate(values):
        rounding = min(
            max(values) * 0.015,
            value * 0.25
        )

        bar = FancyBboxPatch(
            (i - bar_width / 2, 0),
            bar_width,
            value,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0,
            facecolor="#8CC9F0",
            mutation_aspect=1
        )
        ax.add_patch(bar)

        ax.text(
            i,
            value + max(values) * 0.03,
            f"${value:.3f}" if "cost" in metric else f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#344054",
            fontweight="medium"
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        loc="center",
        pad=20,
        color="#1D2939"
    )
    ax.set_ylabel(ylabel, color="#667085", labelpad=12)

    ax.yaxis.grid(True, color="#E9EEF5", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0, colors="#667085")

    plt.tight_layout()

    filename = filename or metric
    fig.savefig(
        CHART_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()



def create_schema(questions: dict):
    """Generate a Pydantic model from JEV questions."""

    fields = {}

    for name, question in questions.items():

        criteria = question["criteria"]

        description = (
            question["instructions"]
            + "\nCriteria:\n"
            + "\n".join(
                f"- {label}: {description}"
                for label, description in criteria.items()
            )
        )

        labels = tuple(criteria.keys())

        fields[name] = (
            Literal[labels],
            Field(description=description)
        )

    return create_model("QuestionSchema", **fields)


def create_prompt(state: dict, questions: dict):
    """Generate a chat prompt equivalent to JEV."""

    state_text = "\n".join(
        f"{key}: {value}"
        for key, value in state.items()
    )

    questions_text = "\n\n".join(
        f"{name}:\n"
        f"Instructions: {question['instructions']}\n"
        "Criteria:\n"
        + "\n".join(
            f"- {label}: {description}"
            for label, description in question["criteria"].items()
        )
        for name, question in questions.items()
    )

    return (
        "Classify the following input based on the "
        "provided instructions and criteria.\n\n"
        f"{state_text}\n\n"
        f"{questions_text}"
    )