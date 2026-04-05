"""
Create preliminary trade-frequency and trade-volume plots from processed data.

Input:
  data/processed/<HS6>_<YEAR>.csv

Outputs:
  figures/preliminary/
    heatmaps/
    exporters/frequencies/
    exporters/volumes/
    importers/frequencies/
    importers/volumes/
    heatmaps.html

Plot set
--------
For each commodity:
1. Heatmap: yearly trade frequency by exporter (max 12 months)
2. Heatmap: yearly trade volume by importer
3. Heatmap: yearly trade volume by exporter
4. Bar plot: top 10 exporters by average yearly trade frequency
5. Bar plot: top 10 exporters by average monthly trade volume
6. Bar plot: top 10 importers by average yearly trade frequency
7. Bar plot: top 10 importers by average monthly trade volume

By default, the script plots all 11 commodities so the heatmap total is 33.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures" / "preliminary"
HEATMAP_DIR = FIGURES_DIR / "heatmaps"
EXPORTER_DIR = FIGURES_DIR / "exporters"
IMPORTER_DIR = FIGURES_DIR / "importers"
EXPORTER_FREQUENCY_DIR = EXPORTER_DIR / "frequencies"
EXPORTER_VOLUME_DIR = EXPORTER_DIR / "volumes"
IMPORTER_FREQUENCY_DIR = IMPORTER_DIR / "frequencies"
IMPORTER_VOLUME_DIR = IMPORTER_DIR / "volumes"
HEATMAP_HTML_PATH = FIGURES_DIR / "heatmaps.html"

HS6_LABELS = {
    "080221": "Hazelnuts in shell",
    "080222": "Hazelnuts shelled",
    "080211": "Almonds in shell",
    "080212": "Almonds shelled",
    "080231": "Walnuts in shell",
    "080232": "Walnuts shelled",
    "080251": "Pistachios in shell",
    "080252": "Pistachios shelled",
    "120241": "Peanuts in shell",
    "120242": "Peanuts shelled",
    "120740": "Sesame seeds",
}


def ensure_dirs() -> None:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTER_FREQUENCY_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTER_VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTER_FREQUENCY_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTER_VOLUME_DIR.mkdir(parents=True, exist_ok=True)


def discover_target_codes() -> list[str]:
    codes = sorted({path.stem.split("_")[0] for path in PROCESSED_DIR.glob("*.csv")})
    if not codes:
        raise FileNotFoundError(f"No processed CSV files found in {PROCESSED_DIR}")

    target_codes = codes
    log.info("Using %s commodity codes for plotting: %s", len(target_codes), ", ".join(target_codes))
    return target_codes


def load_processed_data(target_codes: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for code in target_codes:
        files = sorted(PROCESSED_DIR.glob(f"{code}_*.csv"))
        for path in files:
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["cmdCode"] = df["cmdCode"].astype(str).str.zfill(6)
            frames.append(df)

    if not frames:
        raise ValueError("Processed files were found, but none contained usable rows.")

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data["cmdCode"].isin(target_codes)].copy()
    data["year_month"] = data["year_month"].astype(str)
    data["year"] = data["year_month"].str[:4].astype(int)
    data["month"] = data["year_month"].str[-2:].astype(int)
    data["primaryValue"] = pd.to_numeric(data["primaryValue"], errors="coerce")
    data["NetWeight"] = pd.to_numeric(data["NetWeight"], errors="coerce")
    data["UnitValue"] = pd.to_numeric(data["UnitValue"], errors="coerce")
    data["LogUnitValue"] = pd.to_numeric(data["LogUnitValue"], errors="coerce")
    return data


def get_label(code: str) -> str:
    return HS6_LABELS.get(code, code)


def rotate_x_labels(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelrotation=90)


def build_plotly_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    colorscale: str,
    value_label: str,
    zmax: float | None = None,
) -> go.Figure:
    z_values = pivot_df.to_numpy()
    x_values = [str(col) for col in pivot_df.columns]
    y_values = [str(idx) for idx in pivot_df.index]

    customdata = []
    for year in y_values:
        customdata.append([[country, year] for country in x_values])

    heatmap = go.Heatmap(
        z=z_values,
        x=x_values,
        y=y_values,
        colorscale=colorscale,
        colorbar={"title": value_label},
        customdata=customdata,
        hovertemplate=(
            "Country: %{customdata[0]}<br>"
            "Year: %{customdata[1]}<br>"
            f"{value_label}: " + "%{z}<extra></extra>"
        ),
        zmin=0,
        zmax=zmax,
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Year",
        xaxis={"showticklabels": False},
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 80, "b": 40},
        height=650,
        width=1200,
    )
    return fig


def save_heatmap_html(figures: list[tuple[str, go.Figure]]) -> None:
    sections: list[str] = []
    for title, fig in figures:
        figure_html = to_html(fig, include_plotlyjs=False, full_html=False)
        sections.append(
            "\n".join(
                [
                    '<section class="heatmap-section">',
                    f"<h2>{title}</h2>",
                    figure_html,
                    "</section>",
                ]
            )
        )

    html = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Preliminary Heatmaps</title>",
            '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }",
            "    h1 { margin-bottom: 8px; }",
            "    p { margin-top: 0; color: #444; }",
            "    .heatmap-section { margin-bottom: 36px; padding: 18px; background: white; border: 1px solid #ddd; }",
            "    .heatmap-section h2 { margin-top: 0; font-size: 18px; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Preliminary Heatmaps</h1>",
            "  <p>Country labels are hidden on the x-axis and available on hover.</p>",
            *sections,
            "</body>",
            "</html>",
        ]
    )

    HEATMAP_HTML_PATH.write_text(html, encoding="utf-8")


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
) -> None:
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df, x=x_col, y=y_col, color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    rotate_x_labels(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def create_heatmaps(df: pd.DataFrame, code: str) -> list[tuple[str, go.Figure]]:
    product_df = df.loc[df["cmdCode"] == code].copy()
    label = get_label(code)

    exporter_frequency = (
        product_df.groupby(["year", "reporterISO"])["month"]
        .nunique()
        .reset_index(name="trade_frequency")
        .pivot(index="year", columns="reporterISO", values="trade_frequency")
        .fillna(0)
        .sort_index()
    )

    importer_volume = (
        product_df.groupby(["year", "partnerISO"])["primaryValue"]
        .sum()
        .reset_index(name="trade_volume")
        .pivot(index="year", columns="partnerISO", values="trade_volume")
        .fillna(0)
        .sort_index()
    )

    exporter_volume = (
        product_df.groupby(["year", "reporterISO"])["primaryValue"]
        .sum()
        .reset_index(name="trade_volume")
        .pivot(index="year", columns="reporterISO", values="trade_volume")
        .fillna(0)
        .sort_index()
    )

    frequency_title = f"{code} | {label} | Yearly Trade Frequency by Exporter"
    importer_title = f"{code} | {label} | Yearly Trade Volume by Importer"
    exporter_title = f"{code} | {label} | Yearly Trade Volume by Exporter"

    figures = [
        (
            frequency_title,
            build_plotly_heatmap(
                exporter_frequency,
                frequency_title,
                colorscale="Blues",
                value_label="Trade frequency",
                zmax=12,
            ),
        ),
        (
            importer_title,
            build_plotly_heatmap(
                importer_volume,
                importer_title,
                colorscale="Greens",
                value_label="Trade volume",
            ),
        ),
        (
            exporter_title,
            build_plotly_heatmap(
                exporter_volume,
                exporter_title,
                colorscale="Oranges",
                value_label="Trade volume",
            ),
        ),
    ]

    return figures


def create_exporter_barplots(df: pd.DataFrame, code: str) -> None:
    product_df = df.loc[df["cmdCode"] == code].copy()
    label = get_label(code)

    exporter_year_freq = (
        product_df.groupby(["reporterISO", "year"])["month"]
        .nunique()
        .reset_index(name="yearly_frequency")
    )
    exporter_freq_top10 = (
        exporter_year_freq.groupby("reporterISO")["yearly_frequency"]
        .mean()
        .reset_index(name="avg_yearly_frequency")
        .sort_values("avg_yearly_frequency", ascending=False)
        .head(10)
    )

    exporter_monthly_volume = (
        product_df.groupby(["reporterISO", "year_month"])["primaryValue"]
        .sum()
        .reset_index(name="monthly_volume")
    )
    exporter_volume_top10 = (
        exporter_monthly_volume.groupby("reporterISO")["monthly_volume"]
        .mean()
        .reset_index(name="avg_monthly_trade_volume")
        .sort_values("avg_monthly_trade_volume", ascending=False)
        .head(10)
    )

    plot_bar(
        exporter_freq_top10,
        "reporterISO",
        "avg_yearly_frequency",
        f"{code} | {label} | Top 10 Exporters by Average Yearly Frequency",
        "Average yearly frequency",
        EXPORTER_FREQUENCY_DIR / f"{code}_top10_exporters_avg_yearly_frequency.png",
        color="#4C78A8",
    )
    plot_bar(
        exporter_volume_top10,
        "reporterISO",
        "avg_monthly_trade_volume",
        f"{code} | {label} | Top 10 Exporters by Average Monthly Trade Volume",
        "Average monthly trade volume",
        EXPORTER_VOLUME_DIR / f"{code}_top10_exporters_avg_monthly_volume.png",
        color="#F58518",
    )


def create_importer_barplots(df: pd.DataFrame, code: str) -> None:
    product_df = df.loc[df["cmdCode"] == code].copy()
    label = get_label(code)

    importer_year_freq = (
        product_df.groupby(["partnerISO", "year"])["month"]
        .nunique()
        .reset_index(name="yearly_frequency")
    )
    importer_freq_top10 = (
        importer_year_freq.groupby("partnerISO")["yearly_frequency"]
        .mean()
        .reset_index(name="avg_yearly_frequency")
        .sort_values("avg_yearly_frequency", ascending=False)
        .head(10)
    )

    importer_monthly_volume = (
        product_df.groupby(["partnerISO", "year_month"])["primaryValue"]
        .sum()
        .reset_index(name="monthly_volume")
    )
    importer_volume_top10 = (
        importer_monthly_volume.groupby("partnerISO")["monthly_volume"]
        .mean()
        .reset_index(name="avg_monthly_trade_volume")
        .sort_values("avg_monthly_trade_volume", ascending=False)
        .head(10)
    )

    plot_bar(
        importer_freq_top10,
        "partnerISO",
        "avg_yearly_frequency",
        f"{code} | {label} | Top 10 Importers by Average Yearly Frequency",
        "Average yearly frequency",
        IMPORTER_FREQUENCY_DIR / f"{code}_top10_importers_avg_yearly_frequency.png",
        color="#54A24B",
    )
    plot_bar(
        importer_volume_top10,
        "partnerISO",
        "avg_monthly_trade_volume",
        f"{code} | {label} | Top 10 Importers by Average Monthly Trade Volume",
        "Average monthly trade volume",
        IMPORTER_VOLUME_DIR / f"{code}_top10_importers_avg_monthly_volume.png",
        color="#E45756",
    )


def main() -> None:
    ensure_dirs()
    target_codes = discover_target_codes()
    df = load_processed_data(target_codes)

    sns.set_theme(style="whitegrid")
    heatmap_figures: list[tuple[str, go.Figure]] = []

    for code in target_codes:
        log.info("Creating plots for %s", code)
        heatmap_figures.extend(create_heatmaps(df, code))
        #create_exporter_barplots(df, code)
        #create_importer_barplots(df, code)

    save_heatmap_html(heatmap_figures)
    log.info("All preliminary figures saved under %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
