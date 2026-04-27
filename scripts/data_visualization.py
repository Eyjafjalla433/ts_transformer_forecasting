import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


TFB_REQUIRED_COLUMNS = {"date", "data", "cols"}


def list_csv_files(input_dirs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for directory in input_dirs:
        root = Path(directory)
        if not root.exists():
            continue
        files.extend(sorted(root.glob("*.csv")))
    return files


def infer_is_tfb_long_format(df: pd.DataFrame) -> bool:
    return TFB_REQUIRED_COLUMNS.issubset(set(df.columns))


def choose_columns(columns: Iterable[str], max_columns: int) -> List[str]:
    selected: List[str] = []
    for column in columns:
        selected.append(str(column))
        if len(selected) >= max_columns:
            break
    return selected


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def build_output_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / "{}_sampled_trend.png".format(sanitize_name(csv_path.stem))


def sample_every_n(df: pd.DataFrame, step: int) -> pd.DataFrame:
    if step <= 0:
        raise ValueError("sample_step must be positive, got {}".format(step))
    return df.iloc[::step].reset_index(drop=True)


def resolve_single_series(
    available_series: Sequence[str],
    requested_series: Optional[str],
) -> str:
    if not available_series:
        raise ValueError("No available series to visualize.")
    if requested_series:
        if requested_series not in available_series:
            raise ValueError(
                "Requested series '{}' not found. Available series: {}".format(
                    requested_series, list(available_series)
                )
            )
        return requested_series
    return str(available_series[0])


def plot_tfb_long_csv(
    df: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
    max_points: int,
    sample_step: int,
    series_name: Optional[str],
    dpi: int,
) -> str:
    trimmed = df.loc[:, ["date", "data", "cols"]].copy()
    trimmed["cols"] = trimmed["cols"].astype(str)
    trimmed["date"] = pd.to_numeric(trimmed["date"], errors="coerce").fillna(trimmed["date"])
    available_series = list(dict.fromkeys(trimmed["cols"].astype(str).tolist()))
    selected_series = resolve_single_series(available_series, series_name)
    trimmed = trimmed[trimmed["cols"].astype(str) == selected_series]

    unique_dates = list(dict.fromkeys(trimmed["date"].tolist()))[:max_points]
    trimmed = trimmed[trimmed["date"].isin(unique_dates)]

    plot_df = trimmed.pivot_table(
        index="date",
        columns="cols",
        values="data",
        aggfunc="last",
    ).sort_index()

    if plot_df.empty:
        raise ValueError("No plottable rows found in long-format CSV: {}".format(csv_path))

    plot_df = sample_every_n(plot_df.reset_index(), sample_step)

    plt.figure(figsize=(14, 7))
    x_axis = range(len(plot_df))
    plt.plot(
        x_axis,
        plot_df[selected_series],
        linewidth=1.8,
        color="#1f77b4",
        label=selected_series,
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return selected_series


def plot_wide_csv(
    df: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
    max_points: int,
    sample_step: int,
    series_name: Optional[str],
    dpi: int,
) -> str:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError("No numeric columns found in wide CSV: {}".format(csv_path))

    selected_column = resolve_single_series(
        [str(column) for column in numeric_df.columns],
        series_name,
    )
    plot_df = numeric_df.loc[: max_points - 1, [selected_column]].copy()
    plot_df = plot_df.dropna(how="all")

    if plot_df.empty:
        raise ValueError("No plottable rows found in wide CSV: {}".format(csv_path))

    plot_df = sample_every_n(plot_df.reset_index(drop=True), sample_step)

    plt.figure(figsize=(14, 7))
    x_axis = range(len(plot_df))
    plt.plot(
        x_axis,
        plot_df[selected_column],
        linewidth=1.8,
        color="#1f77b4",
        label=selected_column,
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return selected_column


def visualize_csv(
    csv_path: Path,
    output_dir: Path,
    max_points: int,
    sample_step: int,
    series_name: Optional[str],
    dpi: int,
) -> Optional[Tuple[Path, str]]:
    df = pd.read_csv(csv_path)
    output_path = build_output_path(output_dir, csv_path)

    if infer_is_tfb_long_format(df):
        selected_series = plot_tfb_long_csv(
            df=df,
            csv_path=csv_path,
            output_path=output_path,
            max_points=max_points,
            sample_step=sample_step,
            series_name=series_name,
            dpi=dpi,
        )
    else:
        selected_series = plot_wide_csv(
            df=df,
            csv_path=csv_path,
            output_path=output_path,
            max_points=max_points,
            sample_step=sample_step,
            series_name=series_name,
            dpi=dpi,
        )

    return output_path, selected_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a single sampled series from each CSV dataset to highlight trend patterns."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/dataset"],
        help="Directories that contain CSV files to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/data_visualization",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum number of raw time points or rows to read per CSV.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=5,
        help="Plot every n-th point to make the trend clearer.",
    )
    parser.add_argument(
        "--series-name",
        default=None,
        help="Optional single series/column name to visualize when it exists in a CSV.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    args = parser.parse_args()

    csv_files = list_csv_files(args.input_dirs)
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found under directories: {}".format(", ".join(args.input_dirs))
        )

    output_dir = Path(args.output_dir)
    print("Found CSV files:", len(csv_files))
    print("Saving figures to:", output_dir)

    saved_paths: List[Path] = []
    for csv_path in csv_files:
        try:
            result = visualize_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                max_points=args.max_points,
                sample_step=args.sample_step,
                series_name=args.series_name,
                dpi=args.dpi,
            )
            if result is not None:
                output_path, selected_series = result
                saved_paths.append(output_path)
                print("Saved: {} | series: {}".format(output_path, selected_series))
        except Exception as exc:
            print("Skipped {}: {}".format(csv_path.name, exc))

    print("Completed. Generated {} figures.".format(len(saved_paths)))


if __name__ == "__main__":
    main()
