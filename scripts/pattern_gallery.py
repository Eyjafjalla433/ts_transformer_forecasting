import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SeriesSpec = Tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]


def build_seasonality(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = 1.9 * np.sin(2 * np.pi * x / 60.0)
    harmonic = 0.7 * np.sin(2 * np.pi * x / 13.0 + 0.8)
    noise = rng.normal(0.0, 0.35, size=x.size)
    return 18.0 + base + harmonic + noise


def build_trend(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    slope = 0.055 * x
    curve = 3.0 * np.tanh((x - x.size * 0.35) / 35.0)
    noise = rng.normal(0.0, 0.28, size=x.size).cumsum() * 0.05
    return 8.0 + slope + curve + noise


def build_shifting(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left = 8.5 + 1.3 * np.sin(2 * np.pi * x[: x.size // 2] / 55.0)
    right = 3.2 + 0.5 * np.sin(2 * np.pi * x[x.size // 2 :] / 18.0)
    series = np.concatenate([left, right])
    series += rng.normal(0.0, 0.7, size=x.size)
    return series


def build_non_seasonality(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    random_walk = rng.normal(0.0, 0.55, size=x.size).cumsum()
    drift = np.interp(x, [0, x.size * 0.4, x.size], [0.0, 7.0, 2.0])
    return 6.0 + drift + random_walk * 0.35


def build_non_trend(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    baseline = np.full_like(x, 7.5, dtype=float)
    local_noise = rng.normal(0.0, 0.5, size=x.size)
    spikes = np.zeros_like(x, dtype=float)
    spike_positions = rng.choice(x.size, size=max(6, x.size // 30), replace=False)
    spikes[spike_positions] = rng.normal(0.0, 2.6, size=spike_positions.size)
    return baseline + local_noise + spikes


def build_non_shifting(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    envelope = 1.1 + 0.35 * np.sin(2 * np.pi * x / 85.0)
    carrier = 0.9 * np.sin(2 * np.pi * x / 9.0)
    noise = rng.normal(0.0, 0.28, size=x.size)
    return 9.0 + envelope + carrier + noise


def get_series_specs() -> List[SeriesSpec]:
    return [
        ("(a) Seasonality", build_seasonality),
        ("(b) Trend", build_trend),
        ("(c) Shifting", build_shifting),
        ("(d) Non-Seasonality", build_non_seasonality),
        ("(e) Non-Trend", build_non_trend),
        ("(f) Non-Shifting", build_non_shifting),
    ]


def render_pattern_gallery(
    series_length: int,
    seed: int,
    output_path: Path,
    dpi: int,
    show: bool,
) -> Path:
    rng = np.random.default_rng(seed)
    x = np.arange(series_length, dtype=float)
    specs = get_series_specs()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax, (title, builder) in zip(axes, specs):
        y = builder(x, rng)
        ax.plot(x, y, color="#4C9BE8", linewidth=0.9, alpha=0.95)
        ax.set_title(title, fontsize=12, y=-0.28, fontweight="semibold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.22)
        ax.tick_params(labelsize=12)
        ax.margins(x=0.01)

    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=1.1, h_pad=2.1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic gallery that illustrates common time-series patterns."
    )
    parser.add_argument(
        "--series-length",
        type=int,
        default=360,
        help="Number of points in each synthetic series.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible synthetic data.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("outputs") / "pattern_gallery.png"),
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in an interactive window after saving.",
    )
    args = parser.parse_args()
    if args.series_length <= 20:
        raise ValueError("series-length must be greater than 20, got {}".format(args.series_length))
    if args.dpi <= 0:
        raise ValueError("dpi must be positive, got {}".format(args.dpi))
    return args


def main() -> None:
    args = parse_args()
    output_path = render_pattern_gallery(
        series_length=args.series_length,
        seed=args.seed,
        output_path=Path(args.output),
        dpi=args.dpi,
        show=args.show,
    )
    print("Saved pattern gallery to {}".format(output_path.resolve()))


if __name__ == "__main__":
    main()
