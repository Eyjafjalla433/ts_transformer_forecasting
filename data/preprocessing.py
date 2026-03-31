from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


REQUIRED_TFB_COLUMNS = ("date", "data", "cols")


def _coerce_sort_key(date_series: pd.Series) -> pd.Series:
    """Convert the TFB date column into a sortable numeric or datetime key."""
    numeric = pd.to_numeric(date_series, errors="coerce")
    if numeric.notna().all():
        return numeric

    datetime_values = pd.to_datetime(date_series, errors="coerce")
    if datetime_values.isna().any():
        bad_values = date_series[datetime_values.isna()].head(5).tolist()
        raise ValueError(
            "Date column must be numeric or pandas-compatible datetime. "
            "Invalid examples: {}".format(bad_values)
        )
    return datetime_values


def _move_target_columns_to_end(table: pd.DataFrame, target_cols: Optional[Sequence[str]]) -> pd.DataFrame:
    if not target_cols:
        return table

    target_cols = [str(col) for col in target_cols]
    missing_targets = [col for col in target_cols if col not in table.columns]
    if missing_targets:
        raise ValueError("Target columns not found after pivot: {}".format(missing_targets))

    non_targets = [col for col in table.columns if col not in target_cols]
    return table.loc[:, non_targets + list(target_cols)]


def _handle_missing_values(table: pd.DataFrame, fill_method: str) -> pd.DataFrame:
    if not table.isna().values.any():
        return table

    if fill_method == "ffill":
        table = table.ffill().dropna()
    elif fill_method == "drop":
        table = table.dropna()
    else:
        raise ValueError("Unsupported fill_method: {}".format(fill_method))

    if table.empty:
        raise ValueError("All rows were removed during missing-value handling.")
    if table.isna().values.any():
        raise ValueError("Missing values remain after applying fill_method='{}'.".format(fill_method))
    return table


def load_tfb_csv(
    path: str,
    date_col: str = "date",
    value_col: str = "data",
    var_col: str = "cols",
) -> pd.DataFrame:
    """Load a TFB-format CSV and normalize its core columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError("TFB CSV not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    required = [date_col, value_col, var_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV must contain columns {}. Missing: {}".format(required, missing)
        )

    df = df.loc[:, required].copy()
    df[var_col] = df[var_col].astype(str)
    df["_sort_key"] = _coerce_sort_key(df[date_col])

    numeric_values = pd.to_numeric(df[value_col], errors="coerce")
    if numeric_values.isna().any():
        bad_rows = df.loc[numeric_values.isna(), [date_col, var_col]].head(5).to_dict("records")
        raise ValueError(
            "Value column must be numeric. Invalid examples: {}".format(bad_rows)
        )
    df[value_col] = numeric_values

    return df.sort_values(["_sort_key", var_col], kind="stable").reset_index(drop=True)


def tfb_to_wide_table(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "data",
    var_col: str = "cols",
    fill_method: str = "ffill",
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convert a TFB three-column table into a numeric [T, F] wide table."""
    variable_order = list(dict.fromkeys(df[var_col].tolist()))

    wide_table = df.pivot_table(
        index="_sort_key",
        columns=var_col,
        values=value_col,
        aggfunc="last",
    )
    wide_table = wide_table.reindex(columns=variable_order)
    wide_table = wide_table.sort_index(kind="stable")
    wide_table = _handle_missing_values(wide_table, fill_method=fill_method)
    wide_table = _move_target_columns_to_end(wide_table, target_cols=target_cols)

    wide_table.columns.name = None
    return wide_table.reset_index(drop=True)


def extract_last_input_window(table: pd.DataFrame, input_length: int) -> pd.DataFrame:
    """Return the last window used by inference under the current project setup."""
    if input_length <= 0:
        raise ValueError("input_length must be positive.")
    if len(table) < input_length:
        raise ValueError(
            "Processed table has {} rows but input_length is {}.".format(len(table), input_length)
        )
    return table.tail(input_length).reset_index(drop=True)


def split_wide_table_for_future(
    wide_table: pd.DataFrame,
    future_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split [T, F] table into train part and future holdout."""
    if not 0.0 < future_ratio < 1.0:
        raise ValueError("future_ratio must be between 0 and 1.")

    total_rows = len(wide_table)
    if total_rows < 2:
        raise ValueError("Need at least 2 rows to split train/future.")

    future_rows = max(1, int(total_rows * future_ratio))
    split_idx = total_rows - future_rows
    split_idx = min(max(split_idx, 1), total_rows - 1)

    train_table = wide_table.iloc[:split_idx].reset_index(drop=True)
    future_table = wide_table.iloc[split_idx:].reset_index(drop=True)
    return train_table, future_table


def _derive_split_paths(output_path: str) -> Tuple[str, str]:
    path = Path(output_path)
    train_path = path.with_name("{}_train.csv".format(path.stem))
    future_path = path.with_name("{}_future.csv".format(path.stem))
    return str(train_path), str(future_path)


def save_wide_table(table: pd.DataFrame, output_path: str) -> None:
    """Save the processed [T, F] table as a CSV with feature headers."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def prepare_tfb_dataset(
    input_path: str,
    output_path: str,
    date_col: str = "date",
    value_col: str = "data",
    var_col: str = "cols",
    fill_method: str = "ffill",
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load, convert, and save a TFB dataset as a numeric [T, F] table."""
    df = load_tfb_csv(
        path=input_path,
        date_col=date_col,
        value_col=value_col,
        var_col=var_col,
    )
    wide_table = tfb_to_wide_table(
        df=df,
        date_col=date_col,
        value_col=value_col,
        var_col=var_col,
        fill_method=fill_method,
        target_cols=target_cols,
    )
    save_wide_table(wide_table, output_path=output_path)
    return wide_table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a TFB-format CSV into the [T, F] table used by this project."
    )
    parser.add_argument("--input-path", required=True, help="Path to the raw TFB-format CSV.")
    parser.add_argument("--output-path", required=True, help="Path to save the processed [T, F] CSV.")
    parser.add_argument("--date-col", default="date", help="Timestamp column name.")
    parser.add_argument("--value-col", default="data", help="Value column name.")
    parser.add_argument("--var-col", default="cols", help="Variable-name column.")
    parser.add_argument(
        "--fill-method",
        default="ffill",
        choices=("ffill", "drop"),
        help="How to handle missing values after pivot.",
    )
    parser.add_argument(
        "--target-cols",
        nargs="*",
        default=None,
        help="Optional target columns to move to the end of the output table.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Optional. Validate that the processed table can provide the last inference window.",
    )
    parser.add_argument(
        "--future-ratio",
        type=float,
        default=None,
        help=(
            "Optional chronological holdout ratio for future_wide split. "
            "Example: 0.2 means first 80% train_wide, last 20% future_wide."
        ),
    )
    parser.add_argument(
        "--train-output-path",
        default=None,
        help="Optional output path for train split wide CSV.",
    )
    parser.add_argument(
        "--future-output-path",
        default=None,
        help="Optional output path for future split wide CSV.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    wide_table = prepare_tfb_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        date_col=args.date_col,
        value_col=args.value_col,
        var_col=args.var_col,
        fill_method=args.fill_method,
        target_cols=args.target_cols,
    )

    print("Saved processed table to:", args.output_path)
    print("Processed shape:", tuple(wide_table.shape))
    print("Columns:", list(wide_table.columns))

    if args.future_ratio is not None:
        default_train_path, default_future_path = _derive_split_paths(args.output_path)
        train_output_path = args.train_output_path or default_train_path
        future_output_path = args.future_output_path or default_future_path

        train_table, future_table = split_wide_table_for_future(
            wide_table=wide_table,
            future_ratio=float(args.future_ratio),
        )
        save_wide_table(train_table, train_output_path)
        save_wide_table(future_table, future_output_path)

        print("Saved train split to:", train_output_path)
        print("Train split shape:", tuple(train_table.shape))
        print("Saved future split to:", future_output_path)
        print("Future split shape:", tuple(future_table.shape))

    if args.input_length is not None:
        last_window = extract_last_input_window(wide_table, input_length=args.input_length)
        print("Last inference window shape:", tuple(last_window.shape))


if __name__ == "__main__":
    main()
