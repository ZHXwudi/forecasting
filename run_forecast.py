from __future__ import annotations

import argparse

from forecasting import HOURS_PER_YEAR, load_price_data, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run electricity price forecasting backtests and future forecasts.")
    parser.add_argument("--data", default="price_input_foreign.csv", help="CSV file path.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--future-hours", type=int, default=HOURS_PER_YEAR, help="Future forecast horizon in hours.")
    args = parser.parse_args()

    df = load_price_data(args.data)
    files = save_outputs(df, args.output, args.future_hours)
    print("Generated files:")
    for name, path in files.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
