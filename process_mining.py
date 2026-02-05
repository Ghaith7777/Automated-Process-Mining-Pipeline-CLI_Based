# -*- coding: utf-8 -*-
"""
Portfolio Mini-Project: Process Mapping / Process Mining (DFG) on Event Logs
Author: Ghaith
Created: 2025-12-22

What this script demonstrates (portfolio-focused):
- Event log preparation (timestamps, schema, ordering)
- Basic process map discovery using a Directly-Follows Graph (DFG)
- Operational KPIs: case count, event count, activity frequency, throughput time distribution
- Export of results (CSV/JSON) for reporting / sharing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pm4py as p4


# logging
logger = logging.getLogger("process_mining_portfolio")


def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# Core functions
REQUIRED_RAW_COLS = {"case_id", "activity_name", "timestamp"}
PM4PY_COLS = ["case:concept:name", "concept:name", "time:timestamp"]


def load_csv(path: str) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Input file not found: {fp}")
    df = pd.read_csv(fp)
    missing = REQUIRED_RAW_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    return df


def prepare_event_log(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.loc[:, ["case_id", "activity_name", "timestamp"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(f"{bad} rows have invalid timestamps. Please clean the CSV or fix parsing.")
    df = df.rename(
        columns={
            "case_id": "case:concept:name",
            "activity_name": "concept:name",
            "timestamp": "time:timestamp",
        }
    )
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="mergesort")
    df = p4.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    for c in PM4PY_COLS:
        if c not in df.columns:
            raise ValueError(f"After formatting, expected column missing: {c}")
    return df


def compute_kpis(df: pd.DataFrame) -> Dict:
    n_cases = df["case:concept:name"].nunique()
    n_events = len(df)
    n_activities = df["concept:name"].nunique()
    activity_freq = df["concept:name"].value_counts().to_dict()
    g = df.groupby("case:concept:name")["time:timestamp"]
    case_span = g.agg(start="min", end="max")
    case_span["throughput_hours"] = (case_span["end"] - case_span["start"]).dt.total_seconds() / 3600.0
    kpis = {
        "cases": int(n_cases),
        "events": int(n_events),
        "unique_activities": int(n_activities),
        "avg_events_per_case": float(n_events / n_cases) if n_cases else None,
        "time_window": {
            "first_event": str(df["time:timestamp"].min()),
            "last_event": str(df["time:timestamp"].max()),
        },
        "throughput_hours": {
            "mean": float(case_span["throughput_hours"].mean()),
            "median": float(case_span["throughput_hours"].median()),
            "min": float(case_span["throughput_hours"].min()),
            "max": float(case_span["throughput_hours"].max()),
            "p95": float(case_span["throughput_hours"].quantile(0.95)),
        },
        "activity_frequency_top10": dict(list(activity_freq.items())[:10]),
    }
    return kpis


def discover_dfg(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    dfg, start_acts, end_acts = p4.discover_dfg(df)
    return dfg, start_acts, end_acts


def dfg_to_edges_df(dfg: Dict) -> pd.DataFrame:
    rows = []
    for (src, tgt), freq in dfg.items():
        rows.append({"source": src, "target": tgt, "frequency": int(freq)})
    edges = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    return edges


def export_outputs(
    outdir: str,
    kpis: Dict,
    edges_df: pd.DataFrame,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kpis.json").write_text(json.dumps(kpis, indent=2, ensure_ascii=False), encoding="utf-8")
    edges_df.to_csv(out / "dfg_edges.csv", index=False, encoding="utf-8")
    md = []
    md.append("# Process Mapping Summary (DFG)\n")
    md.append(f"- Cases: **{kpis['cases']}**")
    md.append(f"- Events: **{kpis['events']}**")
    md.append(f"- Unique activities: **{kpis['unique_activities']}**")
    md.append(f"- Time window: **{kpis['time_window']['first_event']}** → **{kpis['time_window']['last_event']}**")
    md.append("\n## Throughput time (hours)\n")
    th = kpis["throughput_hours"]
    md.append(f"- Mean: **{th['mean']:.2f}** | Median: **{th['median']:.2f}** | P95: **{th['p95']:.2f}**")
    md.append(f"- Min: **{th['min']:.2f}** | Max: **{th['max']:.2f}**\n")
    md.append("## Top 10 DFG transitions\n")
    md.append(edges_df.head(10).to_markdown(index=False))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Exported outputs to: %s", out.resolve())


def maybe_view_dfg(df: pd.DataFrame, no_view: bool) -> None:
    if no_view:
        logger.info("Skipping DFG viewer (no_view=True).")
        return
    dfg, start_acts, end_acts = p4.discover_dfg(df)
    p4.view_dfg(dfg, start_acts, end_acts)
    logger.info("DFG viewer opened.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Portfolio Process Mapping (DFG) on Event Logs (pm4py).")
    p.add_argument("--input", required=True, help="Path to CSV (must contain case_id, activity_name, timestamp).")
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs).")
    p.add_argument("--loglevel", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    p.add_argument("--no-view", action="store_true", help="Do not open the pm4py DFG viewer.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.loglevel)
    logger.info("Loading input: %s", args.input)
    df_raw = load_csv(args.input)
    logger.info("Preparing event log for pm4py...")
    df_log = prepare_event_log(df_raw)
    logger.info("Computing KPIs...")
    kpis = compute_kpis(df_log)
    logger.info("Discovering DFG...")
    dfg, start_acts, end_acts = discover_dfg(df_log)
    edges_df = dfg_to_edges_df(dfg)
    logger.info("Exporting outputs...")
    export_outputs(args.outdir, kpis, edges_df)
    maybe_view_dfg(df_log, args.no_view)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
