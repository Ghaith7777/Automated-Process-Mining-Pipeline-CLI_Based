  # Automated Process Mining Pipeline (CLI_Based)

This project implements a reusable command-line pipeline for discovering process maps from event logs using Directly-Follows Graphs (DFG).

It was built as a portfolio project to demonstrate practical process-mining workflows, KPI engineering, and clean Python tooling rather than as a research prototype.
## Overview

This tool analyzes event logs to extract process structure and operational metrics. It's ideal for business process improvement, compliance auditing, and workflow optimization across industries (healthcare, finance, manufacturing, logistics).

**Key Capabilities:**
- Automated process discovery from raw event logs
- Directly-Follows Graph (DFG) generation
- Operational KPI computation (throughput time, case volume, activity frequency)
- Data validation and quality checks
- Multi-format export (JSON, CSV, Markdown)

## Installation

### Requirements
- Python 3.8+
- pandas
- pm4py (Process Mining for Python)

### Setup

```bash

# Clone the repository
git clone https://github.com/Ghaith7777/Automated-Process-Mining-Pipeline-CLI_Based.git 
cd automated-process-mining-pipeline

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate the environment
# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
if you are using Anaconda:
```bash
conda create -n pm4py_env python=3.11
conda activate pm4py_env
pip install -r requirements.txt
```
## Quick Start

### Basic Usage

```bash
python process_mining.py --input events.csv --outdir ./results
```

### With Options

```bash
python process_mining.py \
  --input data/event_log.csv \
  --outdir outputs \
  --loglevel INFO \
  --no-view
```

### Input Format

Your CSV must contain exactly these three columns (column names matter):

```
case_id,activity_name,timestamp
CASE-001,Receive Order,2025-01-01 08:30:00
CASE-001,Validate Order,2025-01-01 09:15:00
CASE-001,Process Payment,2025-01-01 10:00:00
CASE-002,Receive Order,2025-01-01 08:45:00
...
```

**Column Specifications:**
- **case_id**: Unique process instance identifier (string)
- **activity_name**: Activity/task name (string)
- **timestamp**: Event timestamp in ISO 8601 format (YYYY-MM-DD HH:MM:SS)

**Data Quality Requirements:**
- No missing values in required columns
- Timestamps must be parseable and valid
- Cases should have at least 2 events (start → end)
- Chronological ordering within cases is recommended (script sorts automatically)

## Usage Examples

### Example 1: E-commerce Order Processing

```bash
python process_mining.py --input samples/orders.csv --outdir results/orders
```

Analyzes how customer orders flow through validation, payment, fulfillment, and shipping.

### Example 2: Claims Processing (Headless)

```bash
python process_mining.py \
  --input samples/claims.csv \
  --outdir results/claims \
  --loglevel DEBUG \
  --no-view
```

Ideal for batch processing in CI/CD pipelines or cloud environments.

## Output Files

The script generates four files in your output directory:

### 1. `kpis.json` - Operational Metrics
Comprehensive KPI summary in JSON format:

```json
{
  "cases": 1250,
  "events": 8934,
  "unique_activities": 12,
  "avg_events_per_case": 7.15,
  "time_window": {
    "first_event": "2025-01-01 08:00:00",
    "last_event": "2025-01-31 17:30:00"
  },
  "throughput_hours": {
    "mean": 48.5,
    "median": 36.2,
    "min": 0.5,
    "max": 240.0,
    "p95": 120.3
  },
  "activity_frequency_top10": {
    "Review": 1250,
    "Approve": 1189,
    "Send Notification": 1200,
    ...
  }
}
```

### 2. `dfg_edges.csv` - Process Transitions
Directly-Follows Graph edges sorted by frequency:

```csv
source,target,frequency
Receive Order,Validate Order,1200
Validate Order,Process Payment,1189
Process Payment,Pack Items,1150
...
```

### 3. `summary.md` - Executive Summary
Human-readable markdown report with highlights:
- Case and event counts
- Time window span
- Throughput time statistics
- Top 10 process transitions

### 4. Interactive DFG Viewer (if enabled)
Visual process map showing:
- Activities as nodes
- Transitions as directed edges
- Frequency-based edge thickness
- Start/end activity highlights

## Architecture & Design

### Core Components

**Data Pipeline:**
```
CSV Input → Validation → Schema Conversion → Sorting → pm4py Processing → KPI Extraction → Export
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_csv()` | Load and validate CSV structure |
| `prepare_event_log()` | Convert to pm4py format, parse timestamps, sort |
| `compute_kpis()` | Calculate operational metrics |
| `discover_dfg()` | Generate Directly-Follows Graph |
| `export_outputs()` | Write results to files |

### Design Patterns

- **Strict Input Validation:** Fails fast with clear error messages
- **Pandas-First Approach:** Leverages pandas for data manipulation, pm4py for process algorithms
- **Defensive Programming:** Type hints, error handling, data quality checks
- **Modular Architecture:** Composable functions for extensibility
- **Structured Logging:** DEBUG/INFO/WARNING levels for visibility

## Advanced Usage

### Integration with Python Code

```python
from process_mining import prepare_event_log, compute_kpis, discover_dfg

# Load your own data
df = pd.read_csv("my_events.csv")
log = prepare_event_log(df)

# Compute KPIs programmatically
kpis = compute_kpis(log)
print(f"Process efficiency: {kpis['avg_events_per_case']} events/case")

# Get DFG for custom visualization
dfg, start_acts, end_acts = discover_dfg(log)
```

### Filtering & Analysis

```python
# Analyze a specific time window
df_filtered = df[
    (df['timestamp'] >= '2025-01-15') & 
    (df['timestamp'] < '2025-01-22')
]
log = prepare_event_log(df_filtered)

# Focus on slow cases
case_durations = log.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
slow_cases = case_durations[case_durations['max'] - case_durations['min'] > pd.Timedelta(hours=72)]
```

### Extending the Script

Common extensions:

1. **Variant Analysis:** Group cases by their activity sequence
2. **Conformance Checking:** Compare actual process to reference model
3. **Resource Analysis:** Add actor/resource columns to track person responsible
4. **Bottleneck Detection:** Identify activities with longest avg duration
5. **Anomaly Detection:** Flag unusual case paths or durations

## Command Line Reference

```
usage: process_mining.py [-h] --input INPUT [--outdir OUTDIR] 
                         [--loglevel LOGLEVEL] [--no-view]

Portfolio Process Mapping (DFG) on Event Logs (pm4py).

optional arguments:
  -h, --help            Show help message
  --input INPUT         Path to CSV (must contain case_id, activity_name, timestamp)
  --outdir OUTDIR       Output directory (default: outputs)
  --loglevel LOGLEVEL   Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
  --no-view             Do not open the pm4py DFG viewer
```

## Performance & Scalability

| Metric | Typical Values | Notes |
|--------|---|---|
| Cases | Up to 100K | Larger datasets benefit from date filtering |
| Events | Up to 1M | Depends on activity granularity |
| Memory | ~500MB-2GB | For 1M events; scales linearly with data size |
| Runtime | 5-30 seconds | Most time spent in pm4py DFG discovery |

Benchmarks were obtained on a local workstation; actual performance depends on hardware and dataset characteristics.

**Optimization Tips:**
- Filter to time windows of interest before processing
- Use `--no-view` for batch processing (viewer adds 10-30 sec)
- For very large logs (>10M events), consider sampling or windowing

## Troubleshooting

### Missing Columns Error
```
ValueError: Missing required columns in CSV: {'activity_name'}
```
**Solution:** Ensure CSV has exactly: `case_id`, `activity_name`, `timestamp`

### Invalid Timestamps
```
ValueError: 123 rows have invalid timestamps. Please clean the CSV or fix parsing.
```
**Solution:** Check timestamp format. Accepted: `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD`, ISO 8601 variants

### DFG Viewer Won't Open (Headless Environments)
```bash
# Use --no-view flag
python process_mining.py --input data.csv --no-view
```

### Out of Memory
**For large datasets:**
- Filter to smaller time window: `df[df['timestamp'] >= '2025-06-01']`
- Sample cases: `df[df['case_id'].isin(sample_cases)]`
- Increase system RAM or use cloud compute

## Portfolio Highlights

This project demonstrates:

✅ **Software Engineering:** Clean code, error handling, logging, modularity  
✅ **Data Engineering:** CSV parsing, schema validation, timestamp handling  
✅ **Process Mining:** DFG discovery, KPI computation, domain knowledge  
✅ **Python Proficiency:** Type hints, pandas, argparse, pathlib  
✅ **Documentation:** README, docstrings, examples, architecture clarity  
✅ **Testing Mindset:** Input validation, edge cases, sensible defaults  

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation & analysis |
| pm4py | >=2.7.0 | Process mining algorithms (DFG, KPI) |

See `requirements.txt` for complete list.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Conformance checking against reference models
- [ ] Variant discovery and analysis
- [ ] Resource/actor analysis (if resource column provided)
- [ ] Automated bottleneck detection
- [ ] Anomaly detection for unusual traces
- [ ] Web UI for interactive exploration
- [ ] Support for XES (eXtensible Event Stream) format
- [ ] Parallel processing for large datasets

## References

- [pm4py Documentation](https://pm4py.fit.fraunhofer.de/)
- [Process Mining Handbook](https://www.springer.com/gp/book/9783662491379)
- [IEEE 1849 Event Log Standard](https://standards.ieee.org/ieee/1849/6579/)

## Contact & Support

For questions or issues:
- Open a GitHub Issue
- Check the Troubleshooting section above
- Review example files in `samples/`

---

**Author:** Ghaith   
**Last Updated:** 2025-02-05  
**Status:** Production Ready
