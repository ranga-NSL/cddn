# Configuration-Driven Dependency Network (CDN)

A fundamentals-driven dependency engine that computes what each higher-level node *can honestly claim* based on the weakest fundamentals feeding it, and visually flags when visible ratings are inflated.

**Core Concept:** A configurable multi-level dependency graph where each node's true capability is limited by the weakest of its prerequisites, and where any over-optimistic visible rating is explicitly flagged.

## Quick Start

### Setup

```powershell
# 1. Setup environment
.\setup_environment.ps1
# 2. Activate environment
.\cdn_env\Scripts\Activate.ps1
# 3. Run analysis
python CDN_graph.py
```

### Basic Usage

```python
from CDN_graph import load_from_excel
# Load network from Excel
net = load_from_excel('cdn_network.xlsx')
# Analyze with method chaining
net.compute().summary().visualize()
```

## What It Does

1. **Reads Excel configuration** with multiple levels (Level_0, Level_1, Level_2, ...)
2. **Computes fundamental scores** using weakest-link logic (minimum of dependencies)
3. **Compares visible vs fundamental** scores and classifies nodes:
   - `WARNING: INFLATED by X.XX` if visible > fundamental
   - `OK: Aligned` if equal
   - `OK: Conservative by X.XX` if visible < fundamental
4. **Visualizes the network** with red borders on inflated nodes

## Excel Structure

Each level is a separate sheet:

- **Level_0**: Description | Base_Score (fundamentals)
- **Level_1+**: Description | Apparent_Score | [empty] | L{level}_0 | L{level}_1 | ...

**Note:** Description column is always first. Any column containing "score" works (Base_Score, Apparent_Score, etc.). Dependency columns use pattern `L{level}_{index}`.

Edit the Excel file to modify the network - no code changes needed!

## Project Files

- `CDN_graph.py` - Main engine and executable
- `cdn_network.xlsx` - Configuration file (auto-created if missing)
- `PROJECT_DOCUMENTATION.md` - **Detailed documentation with use cases**

## Use Cases

- **Manufacturing**: Asset health, reliability KPIs, safety barriers, AI readiness
- **Education**: Skill trees, competency maps, curriculum design
- **Any hierarchical system** where fundamentals matter

See `PROJECT_DOCUMENTATION.md` for detailed use cases and examples.

## Requirements

- Python 3.13 (located at `D:\python313\python.exe`)
- numpy, networkx, matplotlib, pandas, openpyxl

Install via `pip install -r requirements.txt` or use `setup_environment.ps1`
