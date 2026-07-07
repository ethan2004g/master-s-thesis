# Pokemon Thesis — Phase 1

Transformer-based offline RL for competitive Pokemon. This repo holds thesis code. The local Showdown server lives in `pokemon-thesis-tools/` (not tracked by git).

## Prerequisites

- Python 3.10+
- Node.js 18+
- Local [pokemon-showdown](https://github.com/smogon/pokemon-showdown) clone (see install notes below)

## Setup

```powershell
cd "C:\Users\ethan\Masters Thesis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Showdown server (one-time)

```powershell
cd pokemon-thesis-tools
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
Copy-Item config\config-example.js config\config.js
```

## Daily workflow

**Terminal 1 — start the server**

```powershell
cd "C:\Users\ethan\Masters Thesis\pokemon-thesis-tools\pokemon-showdown"
node pokemon-showdown start --no-security
```

**Terminal 2 — run baseline battles**

```powershell
cd "C:\Users\ethan\Masters Thesis"
.\.venv\Scripts\Activate.ps1
python scripts/run_baseline.py
```

Open [http://localhost:8000](http://localhost:8000) to watch battles in the browser.

## Baseline script

`scripts/run_baseline.py` runs two `RandomPlayer` agents against each other and writes per-turn JSON lines to `logs/`. Each line includes weather, field effects, side conditions, party snapshots, and the chosen action.

```powershell
python scripts/run_baseline.py --battles 10
python scripts/run_baseline.py --battles 5 --min-turns 100
```

## Project layout

```
Masters Thesis/
  scripts/           Python entry points
  logs/              Battle traces (gitignored)
  pokemon-thesis-tools/   Showdown server (gitignored)
  .venv/             Python environment (gitignored)
```

## Turn tokenization (13 tokens per turn)

Each turn is encoded as exactly 13 discrete tokens.

| Index | Token |
|-------|--------|
| 0 | Field (weather, terrain, your hazards, opponent hazards) |
| 1–6 | Your party slots (padded to 6) |
| 7–12 | Opponent party slots (padded to 6, partial info when unknown) |

Encode a baseline log file:

```powershell
python scripts/encode_logs.py logs/baseline_20260529T011709Z.jsonl
```

This writes `logs/encoded_<name>.jsonl` and `logs/vocab.json`.

## BC inference (live battles)

After training, pit the model against a random bot.

```powershell
python scripts/run_bc_match.py --battles 5
```

Requires `checkpoints/bc_best.pt` from training.

## Bulk data collection

Run many baseline battles, encode, and retrain in one command.

```powershell
python scripts/collect_data.py --battles 50 --train
```

## Continuous training (run until you stop)

Keeps looping collect → encode all logs → train. Press **Ctrl+C** to stop.

```powershell
python scripts/continuous_train.py --battles-per-cycle 10 --epochs 15
```

Optional BC evaluation after each cycle.

```powershell
python scripts/continuous_train.py --battles-per-cycle 10 --epochs 15 --eval-battles 3
```

## Behavioral cloning training

Install PyTorch (once).

```powershell
pip install -r requirements.txt
```

Train on encoded logs. Use multiple encoded files for more data.

```powershell
python scripts/train_bc.py logs/encoded_baseline_20260529T011709Z.jsonl --epochs 30
```

Checkpoints land in `checkpoints/bc_best.pt` with `action_vocab.json` and `bc_config.json`.

The model uses self-attention across 13 tokens per turn, then causal attention across the last N turns, and predicts the logged action (move or switch).

## Phase roadmap

1. **Done** — Local server + poke-env baseline logging
2. **Done** — 13-token turn encoder
3. **Done** — PyTorch dataset + BC training loop
4. **Done** — BCPlayer inference + bulk data collection
5. **Now** — Metamon human replay data (see [docs/METAMON.md](docs/METAMON.md))
6. **Now** — BeliefTransformer + combinatorial splits + thesis PDF
7. **Next** — Full Metamon export, belief training, offline RL (CQL/AMAGO)

## Thesis document

Build the LaTeX thesis PDF from `thesis/main.tex`:

```powershell
.\scripts\compile_thesis.ps1
```

Output is `thesis/main.pdf`. Edit chapter files under `thesis/chapters/`.

## Belief-augmented training

After exporting Metamon data with belief labels:

```powershell
python scripts/export_metamon_subset.py --max-battles 1000
python scripts/train_belief_bc.py logs/encoded_metamon_subset.jsonl --epochs 30
```

Checkpoints land in `checkpoints/belief_bc_best.pt`.
