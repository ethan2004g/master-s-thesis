# Research workflow

The thesis PDF under `thesis/` is a **blueprint**. This file tracks live experiments.

## Phase 1 — Human BC baseline (done / in progress)

```powershell
cd "C:\Users\ethan\Masters Thesis"
.\.venv\Scripts\Activate.ps1
python scripts/train_bc.py logs/encoded_metamon_subset.jsonl --epochs 30 --resume checkpoints/human_bc/bc_best.pt --out-dir checkpoints/human_bc
python scripts/eval_bc.py --checkpoint-dir checkpoints/human_bc
```

**Pass bar:** val action accuracy well above chance (~17%+ on 988 classes).

## Phase 2 — Belief + matchup (current)

### Scale-up (done — 2000 battles, 20 epochs)

```powershell
.\.venv-metamon\Scripts\Activate.ps1
python scripts/scale_belief_research.py --battles 2000 --epochs 20
```

Completed Jul 2. Results in `logs/ood_comparison.json` and `logs/research_phase2_summary.json`.

### Step A — Export belief-enriched corpus

```powershell
.\.venv-metamon\Scripts\Activate.ps1
$env:METAMON_CACHE_DIR = "C:\Users\ethan\Masters Thesis\data\metamon_clean"
python scripts/export_belief_corpus.py --battles 2000
```

Labels: hidden **moves / item / ability / tera** on the active opponent, plus **team matchup** keys.

### Step B — Train models

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_matchup_bc.py logs/encoded_belief_corpus.jsonl --epochs 20
python scripts/train_active_belief.py logs/encoded_belief_corpus.jsonl --epochs 20
```

| Checkpoint | Script | What it tests |
|------------|--------|----------------|
| `checkpoints/human_bc/` | `train_bc.py` | Turn history only |
| `checkpoints/matchup_bc/` | `train_matchup_bc.py` | + team composition |
| `checkpoints/active_belief/` | `train_active_belief.py` | + hidden set belief |

### Step C — OOD evaluation

Evaluate on the **belief corpus** so all models share vocabulary and team keys.

```powershell
python scripts/train_bc.py logs/encoded_belief_corpus.jsonl --token-vocab logs/vocab_belief.json --epochs 12 --out-dir checkpoints/human_bc_belief
python scripts/eval_all_ood.py
```

Results land in `logs/ood_comparison.json`.

Compare **id_train_team_accuracy** vs **ood_team_accuracy** (held-out opponent team keys). Random val split is a separate metric (unseen battle trajectories).

Current results (2000-battle corpus, 20 epochs)

| Model | Val acc | ID train teams | OOD teams | Gap |
|-------|---------|----------------|-----------|-----|
| human_bc_belief | 20.4% | 45.2% | 45.1% | 0.05% |
| matchup_bc | 19.5% | 42.6% | 42.1% | 0.4% |
| active_belief | 19.0% | 44.3% | 44.4% | ~0% |

`active_belief` move-belief val accuracy reached **28.1%** (was 15.5% at 500 battles).

Prior 500-battle run showed ~1.9% team gaps. Scaling collapsed the gap, suggesting team memorization was less of a bottleneck at 2000 battles.

### Step D — Live battles (optional)

Requires local Showdown server running.

```powershell
python scripts/run_bc_match.py --battles 5 --checkpoint-dir checkpoints/human_bc
```

## Phase 3 — Thesis results (in progress)

Chapter 7 (Results), Chapter 8 (Analysis), and Chapter 10 (Conclusion) drafted from scaled experiments Jul 3.

```powershell
.\scripts\compile_thesis.ps1
```

### Belief fusion ablation (done)

Concatenating belief softmax outputs into the policy head did **not** help.

| Model | Val action | OOD teams |
|-------|------------|-----------|
| active_belief | 20.2% | 44.4% |
| active_belief_fusion | 19.6% | 43.3% |

Checkpoint: `checkpoints/active_belief_fusion/`. OOD eval: `logs/eval_active_belief_fusion_ood.json`.

### MLP / GRU belief baselines (done)

| Model | Val move belief |
|-------|-----------------|
| Linear probe (frozen BC trunk) | 10.5% |
| GRU | 17.2% |
| ActiveBelief | 28.1% |
| **MLP (latest turn)** | **35.5%** |

Results in `logs/belief_baseline_results.json`. Latest-turn MLP beats the transformer on move belief.

Remaining `\todo` items in the thesis: live battles, Split C, CQL.

### Trunk probe (done)

```powershell
python scripts/probe_trunk_belief.py --epochs 15
```

Results in `logs/trunk_probe_results.json`. Linear probe on frozen BC trunk: **10.5%** move belief vs **27.7%** for ActiveBelief heads on the same val split.

## Notes

- Gen 9 OU team preview reveals all opponent **species**. Species belief is empty by design.
- Active-opponent **move/item/tera** belief is the main supervision signal now.
- Use `logs/research_phase1_summary.json` and `logs/bc_eval_results.json` for quick status.
