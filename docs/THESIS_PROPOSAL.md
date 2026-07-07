# Master's Thesis Proposal

## Title

**Transformer-Based Belief Modeling for Offline Policy Learning in Partially Observed, Simultaneous-Action Games**

*Evaluation domain: competitive Pokémon battles (Gen 9 OU human replays via Metamon)*

---

## Executive Summary

This thesis studies how neural sequence models can infer hidden opponent information from interaction history in partially observed, simultaneous-action environments. The central claim is that a causal transformer can act as an implicit belief-state updater. Better beliefs should improve downstream policy quality, especially when the agent faces team compositions, move sets, or item combinations that were rare or absent during training.

Pokémon battles are used only as a benchmark. They combine partial observability, simultaneous move commitment, and combinatorial structure in a single domain with large-scale offline human data. The scientific contribution is about belief inference and combinatorial generalization, not about game-specific heuristics or lookahead search.

---

## 1. Motivation

Many real decision problems share three properties that standard fully observed RL benchmarks underrepresent.

**Partial observability.** Important state variables are hidden. In competitive Pokémon, unrevealed Pokémon, unknown moves, items, abilities, and terastallization types must be inferred from clues such as damage ranges, switch patterns, and revealed moves.

**Simultaneous action selection.** Both agents commit to actions before the environment resolves the turn. This differs from chess-like turn-taking and creates a distinct learning problem around anticipating opponent intent under uncertainty.

**Combinatorial structure.** The space of valid teams, sets, and turn-level action combinations grows quickly. A model that memorizes frequent patterns may fail on held-out combinations even when overall win rate looks acceptable.

Transformers are a natural fit because they can aggregate evidence across long interaction histories with causal attention. Offline learning is appropriate because high-quality human trajectories are available at scale through the Metamon dataset, and the thesis goal is inference-time policies without search trees or simulators.

---

## 2. Problem Statement

Given a sequence of first-person observations from past turns, learn a model that

1. Maintains a useful belief over hidden opponent attributes.
2. Uses that belief to choose strong actions in new battles.
3. Generalizes to opponent team and set combinations outside the training distribution.

Formally, each turn produces an observation $o_t$ derived from public battle state. Hidden variables $z_t$ include unrevealed species, moves, items, abilities, and terastallization types for opponent party members. Actions $a_t$ are chosen under simultaneous-move constraints. The model receives offline trajectories $\{(o_{1:T}, a_{1:T}, z_{1:T})\}$ reconstructed from human replays where $z$ is known post hoc from full battle logs.

The thesis asks whether explicit belief supervision and transformer-based history modeling improve both belief accuracy and policy performance under combinatorial distribution shift.

---

## 3. Research Questions

**RQ1 (Belief inference).** Can a causal transformer trained on interaction history predict hidden opponent attributes more accurately than classical and neural baselines?

**RQ2 (Belief usefulness).** Does improved belief accuracy translate into better action selection, measured by behavioral cloning loss, offline policy scores, and live win rate against fixed opponents?

**RQ3 (Combinatorial generalization).** Under held-out team and set splits, do belief-augmented transformer policies degrade more gracefully than models that lack explicit belief training or history modeling?

**RQ4 (Architecture).** How much of the gain comes from within-turn structure (13-token self-attention), cross-turn causal attention, and auxiliary belief heads versus a flat sequence baseline?

---

## 4. Hypotheses

**H1.** A transformer with causal attention over turn history will outperform feedforward and recurrent models on hidden-attribute prediction from the same tokenized observations.

**H2.** Adding an auxiliary belief prediction loss during training will improve both belief metrics and downstream policy metrics relative to action-only behavioral cloning.

**H3.** Models with stronger belief accuracy will show smaller performance drops on combinatorial held-out splits than models matched on in-distribution policy accuracy.

**H4.** Conservative offline RL fine-tuning on top of a belief-trained BC initialization will yield additional policy gains without requiring search at inference time.

---

## 5. Related Work (positioning)

| Area | Relevance |
|------|-----------|
| POMDPs and belief-state planning | Theoretical framing for hidden opponent state |
| Opponent modeling in games | Direct precedent for predicting hidden attributes and next actions |
| Transformer policies and decision transformers | Sequence modeling for RL without online planning |
| Offline RL (BC, CQL, AMAGO) | Learning from fixed logs without environment interaction at train time |
| Combinatorial generalization in ML | Evaluation protocol for OOD team and set splits |
| Metamon / poke-env ecosystems | Data and environment infrastructure for this benchmark |

The novelty is not Pokémon playing per se. It is the joint study of transformer belief inference and combinatorial generalization in a simultaneous-move, partially observed offline RL setting with structured discrete tokenization.

---

## 6. Method

### 6.1 Observation and tokenization

Each turn is encoded as exactly **13 discrete tokens**, following the existing project layout.

| Index | Token role |
|-------|------------|
| 0 | Field state (weather, terrain, hazards) |
| 1–6 | Player party slots |
| 7–12 | Opponent party slots (partial info when unrevealed) |

This representation is implemented in `TurnTokenizer` and supports self-attention within a turn. A battle is a sequence of turns with length up to context window $N$.

Data source: reconstructed first-person trajectories from Metamon human replays (Gen 9 OU to start). Random-bot logs remain for pipeline debugging only.

### 6.2 Belief targets (supervision)

Belief heads predict hidden opponent attributes that are knowable from full logs but not from first-person view at time $t$. Primary targets, ordered by importance:

1. **Species identity** for each unrevealed opponent slot.
2. **Moves** for revealed or partially revealed opponent Pokémon.
3. **Item** (when inferable before reveal).
4. **Ability** (when not yet shown).
5. **Terastallization type** (Gen 9 OU).

Each target is a multi-class prediction per slot. Unavailable or already revealed slots are masked out of the loss. Belief labels are extracted from Metamon full-state annotations at each turn boundary.

### 6.3 Model architecture

Extend the existing **TurnTransformer** (`src/pokemon_thesis/model/turn_transformer.py`).

**Within-turn encoder.** Self-attention across the 13 tokens produces one turn embedding.

**Cross-turn encoder.** Causal transformer layers over turn embeddings maintain a history summary that cannot peek at future turns.

**Policy head.** Linear classifier over the final valid turn embedding predicts the expert action (behavioral cloning).

**Belief heads.** One small MLP per belief target type, reading the same final turn embedding (or optionally per-slot opponent token outputs for finer-grained predictions).

**Optional belief fusion variant.** Concatenate belief head softmax outputs to the policy head input to test whether explicit belief features help action selection beyond shared representation learning.

Total parameters stay modest (on the order of 1–5M) for Master's-scale compute.

### 6.4 Training procedure

**Stage A — Belief-augmented behavioral cloning**

$$
\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{bel}} \sum_{k} \mathcal{L}_{\text{bel}}^{(k)}
$$

- $\mathcal{L}_{\text{action}}$ is cross-entropy against the human action.
- $\mathcal{L}_{\text{bel}}^{(k)}$ is cross-entropy for each belief target type with slot masking.
- $\lambda_{\text{bel}}$ is tuned on a validation split.

**Stage B — Conservative offline RL fine-tuning (optional but recommended)**

Initialize from Stage A. Fine-tune with **CQL** or **AMAGO** on the same offline trajectories using the transformer as the policy backbone. No search or simulator rollouts at inference.

**Stage C — Evaluation only**

No online training during benchmark battles. All generalization tests use frozen checkpoints.

### 6.5 Inference

At decision time the model receives the tokenized history window and outputs an action distribution in one forward pass. No lookahead search, no Monte Carlo rollouts, no live simulator queries.

---

## 7. Baselines

Baselines are chosen to cover classical ML, simpler neural models, and ablations of the proposed architecture.

### 7.1 Belief prediction baselines

| ID | Model | Purpose |
|----|-------|---------|
| B1 | **Gradient-boosted trees (XGBoost/LightGBM)** on hand-crafted features from the current turn | Classical ML reference from coursework |
| B2 | **Multinomial logistic regression** on the same features | Simple linear baseline |
| B3 | **Feedforward MLP** on flattened latest-turn tokens only | Tests need for history |
| B4 | **GRU/LSTM** over turn embeddings | Recurrent history baseline |
| B5 | **TurnTransformer without belief loss** | Isolates auxiliary supervision benefit |
| B6 | **Flat token sequence transformer** (13×T tokens, single causal stack) | Tests hierarchical turn structure |

### 7.2 Policy baselines

| ID | Model | Purpose |
|----|-------|---------|
| P1 | **Behavioral cloning MLP** (latest turn only) | Naive imitation |
| P2 | **BC + GRU** | Recurrent imitation |
| P3 | **BC TurnTransformer** (existing implementation) | Current project baseline |
| P4 | **Belief-augmented BC TurnTransformer** (proposed) | Primary method |
| P5 | **P4 + CQL or AMAGO** | Offline RL improvement |
| P6 | **poke-env RandomPlayer / heuristic bot** | Lower bound in live play |
| P7 | **Human expert actions** (dataset actions) | Upper reference for BC metrics |

### 7.3 Interpretability analysis (supporting, not primary)

Train shallow **decision trees** on the same hand-crafted features as B1. Report which observable signals (damage hints, switch timing, move reveals) best predict specific hidden attributes. This connects the thesis to decision tree coursework and supports analysis of what the transformer may be learning implicitly.

---

## 8. Experiment Plan

### 8.1 Dataset

- **Primary corpus:** Metamon Gen 9 OU human replays.
- **Scale target:** at least 50k battles for training after filtering corrupted or extremely short games (exact count depends on download and storage).
- **Observation view:** first-person reconstructed states via Metamon bridge (`src/pokemon_thesis/metamon_bridge/`).
- **Labels:** actions from human moves; belief labels from full-state opponent attributes.

### 8.2 Data splits (core of combinatorial evaluation)

Three complementary split strategies. Report all of them.

**Split A — Random battle split (in-distribution).** 80/10/10 train/val/test by battle ID. Measures standard imitation and belief accuracy.

**Split B — Held-out team cores (combinatorial).** Build a canonical key per team from the six species names (sorted). Hold out entire team keys that appear rarely or belong to a designated OOD set. Train only on ID teams. Test on battles where the opponent team key was never seen in training.

**Split C — Held-out set combinations.** Hold out specific tuples of (species, item, ability, move quartet) that appear in opponent sets. Tests fine-grained combinatorial generalization beyond species identity.

**Split D — Temporal split (optional robustness).** Train on earlier ladder period, test on later period if metadata allows.

### 8.3 Belief evaluation metrics

For each hidden-attribute type and each model

- **Top-1 accuracy** on masked unrevealed slots.
- **Top-3 accuracy** where label space is large (moves, items).
- **Calibration (ECE)** on belief probabilities.
- **Brier score** for multi-class predictions.
- **Reveal-turn anticipation.** How many turns before official reveal does top-1 become correct?

Plot belief accuracy vs turn number to show whether the model narrows uncertainty over time as humans do.

### 8.4 Policy evaluation metrics

- **BC cross-entropy and top-1 action match** vs human actions on each split.
- **Offline policy value estimates** if Q-learning is used (CQL Q-values on held-out states).
- **Live win rate** on poke-env against fixed opponents (RandomPlayer, simple heuristic, frozen BC bots).
- **Generalization gap.** ID metric minus OOD metric for each model. Primary summary statistic for RQ3.

### 8.5 Ablations

| Ablation | What it tests |
|----------|----------------|
| Context window $N \in \{4, 8, 16, 32\}$ | How much history belief needs |
| Remove within-turn self-attention | Value of 13-token structure |
| Remove causal mask (bidirectional) | Whether future leakage hurts policy learning |
| $\lambda_{\text{bel}} = 0$ vs tuned | Necessity of explicit belief loss |
| Belief fusion vs shared trunk only | Whether policy needs explicit belief vector |
| CQL vs AMAGO vs BC-only | Offline RL contribution |

### 8.6 Statistical reporting

- Report mean ± 95% bootstrap confidence intervals over battles.
- Paired comparison on the same test battles between proposed model and each baseline.
- Significance via paired bootstrap or McNemar for win-rate comparisons.

### 8.7 Minimum viable result (thesis pass bar)

The thesis is successful if at least two of the following hold on combinatorial splits.

1. Belief-augmented TurnTransformer beats all belief baselines B1–B4 on aggregate hidden-species prediction.
2. P4 beats P3 on OOD team split policy metrics with a smaller generalization gap.
3. P5 improves live win rate over P3 without search at inference.
4. Ablations confirm that causal history modeling contributes meaningfully (B6 or P1 vs P4).

---

## 9. Implementation Roadmap

| Phase | Work | Status |
|-------|------|--------|
| 1 | Environment, logging, poke-env baseline battles | In progress |
| 2 | 13-token encoder and vocab | Implemented |
| 3 | TurnTransformer BC training | Implemented |
| 4 | Metamon download and bridge to token format | Started |
| 5 | Belief label extraction from Metamon full state | Planned |
| 6 | Belief heads and multi-task training | Planned |
| 7 | Combinatorial split tooling | Planned |
| 8 | Baselines B1–B6 and P1–P5 | Planned |
| 9 | Offline RL fine-tuning (CQL or AMAGO) | Planned |
| 10 | Evaluation harness and thesis figures | Planned |

---

## 10. Expected Contributions

1. A **structured tokenization and belief supervision framework** for partially observed simultaneous-action games.
2. **Empirical evidence** on whether transformer history models learn calibratable beliefs over hidden opponent attributes from offline human play.
3. A **combinatorial generalization benchmark protocol** (team-key and set-combination held-out splits) with belief and policy metrics tied together.
4. **Open-source implementation** building on Metamon, poke-env, and Gymnasium-compatible tooling for reproducibility.

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Metamon label noise in first-person reconstruction | Manual spot checks, filter corrupted battles, compare random-bot vs human stats |
| Belief labels trivial after reveal | Mask revealed slots, focus metrics on pre-reveal turns |
| Compute limits | Start with subset of Gen 9 OU, smaller d_model, shorter context |
| Offline RL instability | Treat CQL/AMAGO as Stage B; thesis still valid with strong BC + belief results |
| Action space complexity | Start with simplified action encoding, expand after pipeline is stable |
| Reviewers see Pokémon as non-serious | Frame paper around POMDPs, belief inference, and OOD generalization; Pokémon in experiments only |

---

## 12. Thesis Outline (document structure)

1. Introduction
2. Background (POMDPs, transformers, offline RL, combinatorial generalization)
3. Benchmark Environment (brief Pokémon / Metamon description)
4. Representation and Belief Targets
5. Model and Training
6. Experimental Setup and Baselines
7. Results (belief, policy, generalization, ablations)
8. Analysis and Interpretability (including tree-based feature study)
9. Limitations and Future Work
10. Conclusion

---

## 13. One-Paragraph Proposal Pitch (for committee or abstract)

Partially observed, simultaneous-action decision problems appear in security, negotiation, and multi-agent settings, yet most learning benchmarks are fully observed or turn-based. This thesis proposes a causal transformer that infers hidden opponent attributes from interaction history and uses those beliefs for offline policy learning. We evaluate on large-scale human competitive game trajectories with combinatorial held-out team and set splits. The goal is to show that explicit belief modeling with structured tokenization improves both interpretable state inference and robust policy performance under distribution shift, without relying on lookahead search at inference time.

---

## References (starter list)

- Vaswani et al., *Attention Is All You Need*, 2017
- Kaelbling et al., *Planning and Acting in Partially Observable Stochastic Domains*, 1998
- Heess et al., *Opponent Learning Awareness*, 2017
- Chen et al., *Decision Transformer*, 2021
- Kumar et al., *Conservative Q-Learning for Offline RL*, 2020
- Grigsby et al., *Metamon* (UT Austin RPL), 2023–2024
- poke-env documentation and Pokémon Showdown replay format

---

*Document version 1.0 — aligned with existing `TurnTransformer`, 13-token layout, and Metamon data pipeline.*
