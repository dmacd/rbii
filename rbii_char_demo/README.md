# Resource Bounded Incremental Induction (RBII) — character-level demo

This repository is a **small, modular, runnable** Python demo of the algorithmic ideas from
**Resource Bounded Incremental Induction (RBII)** as described in *Engineering Principles for Continual Learning*.

The demo is designed to be:

- **Online**: predicts the next character, observes the true character, updates weights and state.
- **Continual**: supports recurring tasks/episodes and measures **reacquisition delay** on task returns.
- **Resource-bounded and modular**:
  - swappable freezing policy
  - swappable newborn weight assignment policy
  - swappable memory mechanism (including state-indexed recall)
  - extensible DSL primitive library (register new Python functions easily)
- **Threaded**: uses `ThreadPoolExecutor` for predictor evaluation, transformer execution, and validation.
  - This benefits most from **free-threaded CPython (PEP 703 / "no GIL")** builds, but runs fine on normal CPython too.

## Quickstart

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run all scenarios

```bash
python run_demo.py
```

Outputs are written to `outputs/`:

- `scenario_*_cumulative_loss.png`
- `scenario_*_cumulative_compression_gain.png`
- `scenario_*_reacquisition_delay.png`
- `scenario_*_reacquisition_excess_loss.png`
- `scenario_*_metrics.json`

## Dr. Seuss text

Dr. Seuss works are copyrighted. This demo **does not bundle** any Dr. Seuss text.
If you have a text corpus you are legally allowed to use, place it at:

`data/dr_seuss.txt`

If that file is not present, the demo falls back to a short synthetic placeholder.

## Where to customize

- `freezing_policies.py`
- `newborn_weight_policies.py`
- `memory_mechanisms.py`
- `primitive_library.py` (add DSL primitives here)
- `transformer_programs.py` (add transformers here)

## Notes on the 4 test-case scenarios

The demo constructs 4 scenario streams inspired by the paper:

- **A: recurring finite task family (context switching)**  
- **B: compositional curriculum (small edits between tasks)**  
- **C: rare / low-value regularities (glimpses)**  
- **D: non-recurrent drift (no tasks return)**

See `test_scenarios.py` for the stream construction.
