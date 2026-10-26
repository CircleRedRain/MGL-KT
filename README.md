# MGL-KT

Training and evaluation code for **Modeling Progressive Cognition: A Multi-Granular Learning Framework for Knowledge Tracing**.

## Installation

Use Python 3.10 or later and install dependencies:

```bash
pip install -r requirements.txt
```

## Data and configuration

The default dataset is ASSIST2017. Obtain its released interaction CSV and place it at:

```text
data/assistments17/anonymized_full_release_competition_dataset.csv
```

The concept relation table is included at `relations/assist2017/skill_relationships.csv`.
It supplies both concept names/IDs and typed concept relations. Raw student responses,
processed caches and checkpoints are not distributed.

Edit `src/common/config.yaml` to select ASSIST2017, Eedi34 or XES3G5M, relative
input paths, device, batch size and training parameters. Eedi34 and XES3G5M input
files and relation tables must be supplied separately. All paths resolve relative
to this directory. `random_seed` controls the student partition and default training
seed. Optional `training_seed` changes training randomness while retaining the partition.

## Training

```bash
python training.py --device cuda
python training.py --device cuda --resume
```

The data handler builds compact caches automatically on first use. Training selects
the checkpoint with the highest validation AUC and writes checkpoints under
`checkpoints/<dataset>/seed<seed>/full/`. Use `--config path/to/config.yaml` for a
separate configuration. Use `--device cpu` to run on CPU.

## Validation and test

```bash
python eval.py --split validation --device cuda
python eval.py --split test --device cuda --result-file outputs/test_results.json
```

Use `--checkpoint path/to/best.ckpt` to evaluate a specific checkpoint from this
package. Evaluation reports response-level AUC, accuracy and RMSE. Multi-concept
candidates share the pre-response history and their probabilities are averaged.
Padding is excluded and actual responses are counted once.

## Model

The model learns a concept–ability–question graph, prunes question–ability
assignments, independently encodes three histories with GRUs, retrieves causal
history through target-conditioned attention, and applies concept–ability–question
residual refinement. Training uses the final response prediction loss. Graph
structure freezes when its configured stability condition is satisfied.

## Layout

- `training.py`: training and validation-based checkpoint selection.
- `eval.py`: validation or test evaluation.
- `src/common/`: configuration, paths and compact record views.
- `src/data_handler/`: raw-data processing and sequence loading.
- `src/models/`: the main model and training objective.
- `src/evaluation.py`: evaluation implementation.
- `relations/assist2017/`: concept relation table.
