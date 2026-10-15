# soccer-temporal-lags

Experimentation framework for **1X2 football match outcome prediction** (home win / draw / away win) from
temporal *lag* features of past matches. It accompanies a journal paper (currently under review) and contains
the code used to train and evaluate all the models compared there:

- MLPs on multi-window temporal lags (our approach),
- gradient boosting (XGBoost / LightGBM) on the feature set of Hubáček et al. (pi-ratings, PageRank, match importance, short/long-term form),
- gradient boosting on the Berrar et al. rating features,
- TabNet and LightGBM variants on the same feature families.

Every experiment is a YAML file placed in a queue folder. A runner picks them up by priority, tunes the
hyperparameters with Optuna on a temporal validation window, retrains the best configuration and stores the
predictions on the test season.

## Repository structure

```
soccer-temporal-lags/
├── src/
│   ├── main.py              # entry point / queue runner: reads _pending/, runs experiments, moves YAMLs to _done/
│   ├── utils.py             # I/O, lag computation, RPS, config validation
│   ├── data_filters.py      # configurable data filters (e.g. leagues)
│   ├── data/                # datasets: temporal train/val/test split + feature engineering
│   │   ├── base.py          #   abstract Dataset, TorchData
│   │   ├── own.py           #   multi-window temporal lags
│   │   ├── top1.py          #   Hubáček et al. features (pi-ratings, PageRank, match importance, form)
│   │   ├── owntop1.py       #   lags + Hubáček et al. features
│   │   ├── pi_ratings.py    #   pi-ratings
│   │   └── ratings_berrar.py#   Berrar et al. ratings
│   ├── ml/                  # models and their Optuna training loops
│   │   ├── experiment.py    #   abstract Experiment (tuning, metrics/logits saving)
│   │   ├── mlp.py           #   MLP (PyTorch Lightning)
│   │   ├── mlp_experiment.py#   MLP training/tuning
│   │   ├── boosting.py      #   XGBoost / LightGBM wrapper
│   │   ├── gradientboosting.py # gradient boosting training/tuning
│   │   └── tabnet.py        #   TabNet training/tuning
│   └── experiments/         # concrete experiments = feature family (data/) + model family (ml/)
│       └── own_original.py, own_top1.py, gb_own.py, gb_top1.py, gb_ratings.py, tabnet_*.py
├── scripts/
│   └── set_configs.py       # helper to create/modify experiment YAMLs
├── _pending/                # queue of experiments to run (YAML files)
├── _done/                   # experiments that finished successfully
├── logs/                    # outputs (created/filled at run time)
│   ├── models/              # one folder per run: config, best hyperparameters, checkpoints
│   └── metadata/            # TensorBoard logs of the MLPs
└── datasets/
    └── raw_datasets/        # input data (not distributed, see below)
```

## Installation

Python 3.8 (tested with 3.8.8).

```bash
conda create -n soccer python=3.8
conda activate soccer
pip install -r requirements.txt
```

Install PyTorch beforehand if you need a specific CUDA build (<https://pytorch.org/get-started/locally/>).

## Data

The data are **not included**. Match results, statistics and odds come from
[Football-Data.co.uk](https://www.football-data.co.uk/) (E0, SP1, SP2, D1, I1 and F1, from season 2000/01).
The framework expects the following files (paths are configurable in each YAML, relative to the project root):

| File | Used by | Content |
|---|---|---|
| `datasets/raw_datasets/datalake.csv` | all | one row per match |
| `datasets/raw_datasets/dtypes.json` | all | pandas dtypes used to read `datalake.csv` |
| `datasets/raw_datasets/page_rank*.csv` | `*top1` experiments | PageRank of both teams per match |
| `datasets/raw_datasets/match_importance*.csv` | `*top1` experiments | match importance features per match |

**`datalake.csv`**: `;` separator, `,` decimal, dates as `dd/mm/YYYY`, sorted by date. Columns:

```
matchId;id_H;id_A;Div;div_order;Date;Weekday;season;HomeTeam;AwayTeam;FTHG;FTAG;FTR;Attendance;Referee;
HS;AS;HST;AST;HF;AF;HC;AC;HY;AY;HR;AR;HO;AO;HHW;AHW;B365H;B365D;B365A;MaxH;MaxD;MaxA;AvgH;AvgD;AvgA
```

`matchId` is a unique integer, `id_H`/`id_A` are integer team ids (consistent across leagues and seasons),
`Div` is the Football-Data league code, `season` a season label (e.g. `T17-18`) and `FTR` the result
(`H`/`D`/`A`). The remaining columns follow the [Football-Data notes](https://www.football-data.co.uk/notes.txt).

**`dtypes.json`**: mapping from column to pandas dtype, e.g.

```json
{"matchId": "Int64", "id_H": "Int64", "id_A": "Int64", "Div": "str", "season": "str",
 "HomeTeam": "str", "AwayTeam": "str", "div_order": "Int64", "FTHG": "Int64", "FTAG": "Int64",
 "FTR": "str", "Attendance": "float", "Referee": "str", "Weekday": "Int64"}
```

**`page_rank*.csv`**: `matchId;pagerank_H;pagerank_A` (`page_rank_lags` variant: `matchId;pagerank_<lag>_...`).
**`match_importance*.csv`**: `matchId;rounds;top1_H..top7_H;down1_H..down5_H;top1_A..top7_A;down1_A..down5_A`.
All features must be computed only with information available **before** the match.

The project root is the first folder above `src/` that contains `datasets/` (i.e. this repository). To keep
the data elsewhere, set the environment variable `TFG_ROOT` to a folder containing `datasets/`.

## Experiment configuration (YAML)

```yaml
experiment: own_original          # experiment type (see table below)
experiment_id: own_original_simp_ss_2018
description: MLP with basic temporal lags
enable: true                      # disabled YAMLs are skipped
queue: queue                      # queue name: the runner only executes YAMLs of its queue
priority: 3                       # higher runs first (default 2)
paths:
  raw_data: datasets/raw_datasets/datalake.csv
  dtypes_path: datasets/raw_datasets/dtypes.json
  data: datasets/data_train/data_  # prefix used when save_data: true
  # page_rank / page_rank_lags / match_importance for *top1 experiments
  # logger: ./logs/                # TensorBoard folder (TabNet)
data:
  test_date: '2018-01-01'         # test = from test_date...
  test_season: true               # ...to the end of the season being played on test_date
  val_date: '2014-01-01'          # validation = [val_date, test_date), train = before val_date
  sample: sequential              # sequential | random (random: validation drawn from the train period)
  filters:                        # optional; stage 'split' filters after computing features
  - {name: leagues, values: [E0, SP1, D1, I1, F1], stage: split}
  # model-specific options, e.g. for own_* experiments:
  lags: [15D, 30D, 60D, 180D, 365D, 730D, 1825D]   # rolling windows
  min_samples: [1, 2, 2, 8, 10, 15, 15]            # min matches per window
  aggregations: {FTG: mean, FTG_rival: mean, Points: mean, Side: mean}
  col_group: [idTeam]
  col_window: Date
  num_feats: [4, 50]              # optional: Optuna explores ANOVA/PCA reduction to 4-50 dimensions
  factor: true                    # optionally weight labels by goal difference
  balance: false                  # oversample draws/away wins in training
training:
  iterations: 5000                # max Optuna trials
  timeout: 6850                   # minutes
  direction: maximize             # must match eval_metric (accuracy -> maximize, rps -> minimize)
  eval_metric: val_accuracy
  max_epochs: 100
  patience: 15
  batch_size: [128, 256, 512]
  cuda: true                      # false hides the GPUs
```

Test split options: `test_season: true` (or `test_range: season`) tests on one whole season,
`test_range: year` on a calendar year and `test_end_date` sets an explicit end. The configuration is validated
before running (mandatory sections, metric/direction consistency, date order, filters...).

Supported `experiment` values:

| `experiment` | Features | Model |
|---|---|---|
| `own_original` | multi-window temporal lags | MLP |
| `own_top1` | temporal lags + Hubáček et al. features | MLP |
| `lgbm_top1` | Hubáček et al. features (pi-ratings, PageRank, match importance, form) | XGBoost |
| `lgbm_owntop1` | temporal lags + Hubáček et al. features | LightGBM |
| `lgbm_own` | temporal lags | LightGBM |
| `ratings_berrar` | Berrar et al. ratings | XGBoost |
| `tabnet_own` / `tabnet_owntop1` / `tabnet_paper` | lags / lags + Hubáček / pi-ratings | TabNet |

## Running

Put the YAMLs in `_pending/` and launch the queue:

```bash
python src/main.py <queue_name> [env]
```

- `queue_name` (default `default`) selects the YAMLs whose `queue` field matches, so several runners can
  share the folder.
- `env` (optional) uses `_pending<env>/` and `_done<env>/` instead (e.g. `dev`), if they exist.
- Each experiment runs in its own process. When it finishes successfully the YAML is moved to `_done/`.
  Failures are logged in `logs/errors.log` and the runner continues with the next YAML.
- The runner keeps polling the folder and stops after a few empty checks.

The script can be launched from any folder: outputs always go to the repository `logs/`.

### Outputs

| Path | Content |
|---|---|
| `logs/<experiment_id>_metrics.csv` | one row per Optuna trial: hyperparameters, validation metric and test metric of the best trial |
| `logs/<experiment_id>_logits.csv` | test predictions: match, date, league, teams, goals, label, predicted class and probabilities (`draw`, `home`, `away`) |
| `logs/models/<experiment_id>_<timestamp>/` | run config, best hyperparameters and best model |
| `logs/metadata/` | TensorBoard logs (MLP) |
| `logs/errors.log` | traceback of failed experiments |

CSV outputs use `;` as separator and `,` as decimal mark.

## Creating configurations: `scripts/set_configs.py`

Creates or modifies YAMLs; the new file is written in the current folder with a name derived from its
configuration (or `-n`).

```bash
cd _pending
# change fields (nested with dots); -r removes the original file
python ../scripts/set_configs.py my_exp.yaml -f data.test_date training.timeout -v 2019-01-01 3600 -s 2019
# set a league filter
python ../scripts/set_configs.py my_exp.yaml --filter leagues:E0,SP1,D1,I1,F1
# quick test version: 2 trials, 2 epochs, 64 samples, queue 'test'
python ../scripts/set_configs.py my_exp.yaml -t
python ../src/main.py test
```

Options: `-f/-v` fields and values, `-d` fields to drop, `-a` list fields to extend, `-s` suffix,
`-n` name (`true` keeps the file name), `--filter NAME:V1,V2[:STAGE]`, `--clear-filters`, `-t` test mode,
`-r` replace.

## Rolling-origin evaluation

To evaluate over several seasons, create one YAML per test season by shifting `test_date` and `val_date`
by one year (e.g. `test_date: 'YYYY-01-01'`, `val_date: 'YYYY-4-01-01'`), so all models share the same test
matches and an equally long validation window.

## License

[MIT](LICENSE)
