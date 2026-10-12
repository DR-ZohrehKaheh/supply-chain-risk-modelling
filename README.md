# Temporal and Graph Models for Supply-Chain Anomaly Ranking

A Python and PyTorch workflow for ranking product-level operational anomalies from time-series and relational data. The project compares temporal, graph and hypergraph models through a documented training, checkpoint-freezing and chronological evaluation process.

> **Reuse status:** No repository-wide software licence has been granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). The execution instructions below describe the workflow; they do not grant reuse permission.

**Project supervisor: Dr. Zohreh Kaheh.**  
**Undergraduate collaborator: Yuyang Zhang.**

Dr. Kaheh designed the project based on her earlier work, provided the initial design and implementation teaching, supervised the student's initial implementation, and subsequently developed the code further. This contribution description reflects the maintainer's account.

## Technical focus

- Preparation of temporal samples and relational model inputs.
- Comparison of recurrent, multilayer perceptron, graph and hypergraph models.
- Seeded training, validation-based model selection and checkpoint integrity checks.
- Chronological evaluation and block-bootstrap comparisons.

## Environment and inputs

Python 3.12 and the CPU build of PyTorch were used for verification. Install dependencies in an isolated environment. To select the CPU build explicitly:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Provide the [SupplyGraph](https://github.com/ciol-researchlab/SupplyGraph) source tables separately as `SupplyGraph-main/Raw Dataset/...` beside the scripts. The tested input snapshot is identified by upstream commit `0e6d010e256ee08e34ab6bef5a111ef1c758b32d`. Data are not distributed in this code-only release; preserve the upstream data terms when obtaining them.

Run from this directory, in order:

```bash
python 03b_train_genuine_hgnn_baselines.py --self-test
python build_supplygraph_disruption_events.py
python 01_build_raht_training_package.py
python 02_train_self_hazard_baseline.py
python 03b_train_genuine_hgnn_baselines.py
python 03e_train_credible_mlp_gnn_baselines.py
python 03c_freeze_hgnn_extension.py
python 03f_freeze_credible_baselines.py
python 03d_evaluate_hgnn_extension_test.py
python 03g_evaluate_credible_baselines_test.py
```

`03_train_raht_residual_hazard.py` is retained as a required imported dependency. Its legacy standalone training entry point is not part of the documented release path.

Training uses the configured search, seeds, epochs and early stopping. The CPU thread count defaults to one and can be changed with `PORTFOLIO_TORCH_THREADS`. This setting controls compute parallelism, not the number of training runs.

Generated folders, predictions and checkpoints must not be committed. Keep `00_raht_protocol_locked.json` unchanged during a run: downstream stages verify its hash. If configuration changes, rebuild the complete workflow from the first stage.

## Scope

The prediction target is a constructed operational-anomaly label. Results should be interpreted as exploratory anomaly-ranking comparisons. The held-out split had been used in earlier project work, so these runs do not constitute a new, untouched confirmatory evaluation. Operational disruption detection, risk calibration and causal effects have not been established. No model scores, research figures, paper text or previous fitted objects are included in this repository.

## Verification and contributions

See [VERIFICATION.md](VERIFICATION.md) for the tested execution scope and [ATTRIBUTION.md](ATTRIBUTION.md) for supervision and student contributions. This repository contains code and technical documentation only.

## Citation

The maintainer requests scholarly citation and retention of contributor credit. MIT is the intended future software licence after rights clearance; currently confirmed student-code permission covers public publication only. See [LICENSE_STATUS.md](LICENSE_STATUS.md#maintainer-clarification--12-october-2026). No open-source reuse licence is in force for the repository as a whole.

For authorised use in research, publications or presentations, please cite both contributors:

Kaheh, Z., & Zhang, Y. (2026). *Temporal and Graph Models for Supply-Chain Anomaly Ranking* [Computer software]. GitHub. https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling

Citation formats: [CITATION.cff](CITATION.cff) and [BibTeX](CITATION.bib). The current citation identifies code snapshot [`278ce6c`](https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling/tree/278ce6c0003876555cd6e39de0a68ca88bc1e321); cite the exact commit or released version you used. This citation credits the documented collaboration and does not assign copyright or constitute student approval of a licence.

No DOI has yet been assigned by this workflow. [Prepared Zenodo metadata](ZENODO_DEPOSIT.md) supports a DOI deposit with the current publication-only rights status. A future registered DOI will be added here and to the citation files. No unverified article DOI or affiliation has been added.
