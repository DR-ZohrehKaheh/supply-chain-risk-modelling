# Temporal and Graph Models for Supply-Chain Anomaly Ranking

[Run and inputs](#environment-and-inputs) · [Scope](#scope) · [Citation](#citation) · [Contributions](ATTRIBUTION.md) · [Reuse status](LICENSE_STATUS.md)

A Python and PyTorch workflow for ranking product-level operational anomalies from time-series and relational data. The project compares temporal, graph and hypergraph models through a documented training, checkpoint-freezing and chronological evaluation process.

> **Reuse status:** No repository-wide software licence has been granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). The execution instructions below describe the workflow; they do not grant reuse permission.

**Project design, supervision and subsequent development: Dr. Zohreh Kaheh.**  
**Initial implementation (undergraduate collaborator): Yuyang Zhang.**

The project builds on Dr. Kaheh’s earlier work. She provided the initial design and implementation teaching, supervised the student’s initial implementation, and developed the code further. See [ATTRIBUTION.md](ATTRIBUTION.md) for the maintainer’s contribution account and disclosed AI assistance.

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

If you refer to this work or use it with the necessary permission, please cite both contributors:

Kaheh, Z., & Zhang, Y. (2026). *Temporal and Graph Models for Supply-Chain Anomaly Ranking* (Version snapshot-c78a6a5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23312621

Download [CITATION.cff](CITATION.cff) or [BibTeX](CITATION.bib). The [archived version](https://zenodo.org/records/23312621) is `snapshot-c78a6a5`, from source commit [`c78a6a5`](https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling/tree/c78a6a50e8a969c746d52740c821486357f18eab). Cite the version or commit you used; later GitHub documentation may differ from the fixed archive. See [ZENODO_DEPOSIT.md](ZENODO_DEPOSIT.md) for the deposit record and checksums.

Please retain contributor credit. Citation and DOI registration do not grant reuse permission or transfer copyright; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Questions and corrections

For technical questions or documentation corrections, [open an issue](https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling/issues) with the version or commit, environment, command and a minimal error report. Do not include private datasets, credentials or unpublished results. Discuss the scope and permissions of proposed code contributions with the maintainer first; [CONTRIBUTOR_RIGHTS.md](CONTRIBUTOR_RIGHTS.md) describes the required record.

## Continuous checks

[GitHub Actions](https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling/actions/workflows/checks.yml) runs the existing dataset-independent checks on Python 3.12 and Ubuntu. This covers utility tests or synthetic native/model checks, not the complete data-dependent research pipeline or scientific conclusions. The workflow does not publish generated research outputs.
