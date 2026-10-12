# Execution verification

**Project supervisor: Dr. Zohreh Kaheh.**

The complete documented pipeline completed: raw-table preparation, training-package audits, the self-hazard baseline search and seeded training, all 30 hypergraph fits, all 30 comparison-model fits, both checkpoint freezes and both held-out evaluation stages. The supplied seeds, search configurations, epoch limits and early stopping were retained. All 131 checks across the eight generated audit reports passed.

An independent check confirmed finite, bounded prediction probabilities and recomputed average precision and ROC-AUC from both final prediction tables, matching the generated comparison tables. The released Python sources have the same parsed executable structure as the executed files; the locked protocol JSON is byte-identical. The required residual-hazard helper is included as an imported dependency; its standalone training entry point is outside the verified path.

Supply-chain execution used Python 3.12.14 and PyTorch 2.14.1+cpu with one CPU thread. No training outputs, fitted checkpoints or research metrics are distributed.

Verification covers the documented execution path and supplied inputs; it is not certification of research findings or all possible inputs.

## Private execution receipts

The logs remain private because they contain research outputs. Hashes identify the completed local runs; they are not independent external certification.

- `stage0.log`: `acf5e34e3d48024db5c03e2401d955424b961abf6f6c5a5c240c452a32aab4d4`
- `stage1.log`: `3d173863f533ac1c6c900a268b251236d53b5a3d14a41b33e7ec87c5f661fbc5`
- `stage2.log`: `4b2c208a55cba7e874a42c92ca4ed4d29e1ab8d78e303e45804474480d218871`
- `stage3b.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `stage3e.log`: `4619d4ef534f95b06bfc54153b0a7913625f3a30ba0ca5894343357b13272b52`
- `stage3c.log`: `562217b32ab907e3e12a2a754a1b335356ea7daee5e46e19a86829da4c09a27e`
- `stage3f.log`: `fbe5be03ba17d0139d2e9032ef0309892412f91f8add4aa651e2b4051d42709a`
- `stage3d.log`: `1544a76efc385f0d5a6e9ddac5c1a70deddada451b208b551b38d922b45fdac2`
- `stage3g.log`: `02ff61d1a5fccd780e15c762e1a6f580a5990e8b089323c7302da51280eaf275`

## Documentation review

Repository documentation was reviewed on 11 October 2026 for consistency with the included source files and recorded execution scope. This editorial review did not change executable code or rerun training. `SHA256SUMS.txt` records the current file contents.

## Rights and citation documentation — 12 October 2026

A documentation-only review added the reuse-status notice, contributor-rights register and software citation, and clarified the contribution account. Executable sources, dependencies and configuration are unchanged from commit `278ce6c0003876555cd6e39de0a68ca88bc1e321`. No training or scientific validation was rerun for this update. Existing execution receipts do not establish publication or licensing authority. `SHA256SUMS.txt` was refreshed for the current files, excluding itself.

The subsequent clarification on 12 October 2026 records publication-only permission and the maintainer's acceptance of standard notice retention with a scholarly citation request. MIT remains conditional and was not added. This follow-up changes documentation and its checksums only.

DOI preparation added a BibTeX export and a Zenodo metadata/rights record on 12 October 2026. At that stage, these were preparation documents and no deposit or DOI had been created. Source code and the existing CFF citation were unchanged; documentation checksums were refreshed.

## DOI publication — 12 October 2026

Archived snapshot `c78a6a50e8a969c746d52740c821486357f18eab` was published as `snapshot-c78a6a5` in [Zenodo](https://zenodo.org/records/23312621), DOI `10.5281/zenodo.23312621`. The public record, both creator names, archive checksum and custom rights notice were checked. Citation files and README now identify this archived version. The original archive retains its pre-DOI citation files. Source code, dependency pins and scientific execution scope are unchanged; no training was rerun. See [ZENODO_DEPOSIT.md](ZENODO_DEPOSIT.md). Documentation checksums were refreshed.
