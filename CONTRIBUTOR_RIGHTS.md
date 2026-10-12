# Contributor rights and permission record

Status: **pending**. Last reviewed: 12 October 2026.

This register records attribution separately from ownership and permission. It does not transfer copyright, certify title, or grant a licence. No entry below is an executed consent.

## Evidence register

| Contribution or material | Available evidence | Rights/permission status |
| --- | --- | --- |
| Dr. Zohreh Kaheh: earlier project basis, initial design, implementation teaching, supervision and subsequent code development | Maintainer's account, updated on 12 October 2026; ATTRIBUTION | Maintainer states publication authority. Exact pre-existing and later code scope, applicable employment rights and downstream licensing authority require documentation. |
| Yuyang Zhang: initial implementation developed under supervision | Existing ATTRIBUTION, based on the supervisor's account | Student's file-level confirmation and permission for the proposed licence are not recorded in the reviewed evidence. No assignment is assumed. |
| Subsequent edits and AI-assisted preparation | ATTRIBUTION explicitly notes incomplete per-person edit history | Map surviving contributions and provenance; human or institutional authorisation cannot be inferred from AI assistance. |
| Third-party code, dependencies and external inputs | Source/dependency review and README | See LICENSE_STATUS; compatibility and any copied-material permissions remain to be resolved. |

Source archive named in existing attribution: `ESM_1_Code_and_Verified_Results.zip`. The archive is not publicly included and was not reviewed for rights in this audit.

### Scope to map

Code/configuration paths requiring a contribution-to-rights mapping:

`build_supplygraph_disruption_events.py`, `01_build_raht_training_package.py`, `02_train_self_hazard_baseline.py`, `03_train_raht_residual_hazard.py`, `03b_train_genuine_hgnn_baselines.py`, `03c_freeze_hgnn_extension.py`, `03d_evaluate_hgnn_extension_test.py`, `03e_train_credible_mlp_gnn_baselines.py`, `03f_freeze_credible_baselines.py`, `03g_evaluate_credible_baselines_test.py`, and `00_raht_protocol_locked.json`.

Also account for technical documentation, wrappers, tests, dependency declarations and any earlier project code incorporated into these files. The paths above do **not** attribute every line to either named person. Preserve the original material and evidence privately; do not recreate a fictitious commit history.

## Clearance checklist

- [ ] Identify each copyright holder or authorised licensor, including any relevant institution; confirm the applicable student, employment, joint-programme, funding and project agreements.
- [ ] Tie each permission to repository URL, commit or file hashes, covered paths and exclusions, including earlier work and later edits.
- [ ] Confirm permission to publish on GitHub separately from permission to distribute under the exact proposed licence, including its commercial-reuse terms.
- [ ] Record the relevant student's contribution confirmation; where rights are held by an institution, obtain approval from its authorised representative as needed.
- [ ] Resolve third-party code/data provenance and licence compatibility. Preserve required upstream notices.
- [ ] Agree the accurate attribution, contributor names and copyright notices without making the supervisor sole owner by default.
- [ ] Decide whether standard notice retention plus a citation request meets the intended credit requirement; MIT does not impose mandatory academic citation.
- [ ] Record archival permission before a Zenodo deposit, then publish a scoped licensing decision.

Do not accept silence, assessment submission, a GitHub upload, academic coauthorship or supervision as a substitute for the required permission. Do not merge new code contributions without agreed contribution terms while licensing is unresolved.

## Proposed permission record — draft, not executed

Use a separate completed record for each relevant rights holder. A contributor can confirm facts about their work, but cannot grant rights held by their university, employer or another party.

- Record ID:
- Repository: https://github.com/DR-ZohrehKaheh/supply-chain-risk-modelling
- Contributor / rights holder:
- Capacity and basis of authority:
- Covered commit, paths and hashes:
- Pre-existing material and exclusions:
- Publication on GitHub: approved / not approved / limited (specify)
- Exact public software licence and authoritative text link:
- Archival deposit (including Zenodo): approved / not approved / limited (specify)
- Third-party material, licences and notices:
- Applicable institutional/contractual approval, or documented reason none is required:
- Agreed public attribution:
- Dated affirmative consent and private evidence location:
- Reviewer and review date:

Suggested wording **for review and affirmative completion**, if MIT is chosen:

> I confirm that I hold, or am authorised to license, the rights identified in the completed record above. For that specified material, I authorise Dr. Zohreh Kaheh to publish it in the named GitHub repository and distribute it under the standard MIT License linked in this record. I understand that MIT permits modification, redistribution and commercial reuse, subject to its notice condition, and does not require a scholarly citation for every use. This is a non-exclusive authorisation, not a transfer of copyright. It excludes the third-party material and rights expressly listed above. My decision about a Zenodo archival deposit is recorded separately.

Keep executed emails, agreements, signatures and private contact details outside the public repository. Publish only a minimal verified summary: record ID, covered paths/version, approved licence, approval date, attribution and any exclusions. Never mark a pending entry approved without supporting evidence. Any later licence change outside the recorded authorisation needs a fresh rights check.

See [LICENSE_STATUS.md](LICENSE_STATUS.md) for the audit and sources.
