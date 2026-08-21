# Project Context: MPhil Pediatric ECG Thesis

Reference document for resuming this project in a new conversation. Written to preserve specific, hard-won detail, not just a general summary.

## Identity

Jonathan Kalami (jkalami1@st.knust.edu.gh), MPhil Biomedical Data Science (DSCHANGE), National Institute for Mathematical Sciences Ghana, KNUST. Thesis: "Automated Classification of Pediatric Electrocardiogram Abnormalities Using a Deep Learning Framework." Timeline: mid-July to end of September 2026, tight. Supervised through the DSCHANGE program; LaTeX Beamer template style from colleague Tayyiba Amartey (GitHub: tayy00).

Core design: reproduce and extend Chen et al. (arXiv:2510.03780) and PEACE/Liu et al. (arXiv:2605.00647), four-stage methodology: **Reproduce → Adopt → Adapt → Extend**, culminating in an age-stratified domain-shift analysis on pediatric ECG classification as the thesis's actual novel contribution.

## Research plan status (5-week structure, confirmed against the actual plan document)

| Week | Content | Status |
|---|---|---|
| 1 | Define Review Focus + Protocol | Complete |
| 2 | Literature Search, Screening, Extraction, Quality Assessment, Synthesis | Complete. 516→358→39→17→16 included (1 excluded: Rahman, bradycardia alarm correction, not diagnostic classification). Systematic review report written. Evidence synthesis matrix built. |
| 3 | Selection, Critical Analysis, Implementation Planning for Two Key Papers | Complete. Chen (2025) and PEACE/Liu (2026) selected, neither in the 16-study review (they're methods papers on the target dataset, not pediatric-population studies, different inclusion logic). Full technical teardown, comparison matrix, reproducibility assessment, research gap statement, and Reproduce/Adopt/Adapt/Extend categorization all done. |
| 4 | Methodology and Implementation | Methodology fully written (all 4 categories). **Implementation: ResNet-1D complete and verified (see Results section below). BiLSTM, Transformer not yet run. Mamba-2 on fallback layer.** |
| 5 | Presentation of Results | Not started, depends on Week 4 finishing all four architectures. |

## Key methodological decisions, with reasoning (don't re-litigate these)

- **Chen's subset identification**: a record belongs to Chen's 3,716-record CVD subset if any of its ICD-10 codes (semicolon-separated in `AttributesDictionary.csv`) match one of 19 target codes in `DiseaseCode.csv` (which has 20 rows, but the 20th, "Other diseases," is a catch-all with no real code, excluded). Codes need prefix-stripping (`(F)`, `(V)`, `(FO)`, `(OSD)` etc. removed) before matching, since `AttributesDictionary`'s own codes don't carry these subtype prefixes. This logic, run against the real files, selects **exactly 3,716 records**, confirmed twice, matching the paper's reported figure precisely. Collapses to **16 distinct label columns** (not 19), since four of the 19 DiseaseCode.csv rows share a base code with another row.

- **Reproduction (Section 3) vs. Adoption (Section 4) split methodology, deliberately different, don't merge them**: Section 3 uses Chen's own **window-level split** (individual 3-second windows assigned to train/val/test, a given recording's windows CAN and do land in different splits), reproducing his method exactly, leakage included, he names this himself as a limitation in his own paper ("may not fully guarantee subject-level independence... reported numbers may partially reflect optimistic estimates," and lists patient-level splitting as explicit future work). Section 4 onward uses **record-level split** (every window from a record stays together), the actual correction. Both implemented as genuinely separate functions in `dataset.py`, tested to confirm they produce different behavior (window-level split showed 17/49 records with cross-split leakage in a synthetic test; record-level split showed zero overlap).

- **Split ratio**: Chen specifies 7:1:2 (train:val:test), confirmed directly from his paper. Implemented via multi-label stratified sampling (`iterative-stratification` package, `MultilabelStratifiedShuffleSplit`), not plain random split, since a single-category stratify can't properly balance 16-19 co-occurring binary labels at once.

- **Why Chen and PEACE specifically, not papers from the 16-study review**: the review's inclusion criteria required a pediatric-population study; Chen and PEACE are methods/benchmark papers on the target dataset instead, a different, correct selection logic for "what to reproduce and extend" versus "what the existing pediatric AI-ECG literature shows."

- **Research gap** (for thesis framing): established across the review that DL classifies pediatric ECG well (settled, not the contribution) and that performance varies with age where anyone checks (Siontis, Ghelani, both in the 16-study review) and that adult-trained models underperform on children (Anjewierden, in the review; PEACE, quantified precisely). No study, in the review or in Chen/PEACE, breaks this down systematically across pediatric age bands. That gap is the thesis's actual contribution.

## Files delivered so far (see separate TECHNICAL_STATE.md for the code specifically)

Systematic review report, MPhil proposal draft, dataset landscape/gap analysis, literature tracker (Excel, 4 sheets: studies, quality assessment, evidence synthesis, methods references), Week 1-4 LaTeX Beamer progress decks, Week 3 paper-analysis deck, Week 4 methodology deck (numbered equations, standing rule for all future decks), progress-update decks (ResNet-1D results, BiLSTM concept/math), the four-architectures-explained and ResNet1D-explained-beginner reference documents, a draft email to Chen requesting code/hyperparameters (not yet sent, his real email not yet found, only Feng's, a likely co-author/supervisor contact, fengzhenghui@hit.edu.cn, confirmed correction: Chen's actual institution is Harbin Institute of Technology, Shenzhen, not Beijing Institute of Technology as earlier stated in error).
