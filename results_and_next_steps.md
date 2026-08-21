# Results and Next Steps

## ResNet-1D: complete, verified, this is the real Table 3 entry

Full 200-epoch run, real HPC hardware, Chen's exact confirmed parameters throughout (not the earlier default-hyperparameter version). Best checkpoint at epoch 177 (val macro F1 0.9434), saved to `~/ecg_work/outputs/resnet1d_real_best.pt`.

**Final test-set result:**

| Metric | Reproduced | Chen reported | Difference |
|---|---|---|---|
| Macro F1 | 0.9499 | 0.9467 | +0.0032 |
| Hamming loss | 0.0069 | 0.0069 | +0.0000 |

This is a genuinely close reproduction, not an approximate one. Worth noting for the thesis writeup: training loss reached near-zero (0.0005-0.001) in the final ~25 epochs while validation F1 occasionally dropped sharply (e.g. 0.9434→0.5773 at epoch 180, recovering within a few epochs) before settling. This is a real, expected consequence of Chen's own fixed learning rate with no schedule across 200 epochs, not a bug, the checkpointing-on-improvement approach correctly protected the best epoch (177) through this instability. Worth mentioning explicitly if asked why the training curve isn't smooth.

**Earlier, pre-paper-correction attempt for comparison** (CPU only, old default hyperparameters, 1500-sample windows instead of 300, wrong loss formula, wrong architecture width): topped out around test macro F1 0.36 (0.3568 baseline, 0.3626 with per-class threshold tuning). This was an honest result given incomplete information at the time, not an error, worth keeping as a "before/after having the real paper" data point if useful for the thesis narrative, but the 0.9499 result above is the one that actually goes in Table 3.

## Not yet done

- **BiLSTM**: not yet trained on real hardware. Code is ready (`models.py`, confirmed correct output shape). Same notebook pattern as ResNet-1D will work directly, just change `build_model("resnet1d", ...)` to `build_model("bilstm", ...)`.
- **Transformer**: same status, code ready, not yet run.
- **Mamba-2**: running on the pure-PyTorch fallback layer, not the official CUDA kernel (see TECHNICAL_STATE.md for why). Usable as-is if needed, but won't match Chen's exact reported numbers as closely as the other three likely will, since it's not the same implementation he used.
- **Section 4 (Adoption stage)**: patient/record-level split version, full 14,190-record taxonomy instead of the 3,716-subset, not yet started for any architecture. This is genuinely next after all four architectures have a Section-3 reproduction number.
- **Adaptation/Extension (age-band study)**: not started, this is the thesis's actual novel contribution, comes after Section 4.

## Open, lower-priority threads

- Email to Chen drafted (two variants, concise and detailed), not sent. His direct email not found; his likely co-author/supervisor Feng's is confirmed (fengzhenghui@hit.edu.cn). Worth sending regardless of the strong reproduction result, could still explain the residual +0.0032 gap and would help with BiLSTM/Transformer/Mamba-2 if those don't reproduce as cleanly.
- GitHub repo (`tayy00/pECG-Classification`) needs the README/.gitignore/requirements.txt just drafted, not yet pushed as of this writing, and the large CSV + checkpoint currently sitting in git history should ideally be removed (not just excluded going forward) if repo size becomes a problem later.
- Two systematic-review papers had pending full-text confirmations (Mayourian infant-CHD article type, Liu et al. 2018 PVC metric); Jonathan confirmed these resolved via his own UW access, both included, no further action needed.
- A possible newer/renamed version of PEACE exists (arXiv:2607.15928, "Knowledge-Guided Cross-Modal Fusion..."), not confirmed whether it's a direct revision of the 2605.00647 version already cited. Worth a direct check before finalizing Week 3 references if this becomes relevant again.

## If resuming after this conversation is compacted

Read this file plus `PROJECT_CONTEXT.md` and `TECHNICAL_STATE.md` first. The single most important thing not to relitigate: **`torch.backends.cudnn.enabled = False` is required every fresh session** before building any model on this cluster, it's not saved anywhere automatically. Second most important: don't re-derive Chen's parameters from scratch, they're fully captured in TECHNICAL_STATE.md, confirmed directly from the actual paper, not the earlier placeholder defaults.
