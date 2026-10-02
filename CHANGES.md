# What changed after review, and why

Four independent reviewers graded the first version, each from a different angle. Together they put it at about 835 / 1000.

| Reviewer | Angle | Score |
|---|---|---|
| Strict reviewer | The task spec, item by item | 842 |
| Bug hunter | Correctness, with its own tests | 850 |
| Reproducibility | Rebuilt and re-ran the notebook | 860 |
| ML reasoning | Explanations and write-up | 790 |

None of them found a wrong value in the loss path itself. The shift, the masks, perplexity, parameter counts and the
chunked loss all checked out, and every README number matched the notebook. The deductions were about what the
measurements meant, a missing demonstration, an overstated explanation and the deliverable. This file lists each
finding, what was changed, and what the new run shows.

---

## 1. Item 4 measured an unseen token, not the boundary (largest issue)

**Finding (two reviewers, independently).**
* The model was trained on raw text that never contains `<|endoftext|>`. So at every seam the model saw an input
  token it had never been trained on, and the "seam loss" mostly measured that.
* The 218 kept end-of-document targets cost 14 nats each, which also inflated the "in-document" average.
* The packed documents were consecutive speeches from the same scene, so they were not unrelated.
* "(seam − in-doc) × seams / targets = the observed drop" was presented as a check, but it is an algebraic
  identity and can't fail.

**Changes.**
* **Training data now ends every speech with `<|endoftext|>`** (GPT-2 style), and the split is by speech, so no speech
  is in both train and validation. The model has seen 6,499 `<|endoftext|>` tokens in training.
* **Validation speeches are shuffled before packing**, so neighbours are unrelated, as in real packing.
* **Losses are reported by target kind**: in-document, end-of-document and seam.
* **The two-document demo is now in the README**, as the task asks, before the 64-sequence aggregate.
* **The identity is described as an identity.** It gives the size of the drop, not evidence for it.
* **Isolated attention now resets position ids per document.** Previously B kept its offset position ids, so
  "isolated" was not the same as "alone". The notebook now checks the two match (max logit difference 7e-6).

**Result.**
* End-of-document targets now cost 0.64 nats: learned and legitimate, and kept.
* Seam targets cost 6.69 vs 4.89 in-document. The difference is now about the boundary itself (whose speech comes
  next), not an unseen token.
* Isolated attention is still slightly worse (4.769 vs 4.762). The explanation is now backed by a fact: all
  validation speeches come from two plays (*The Taming of the Shrew* and *The Tempest*), and the model was trained
  with attention across boundaries.

## 2. The central warning was printed, not demonstrated

**Finding.** The task warns that a wrong shift can produce a beautiful loss curve. The first version only printed
the wrong-direction strings.

**Change.** A new section trains the same model with the shift reversed for 150 steps, with the same data, seed and schedule.

**Result.** The wrong-shift model reaches a *lower* training loss than the correct model (4.66 vs 5.21) with no error.
Under the correct harness it scores 8.81 nats (perplexity 6,730, against about 240). Its predictions, printed as strings,
show it learned to copy the previous token. A plot of both curves is saved as `wrong_shift.png`.

> Note: in 150 steps the wrong-shift loss is lower but not near zero. Copying the previous token needs a dedicated
> "previous-token" attention pattern that this small model is still learning, and only 28% of its top predictions are
> copies so far. The curve keeps diverging from the correct one, but this run doesn't claim it reaches zero.

## 3. The item 2 machine check was circular

**Finding.** It asserted `shifted_targets(tokens) == tokens[:, 1:]`, which compares a definition with itself.

**Change.** The batch now records the corpus offset each window was cut from. The check compares every target with
`corpus[offset + t + 1]`, which doesn't depend on the harness code. Result: 0 mismatches over 508 positions.

## 4. Part 2 overclaimed

**Findings.**
* "Strictly higher entropy" is not guaranteed in general.
* The plateau after step 450 coincides with the learning rate decaying to zero.
* The 0.07-nat cost came from one seed.
* Gradient clipping applied to the *summed* loss could bias the comparison.
* "Reported to help at scale" had no citation.
* "Behaves like a regulariser" was unsupported: a worse validation loss is not evidence of regularisation.

**Changes.**
* The entropy argument is stated precisely: H(t+2 | ≤t) ≥ H(t+2 | ≤t+1) = H(t+1 | ≤t) for a stationary source, with
  equality only if t+1 carries no information about t+2.
* The plateau/learning-rate coincidence, and the capacity confound (head 2 is one MLP block), are stated as things this run can't separate.
  The gap plot now overlays the learning rate.
* **Two seeds** for both single-head and two-head models.
* Clipping frequency and pre-clip gradient norms are logged and reported.
* The paper is cited: Gloeckle et al. (2024), *Better & Faster Large Language Models via Multi-token Prediction*.
* The "regulariser" sentence was removed.

**Result.**
* The cost holds across seeds: +0.071 and +0.091 nats, against a seed-to-seed spread of 0.0007 for the single-head model.
* Clipping is active on 99% of the two-head model's late steps, against 1% for the single-head model. This is reported
  as a caveat. Adam's scale invariance makes it an unlikely explanation, but it was not tested.

## 5. Item 5 and item 3 explanations were vague

**Changes.**
* **Item 5.** The notebook now measures the logit variance at initialisation: 0.0512, exactly d · 0.02². From that it
  predicts perplexity / V = e^{σ²/2} = 1.026, against an observed 1.045. The README explains the small remainder
  (the tied head links each logit to the input).
* **Item 3.** It now prints the loss on padding targets vs real targets (10.59 vs 10.80) and explains why masking
  *raises* the untrained loss.

## 6. The chunked loss crashed under mixed precision

**Finding.** With bf16 hidden states and an fp32 head weight (the normal autocast setup), backward raised a dtype error.
Smaller edge cases:
* An all-ignored batch returned 0 instead of NaN.
* An `ignore_index` ≥ V crashed in `gather`.

**Changes.**
* All softmax arithmetic runs in fp32 with autocast turned off inside the function. Gradients are cast back to each input's dtype.
* Ignored targets are made safe before `gather`.
* Division is by the true count, so an all-ignored batch gives NaN, like PyTorch.
* New tests: `torch.autograd.gradcheck`, and a bf16-autocast run that now matches the fp32 reference.
* The memory probe falls back to `ru_maxrss` when Linux's `/proc` reset isn't available, and prints the subprocess's
  error output if it fails.

## 7. Item 7 reported one configuration only

**Change.** The notebook sweeps chunk sizes 64 / 256 / 1,024 rows: 65.0 / 102.2 / 396.0 MiB against 1,179.6 MiB, i.e.
18.1× / 11.5× / 3.0×. The ratio clearly depends on the configuration. 256 is still the headline number.

**Also corrected.** The description of what ordinary CE holds at its peak is now right: the saved log-softmax, the
gradient into it and the gradient into the logits. The arithmetic now reads 3 × 393 MiB ≈ 1.15 GiB (it said 1.18).

## 8. Deliverable and presentation

* The README says CPU-only Colab takes about 1.5–2 hours and recommends a GPU runtime.
* Plots are saved by the notebook itself (`savefig`), so the committed images come from the submitted code.
* The left Part 2 plot's x-axis now accounts for the 20-step smoothing window, and the plot adds the single-head
  baseline and validation curves.
* Prose that quotes numbers says it refers to the committed run.
* `from chunked_ce import …` is reloaded, so re-running an edited cell picks up the change.
* The README is shorter and leads with the tables.
* The Colab badge points at `kunal2016/lm-loss-harness`.

## What was deliberately not changed

* **Gradient clipping stays on in both runs.** Turning it off for one model would add a different confound. It is reported instead.
* **Two seeds, not more.** Each two-head run takes about 33 minutes on CPU. On a Colab GPU you can set
  `HARNESS_SEEDS=0,1,2` for a third seed.
* **The wrong-shift demo runs 150 steps, not 600.** It already shows the point the warning makes, and it keeps CPU run time down.
* **The training loss itself does not use the boundary mask.** The task asks for it on the measured loss. Training with
  it is a separate experiment, listed under future work in `DESIGN_NOTES.md`.
