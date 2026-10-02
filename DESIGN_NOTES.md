# Design notes: how this was set up and why

These notes give the reasoning behind each choice in the notebook, so that each one can be explained and defended.
Each section says what was done, why, and what the alternative would have been. The last section lists likely
questions with short answers.

---

## 1. Environment and tooling

| Choice | Why |
|---|---|
| **PyTorch, plain `nn.Module`s, no HuggingFace model** | This project is about the few lines between `hidden` and the scalar loss. A hand-written model keeps `model(tokens) → hidden` and `output_head(hidden) → logits` as two separate, visible steps. A library model hides the shift and the loss inside `forward(labels=...)`. |
| **`tiktoken` GPT-2 BPE (V = 50,257)** | A real subword vocabulary makes items 5–7 meaningful. Perplexity ≈ 50k is a sharp sanity check, the tied/untied difference is large (6.4 M parameters), and a 50k-wide logits tensor is what makes memory a real problem. A character vocabulary (V ≈ 65) would make items 6 and 7 almost trivial. |
| **TinyShakespeare** | Small (1.1 MB), public, downloads in a second on Colab, and has natural document boundaries (speeches). Its strings are readable, which item 2 depends on. |
| **Colab-first, device-agnostic** | `DEVICE = "cuda" if available else "cpu"`. Everything is created on CPU with a fixed seed and then moved, so items 1–3, 5 and 6 give identical numbers on GPU and CPU. |

## 2. Data

* **Speeches as documents.** The text is split on blank lines. Each split is one speech (`NAME:\n…`). There are 7,222 speeches.
* **`<|endoftext|>` after every speech, in training as well.** This is how GPT-2 was trained.
  It also fixes a flaw the first version had: if the model never sees `<|endoftext|>` in training, any loss
  measured at a document boundary mostly measures "a token I have never seen", not "the boundary".
  *Alternative:* train on raw continuous text. That is simpler, but it makes item 4 dishonest.
* **Split by speech, 90/10.** The first 90% of speeches go to train and the last 10% to validation. No speech is in both.
  A split by token position would cut a speech in half across the two sets.
* **Training batches are random windows** of 128 tokens from the concatenated stream. Windows freely cross
  `<|endoftext|>`. This is standard GPT-2-style pre-training, and the model learns to use `<|endoftext|>` as a reset signal.
* **Validation for item 4 shuffles speeches before packing.** Real packing combines unrelated documents.
  Consecutive speeches from the same scene are related (same speakers, same topic), so packing them in order would
  understate how unpredictable a boundary is.

## 3. Model

```
tokens (B,T) ─ tok_emb + pos_emb ─ 4 × [x + Attn(LN(x)); x + MLP(LN(x))] ─ LN_f ─→ hidden (B,T,128)
hidden ─ output_head (Linear 128→50257, no bias, weight tied to tok_emb) ─→ logits (B,T,50257)
```

| Choice | Value | Why |
|---|---|---|
| Width / depth / heads | d = 128, 4 layers, 4 heads (head dim 32) | Small enough to train on a free GPU in minutes and on CPU in about 15 minutes, big enough to learn real structure (validation perplexity 142 after 600 steps). |
| Context | T = 128 | Long enough to pack several speeches into one sequence for item 4 (about 4–5 on average), short enough to keep logits at 128 × 50k per sequence. |
| Pre-LayerNorm blocks + final LN | GPT-2 style | Trains stably without careful tuning. The final LN means `hidden` has unit scale, which is what makes the item 5 calculation work. |
| Attention | `F.scaled_dot_product_attention`, `is_causal=True` | Fused, uses less memory, and is correct by construction. For item 4 a boolean `(B,1,T,T)` mask is passed instead (True = may attend). |
| Learned absolute positions, and a `pos_ids` argument | | Needed so that document-isolated attention can restart positions at 0 per document. Without that, a document packed second sees position ids it would never have at its start, and "isolated" is not the same as "alone". The notebook checks that the two agree (max logit difference 7e-6, float rounding). |
| Init | N(0, 0.02) for Linear/Embedding, zero biases | GPT-2 init. It is also what makes the untrained logits almost flat, so item 5 gives ≈ V. |
| **Body returns `hidden`; the head is a separate module** | | Mirrors the original two lines. It lets the same body take a tied head, an untied head, a second head (Part 2) or the chunked loss (item 7, which needs `hidden` and the head weight, not the logits). |
| **Tied head by default** | `head.weight = model.tok.weight` | It is the same `Parameter` object, not a copy. That saves 6.4 M of 13.7 M parameters (47%). At this size the vocabulary matrices are most of the model, so untied would mean 2× the parameters for almost no extra capacity in the body. |

## 4. The loss harness (the core of the project)

```python
def shifted_targets(tokens, target_mask=None):
    tgt = tokens[:, 1:].clone()            # target at t is the token at t+1
    if target_mask is not None:
        tgt[~target_mask] = IGNORE         # -100
    return tgt

def lm_loss(logits, tokens, target_mask=None, reduction="mean"):
    tgt = shifted_targets(tokens, target_mask)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), tgt.reshape(-1),
                           ignore_index=IGNORE, reduction=reduction)
    return loss, int((tgt != IGNORE).sum())
```

| Decision | Why |
|---|---|
| **Build the target tensor explicitly** | Every mask (padding, boundary) becomes "write −100 into a target slot". There is one code path, it can be printed, and the masks compose (`&`). |
| **`ignore_index=-100`, not multiplying the loss by a mask** | With `reduction="mean"`, PyTorch divides by the number of *non-ignored* targets. Multiplying the per-token loss by a mask and then calling `.mean()` divides by *all* targets, a silent bug that makes the loss look lower whenever there is padding. |
| **Return the count of contributing tokens** | Item 3 asks for it. It is also the only reliable way to notice a mask that masks nothing, or everything. |
| **`.float()` before the softmax** | Under fp16/bf16 autocast, log-softmax over 50k entries loses precision. Upcasting only the logits fed to the loss costs little. |
| **`reduction="sum"` in evaluation, then divide by the total count** | This gives a token-weighted mean. Averaging per-batch means would weight batches with fewer real tokens more. It is the same "mean of means" bug as gradient accumulation without rescaling. |
| **Padding found by length, never by token id** | GPT-2 has no pad token, so `<|endoftext|>` is reused for padding. Masking "all `<|endoftext|>`" would also delete real end-of-document targets. |
| **Right padding** | With causal attention, real tokens never attend to padding to their right, so their logits are unaffected (verified: identical to the unpadded run). |

### Shift verification (item 2)
* The table prints **strings** (`repr` of each decoded token, so spaces and `\n` are visible), not ids.
* The machine check goes back to the corpus. For each batch row it knows the offset `i` the window was cut from,
  and checks `target[t] == corpus[i + t + 1]`. The first version compared `shifted_targets(tokens)` with
  `tokens[:, 1:]`, which compares a definition with itself and can never fail.
* **The warning, demonstrated.** A model trained with the shift reversed (`logits[:, 1:]` vs `tokens[:, :-1]`)
  ends 150 steps with a *lower* training loss than the correct model (4.66 vs 5.21), and the gap is still widening.
  That is because position `t+1` has already *seen* token `t`, so the task is copying. Scored by the correct harness
  it gets perplexity 6,730, against about 240 for the correct model at the same step. Its predictions, printed as
  strings, are the previous token (28% of positions so far; a small model needs time to learn a clean
  "look one step back" attention pattern, which is why the loss has not reached zero by step 150).

### Document packing (item 4)
* **What is masked:** only the prediction made *at* the `<|endoftext|>` that closes document A, whose target is the
  first token of document B. Everything else stays, including `<|endoftext|>` as a *target*, so the model still
  learns when a document ends.
* **Mask definition:** `doc_id[:, 1:] == doc_id[:, :-1]`. A target counts only if it belongs to the same document as
  its input. `<|endoftext|>` belongs to the document it closes.
* **Two different fixes, kept separate:**
  1. *Boundary target mask:* removes a target the context cannot justify. This is what the task asks for.
  2. *Document-isolated attention + position reset:* stops B from reading A. It is reported as a separate line,
     because it changes the forward pass, not the loss.
* **Reported three ways:** in-document, end-of-document and seam target losses, so no single average hides what is going on.
* **The decomposition is an identity, not evidence.** `before − after = (seams / all) × (seam_loss − after)`
  holds for any numbers. It explains the *size* of the change; the evidence is the per-kind losses.

### Perplexity sanity check (item 5)
* Perplexity is `exp(mean token NLL)`, computed token-weighted on validation.
* Why ≈ V: `hidden = LN(·)` has unit variance per coordinate, and the head weights are N(0, 0.02²), so each logit has
  variance about `d · 0.02² = 0.051`. Logits that are nearly constant mean a nearly uniform softmax: loss ≈ ln V, PPL ≈ V.
* Why slightly *above* V: for Gaussian logits with variance σ², E[log Σ exp zᵢ] ≈ ln V + σ²/2, so PPL/V ≈ e^{σ²/2}.
  The notebook measures σ² = 0.0512 (exactly d · 0.02²), which predicts 1.026. It observes 1.045. The extra ~2% comes
  from the tied head: the input token's own logit is correlated with the hidden state, so the logits are not
  independent.
* An assert stops the notebook if the ratio is outside 0.8–1.5 × V.

### Tied vs untied (item 6)
* Parameters are counted by deduplicating on `id(p)`. A tied weight is one tensor listed twice by `.parameters()`
  across modules, so it must be counted once. `model.parameters()` already deduplicates within one module, but
  `n_params(model, head)` across two modules would not.
* The difference is exactly V × d = 6,432,896, and the notebook asserts that.

### Chunked cross-entropy (item 7)
* **Why the ordinary version is expensive:** at its peak in backward, `F.cross_entropy(h @ W.T, y)` holds three
  tensors of size N × V in fp32: the saved log-softmax, the gradient into it, and the gradient into the logits.
  For N = 2,048 and V = 50,257 that is 3 × 393 MiB ≈ 1.15 GiB.
* **Design:** a custom `torch.autograd.Function` taking `hidden`, the head weight and the targets, not the logits.
  * Forward loops over chunks of rows and saves only the per-row log-sum-exp (N floats).
  * Backward recomputes each chunk's logits, turns them into `softmax − onehot` in place, then accumulates
    `grad_h = p @ W` and `grad_W += pᵀ @ h`. Peak memory is about one chunk of logits, plus the gradient buffers.
  * It trades memory for compute: the logits matmul runs twice, once in forward and once in backward.
* **Why a custom Function and not a Python loop with autograd:** autograd would save every chunk's logits for
  backward, and the peak would be the same as ordinary CE.
* **Mixed precision:** all softmax arithmetic runs in fp32 with autocast disabled inside the function, so it works
  when `hidden` is bf16/fp16 and the weight is fp32 (the normal mixed-precision setup). This was a bug in the
  first version.
* **Correctness:** it matches `F.cross_entropy` loss and both gradients in float64, including ignored targets and an
  uneven last chunk, and passes `torch.autograd.gradcheck`.
* **Measurement:** each variant runs forward and backward in its own subprocess, after a tiny warm-up.
  * GPU: `torch.cuda.max_memory_allocated()` minus the allocation before the call.
  * CPU: Linux's peak-RSS watermark (`VmHWM`), reset just before the call through `/proc/self/clear_refs`.
    It falls back to `ru_maxrss` where that is not available.
* **The ratio depends on the configuration.** It is roughly `3·N·V / (chunk·V + const)`, so the notebook sweeps chunk sizes 64 / 256 / 1024.

## 5. Training

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 | GPT-style defaults. β₂ = 0.95 reacts faster to gradient-scale changes, which matters on short runs. |
| Learning rate | 2e-3 peak, 50-step linear warm-up, cosine to 0 | Small models tolerate a high LR. Warm-up avoids early instability while Adam's statistics are still poor. |
| Batch | 16 × 128 = 2,048 tokens/step | Fits easily everywhere. |
| Steps | 600 | Long enough that the curves flatten and the Part 2 heads clearly separate, and short enough for CPU. |
| Gradient clipping | global norm 1.0 | Standard safety net. The two-head loss has about 1.8× the gradient norm (1.34 vs 0.73 late in training), so it is clipped on 99% of late steps against 1% for the single-head model. **With Adam, a roughly constant rescaling largely cancels**: the update is m/√v, and scaling every gradient by c scales both m and √v by c. So clipping should not act like halving head 1's learning rate. It is only exact when the scale factor is steady over Adam's averaging window (about 20 steps for β₂ = 0.95), and it wasn't tested directly, so the write-up lists it as a caveat. |
| Seeds | 0 and 1 for every model in Part 2 | The seed fixes init and data order. Two seeds give a minimal noise estimate for the single-head vs two-head comparison. Evaluation uses its own fixed generator, so every model is evaluated on the same 20 validation batches. |

## 6. Part 2: the t+2 head

* **Architecture:** `head2(h) = unembed(LN(h + MLP(h)))`, using the same tied unembedding as head 1.
  * This is the shape of a Medusa head (Cai et al., 2024).
  * It adds 132k parameters instead of a second 6.4 M-parameter output matrix.
* **The MLP's last layer starts at zero.** At init, head 2 equals head 1 plus one LayerNorm, so both start at ln V
  and any difference that appears later was learned.
* **Shift:** `logits2[:, :-2]` vs `tokens[:, 2:]`, printed as strings next to head 1's targets.
* **Objective:** `loss = loss_t+1 + loss_t+2`, with equal weights, as the task asks for "their sum".
* **What to expect, and why** (the formal version):
  * For a stationary source, H(X_{t+2} | X_{≤t}) ≥ H(X_{t+2} | X_{≤t+1}) = H(X_{t+1} | X_{≤t}).
  * The inequality holds because conditioning on less cannot reduce entropy; the equality is stationarity.
  * So head 2's best achievable loss is *at least* head 1's. It is equal only if token t+1 carries no information about t+2,
    which is false for text.
  * In practice most of the easy structure in BPE text (finishing a word split into pieces, punctuation, the newline
    after `NAME:`) is about the very next token, so the gap opens and stays open.
* **Caveats stated in the write-up:** both heads are far above any entropy floor, so the gap size also reflects head 2's
  smaller capacity (one MLP block). The flattening late in training coincides with the cosine LR decaying to zero.
  Two seeds give only a rough noise estimate.
* **Literature:** Gloeckle et al. (2024), *Better & Faster Large Language Models via Multi-token Prediction*, found
  that auxiliary future-token heads hurt small models and help at billions of parameters, especially on code.
  A small cost at 0.8 M body parameters is consistent with that.

## 7. Known limitations (say these before someone else does)

* CPU-only, 600 steps, two seeds, one tiny configuration. The numbers are for illustration, not benchmarks.
* Item 7's CPU number is process RSS (includes allocator behaviour). The GPU number from `max_memory_allocated` is cleaner.
* The training data does **not** use the boundary mask or isolated attention. Only evaluation does. Training with them is
  a separate experiment.
* Item 4 uses validation speeches packed greedily. The last speech in each pack is cut at 128 tokens and its tail dropped.

---

## 8. Likely questions and answers

**Q: Is the original snippet's shift correct?**
Yes. `logits[:, :-1]` (positions 0..T−2) is scored against `tokens[:, 1:]` (tokens 1..T−1): position t predicts token t+1.
The last position is dropped because its target is outside the window. The snippet's problems were that you
couldn't *see* any of this, and there was no masking.

**Q: Why does a wrong shift give a beautiful loss curve?**
With `logits[:, 1:]` vs `tokens[:, :-1]`, position t+1 is asked for token t, which it already has in its causal window
(one step back). The model learns to copy, and its loss drops *below* the correct model's (4.66 vs 5.21 after 150
steps here, and still falling). Nothing crashes, and the shapes match. Only reading the strings, or scoring with the
correct harness (perplexity 6,730 vs ~240), reveals it.

**Q: Then why didn't the wrong-shift loss go to zero?**
Copying the previous token needs an attention head that reliably looks exactly one step back. A 4-layer model with
learned positions takes a while to form one. By step 150 only 28% of its top predictions are copies. Given longer
training, the loss would keep falling toward 0.

**Q: How would you catch an off-by-one in a real codebase?**
Print decoded input/target pairs for one batch, and assert targets against the raw corpus offsets (not against the
code's own shift). Check that untrained perplexity ≈ V. Be suspicious of a training loss far below what a sensible model
could reach, for example near 0 on natural text.

**Q: Why `ignore_index` instead of multiplying by a mask?**
`reduction="mean"` with `ignore_index` divides by the number of real targets. `(loss * mask).mean()` divides by all
positions, which deflates the loss in proportion to the amount of padding. If you do multiply by a mask, divide by `mask.sum()`.

**Q: Why did masking padding make the loss go *up*?**
At init with a tied head, the logit of the input token itself is slightly higher (the hidden state still contains its
embedding), so pad→pad targets (`<|endoftext|>` → `<|endoftext|>`) are a bit cheaper than real ones and drag the mean
down. The masked number is the true loss. In a trained model pad→pad becomes trivially predictable, and the
unmasked loss would look much better than it really is.

**Q: Why keep `<|endoftext|>` as a target but mask the token after it?**
The end of a document is predictable from the document ("…and so I end."), so it is a fair target. The first token
of the next document is not predictable from the previous document, so asking for it trains a spurious A→B association.

**Q: Isn't the boundary mask useless if B can still attend to A?**
They solve different problems. The target mask removes an unjustified label. Isolated attention removes unjustified
*context*. Many pipelines use only the first (GPT-2 style); others use both (block-diagonal or "varlen" attention).
The notebook reports both so the effects are not mixed up.

**Q: Why is untrained perplexity slightly above V, not exactly V?**
Logits are not exactly equal. They have variance σ² ≈ d·0.02² ≈ 0.05, and for random logits the expected
log-sum-exp is ln V + σ²/2, so PPL ≈ V·e^{σ²/2} ≈ 1.03 V. The tied head adds a little more. Anything like 0.1 V or 10 V
means a bug (wrong shift, wrong reduction, a logits scale problem, or a mask that hits too much).

**Q: When is tying the head a bad idea?**
When the vocabulary is small relative to d (the saving is negligible), or when input and output roles of a token differ
a lot and you can afford the parameters. Most small and medium LMs tie; several large ones untie.

**Q: Why does the chunked loss use less memory, and what does it cost?**
It never stores the N × V logits; it keeps N log-sum-exp values and recomputes one chunk's logits at a time in backward.
The cost is one extra `h @ Wᵀ` matmul, roughly +50% of the loss layer's compute. Smaller chunks mean less memory but
more kernel launches. Production versions (Liger kernel, Apple's "Cut Cross-Entropy") fuse this on GPU.

**Q: Why measure in a subprocess?**
PyTorch's caching allocator, and the C allocator on CPU, keep memory around after it is freed. Running each variant in a
fresh process means neither inherits the other's peak or cached blocks.

**Q: Why does head 2's loss stay above head 1's?**
Head 2 has to predict t+2 without knowing t+1, so it pays for its uncertainty about t+1 as well. Formally,
H(t+2 | ≤t) ≥ H(t+1 | ≤t) for a stationary source. Here the gap grows from 0.26 nats (step 75) to 0.75 (step 450), then
stays flat. Head 1 keeps learning local, next-token-only patterns that head 2 can't use.

**Q: Is the plateau of the gap a property of the task or of the schedule?**
This run can't tell. The gap stops growing at step 450, when the learning rate is already at 17% of peak and falling to
zero. A constant-LR tail, or a longer run, would separate the two.

**Q: Did the extra head help?**
At this scale, no. Next-token loss with the extra head is 5.038 against 4.957 without it (+0.071 and +0.091 for the two
seeds), while the two single-head seeds differ by only 0.0007. This is expected from Gloeckle et al. (2024), where the
benefit appears only in large models. The extra objective competes for a very small body's capacity.

**Q: Doesn't gradient clipping on the summed loss bias the comparison?**
It is a fair concern: the two-head model is clipped on 99% of late steps, the single-head model on 1%. But with Adam a
steady rescaling of the gradient cancels in m/√v, so this should not act like a lower learning rate. The honest answer:
it is unlikely to explain 0.08 nats, and the clean test (same run with clipping off, or clipping each loss separately)
is listed as future work.

**Q: What would you do with more time?**
Run on GPU with more seeds and longer training (and a constant-LR tail). Train with the boundary mask and isolated
attention and compare. Re-run the two-head model with clipping off, or clipping per loss. Weight the t+2 loss
(for example 0.5) and sweep the weight. Give head 2 more capacity (a full transformer layer, as in Gloeckle et al.) to
separate capacity from the entropy floor. Use the t+2 head for self-speculative decoding and measure acceptance rate.
Report GPU memory with the fused chunked loss.
