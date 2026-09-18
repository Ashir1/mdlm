# RELAY adaptation for MDLM (OWT checkpoint) — implementation plan

References read: `relay_prev_implementations/fast-dllm-v2/v2/src/lmflow/models/fast_dllm/{configuration,modeling}.py`,
`.../pipeline/utils/{block_bptt_loss,streaming_batch,bptt_trainer}.py`, `.../generation_functions.py`, `.../eval.py`,
and `relay_prev_implementations/sudoku/relay/{model,loss,streaming_batch,predictor}.py`.
Target: this repo (`models/dit.py`, `diffusion.py`, `main.py`, `configs/`).

**Status (2026-09-18):** M0 (pretrained init, `mode=parity_check`) and the M1 model changes it
needs (`relay_ln`, `h_prev`/`return_state` in `DIT.forward` and `Diffusion.forward`) are
implemented and verified on Modal (A10): local DIT vs `hf_dit` log-probs bitwise identical,
zero-init relay is an exact no-op, EMA rebuilt from the loaded weights. Decision: only the base
MDLM schedule is used as the reveal policy — no confidence-threshold policy, no `relay_policy.py`.
Everything runs on Modal (`modal_app.py`); no local/CPU code paths.
M3 done: `streaming_batch.py`, `relay_loss.py` (`Schedule` + `RelayRolloutLoss`), `training_step` /
`validation_step` dispatch, relay grad-norm diagnostic, `relay.*` config; `tests/test_relay.py`
(8 tests) passes on Modal (`modal run modal_app.py --cmd "python -m pytest -q tests"`). Test
gotcha: an untrained DiT has a zero-initialized output head and adaLN gates, so gradients only
reach the head — tests randomize those first.
M4 done: end-to-end `trainer.fit` on Modal (wikitext2 tokenized into `/root/outputs/data`,
pretrained init, relay on, grad accumulation 2, validation + sample generation, `best.ckpt`/
`last.ckpt` with relay params + EMA, CSV metrics via `wandb=null`), then a resume run from
`last.ckpt` (pretrained init skipped, states restored, 40 → 60 steps). Observations: relay grad
norm ≈ 0.1–0.2 of the backbone's from step 1 (gate opens); buffer rows keep the phases the
cold-start stagger gave them (every row lives exactly ceil(T/K) calls), so per-call noise levels
are the B phases of that rank — uniform over a trajectory, correlated within a call.
M5 done: `sampling.predictor=relay_ddpm` (`Diffusion._relay_sample`: `relay_loss.Schedule` reveals,
tokens from p_x0, state carried through all T+1 forwards, no cache; `use_state=false` gives the
explicit-select `ddpm` without cache), `Diffusion.nfe` counts forwards per `_sample` (logged as
`val/sampling_nfe`, printed by `mode=sample_eval`). Verified: sampler test (closed relay ≡ no
state under the same seed, open relay changes samples, NFE = T+1) and `mode=sample_eval` from the
M4 checkpoint (65 forwards at T=64, gen-ppl 31.3 under gpt2-large). Next: OWT data prep on Modal
(CPU job) and the M6 runs.

## 0. What the references actually do (distilled)

| Piece | Fast-dLLM v2 fork | Sudoku package | What we take for MDLM |
|---|---|---|---|
| Injection | `x = tok_emb + relay_layer_norm(h_t)` **at mask positions only** (`torch.where(guard, …)`), `nn.LayerNorm` with `weight` zero-init (bias default 0) | same, but at all positions | mask-only, affine LN, weight+bias zero-init |
| Relay state `h_s` | hidden after decoder layer `relay_layer` (`-1` = final RMSNorm output) | hidden after encoder layer `relay_layer` (`-1` = last) | residual stream after DDiT block `relay.layer` (default last), **before** `output_layer` |
| Position shift | right-shifts `h_s` and logits by one (causal next-token model) | none | **none** — MDLM predicts token *i* at position *i* |
| First-step state | training: zeros; `batch_sample` inference: `None` (no injection) — a train/infer mismatch | zeros in both | **zeros in both** (`h_prev=None` is reserved for "relay off") |
| Reveal policy | threshold on top-p-sampled token prob + per-block argmax fallback | confidence-mass threshold + force-1 fallback, or uniform | MDLM's own `_ddpm_update` marginal (schedule), confidence policy later |
| Loss | `L1 + L2`, masked-mean CE per step, `h_s1` attached (or detached for sg), `h_s2` detached into buffer | `num_steps` generic loop, same idea | same, K=2 |
| Buffer | per-rank `StreamingBatch`: capacity = batch size, evict rows with no masks left, refill from the incoming batch (unused incoming rows are dropped) | same + `fixed` mask | same + per-row schedule step + `mutable` mask |
| Validation | buffer reset per call, 2 steps from all-mask | same | same, plus MDLM's ELBO `val/ppl` and gen-ppl on carried-state samples |

## 1. Design decisions (fixed up front)

1. **Interface.** `models.dit.DIT.forward(indices, sigma, h_prev=None, return_state=False)` and
   `Diffusion.forward(x, sigma, h_prev=None, return_state=False)`. Legacy call sites keep working
   (`return_state=False` returns the log-probs tensor as today). Relay paths call with
   `return_state=True` and get `(log_probs, h_next)`, `h_next: [B, L, D]` fp32.
2. **`h_prev=None` ≠ zeros.** `None` skips the injection entirely (legacy MDLM behaviour, and the
   rollout-only ablation). A *fresh* row / generation step 0 passes `zeros(B, L, D)`, which injects
   `LN(0) = bias` once the bias has trained. Training and inference must agree here (follow Sudoku,
   not `generation_functions.batch_sample`).
3. **Relay state = residual stream after block `relay.layer`** (fp32; the residual stream stays
   fp32 through the bf16 block autocast because `x_skip` is fp32). Not the `norm_final`/adaLN
   output — `DDitFinalLayer` (`models/dit.py:302-321`) is only ever applied on the way to logits.
4. **One reveal rule, two consumers.** `P(reveal | masked, step i) = (mc_t − mc_s)/mc_t` with
   `mc = 1 − exp(−σ)`; this is exactly the marginal of `_ddpm_update` (`diffusion.py:612-637`):
   `q_xs ∝ p_x0·(mc_t − mc_s)` for tokens and `mc_s` for MASK ⇒ P(stay masked) = `mc_s/mc_t`.
   Training fills selected positions with ground truth; inference with a categorical sample from
   `p_x0`. Step `T` (after the `T` schedule steps) is the analogue of `sampling.noise_removal`:
   reveal everything that is still masked (argmax at inference).
5. **Per-row schedule position** (`step[b]`), `σ_b = noise(t_{step[b]})`. Since the OWT
   checkpoint has `time_conditioning=False`, σ is zeroed inside `_process_sigma` anyway — the relay
   state is the only cross-step signal, which is the point of the experiment.
6. **Loss** = plain masked-mean CE per step (trajectory CE), not the ELBO weight
   `dsigma/expm1(sigma)`. `_forward_pass_diffusion` is left untouched; `training_step` dispatches.
7. **Cold-start stagger (config flag, default on).** With a deterministic schedule and all rows
   starting from all-mask, every row finishes on the same call, the whole buffer is replaced at once,
   and every micro-batch sits at one noise level forever. At cold start only, give row *b* a random
   step `i_b ~ U{0..T−1}` and mask each mutable token w.p. `mc(t_{i_b})` — that is MDLM's own `q_xt`
   distribution with uniform `t`, with zero relay state. Steady-state refills start from all-mask
   (as specified).
8. **EMA is rebuilt after loading pretrained weights** (`Diffusion.__init__` clones the *random*
   init into `ema.shadow_params`, `diffusion.py:139-145`). Otherwise validation swaps in garbage.
9. **`ddpm_cache` is never used with relay** (`_ddpm_caching_update` reuses `p_x0` when `x` is
   unchanged; with a carried state the prediction is stale). The relay sampler recomputes every step
   and counts NFE.

## 2. Milestones

### M0 — pretrained init + parity with zero relay contribution

**`main.py`**
- New config key `training.init_from_pretrained: null` (HF repo id, `model.safetensors` path, or a
  Lightning `.ckpt`). In `_train` (`main.py:148-183`), right after `model = diffusion.Diffusion(...)`:
  `if config.training.init_from_pretrained: model.load_pretrained_backbone(path)`. Resume still wins:
  `trainer.fit(ckpt_path=...)` loads *after* construction, so a resumed run is unaffected.
- New `mode=parity_check` (same style as `ppl_eval`/`sample_eval`, `main.py:196-201`).

**`diffusion.py`**
```python
def load_pretrained_backbone(self, path):
  if path.endswith('.ckpt'):
    sd = torch.load(path, map_location='cpu')['state_dict']
  else:
    f = path if path.endswith('.safetensors') else \
        huggingface_hub.hf_hub_download(path, 'model.safetensors')
    sd = safetensors.torch.load_file(f)          # keys: backbone.vocab_embed.embedding, backbone.blocks.*, ...
  missing, unexpected = self.load_state_dict(sd, strict=False)
  relay_keys = {k for k in self.state_dict() if '.relay_ln.' in k}
  assert not unexpected, unexpected
  assert set(missing) <= relay_keys, set(missing) - relay_keys
  if self.ema:                                   # decision 8
    self.ema = models.ema.ExponentialMovingAverage(
      itertools.chain(self.backbone.parameters(), self.noise.parameters()),
      decay=self.config.training.ema)
```
The HF repo's `modeling_mdlm.py` uses the same submodule names as `models/dit.py`
(`vocab_embed`, `sigma_map`, `rotary_emb`, `blocks`, `output_layer`) under `MDLM.backbone`, and
`Diffusion` also names its backbone `backbone` (`diffusion.py:92-94`), so the safetensors keys map
1:1. `config.json`: `hidden_dim=768, n_blocks=12, n_heads=12, cond_dim=128, vocab_size=50258,
model_length=1024, time_conditioning=false` = `configs/model/small.yaml` + GPT-2 tokenizer
(`mask_index = 50257`, `vocab_size → 50258`, `diffusion.py:85-90`). `noise` (LogLinear) has no
state-dict entries. Allow `rotary_emb.inv_freq` to be missing (it is recomputed).

**Parity check** (`mode=parity_check`): build `Diffusion(backbone='dit', relay.enabled=true)` +
`load_pretrained_backbone`, and `Diffusion(backbone='hf_dit')` via `_load_from_checkpoint`
(`main.py:25-33`, needs `eval.checkpoint_path=kuleshov-group/mdlm-owt`). Both `.eval()`, same
random `x` with ~50 % `mask_index`, `sigma=zeros`. Assert `torch.testing.assert_close` on
`Diffusion.forward` outputs (post-SUBS), then again with `h_prev=randn(B, L, D)` — zero-init LN
(`weight=0, bias=0`) makes the delta exactly 0 for *any* `h_prev`. Also assert
`model.ema.shadow_params` equals the loaded params. Print max-abs diff.

**Guardrails:** `assert config.backbone == 'dit'` when `relay.enabled` (the `hf_dit` path loads
the remote `modeling_mdlm.py`, so edits to `models/dit.py` would be silently ignored). Also assert
`parameterization == 'subs'` and `T == 0`.

### M1 — relay connection in one DIT forward

**`models/dit.py`** (`DIT.__init__` 324-351, `DIT.forward` 359-370):
```python
def __init__(self, config, vocab_size, mask_index=None):
  ...
  relay = config.get('relay', None)
  self.relay_enabled = bool(relay and relay.enabled and relay.use_state)
  if self.relay_enabled:
    assert mask_index is not None
    self.mask_index = mask_index
    self.relay_layer = relay.layer % config.model.n_blocks     # -1 -> last block
    self.relay_ln = nn.LayerNorm(config.model.hidden_size)      # affine; R(h) = LN(h)
    nn.init.zeros_(self.relay_ln.weight)                        # bias is already 0 -> R == 0 at init

def forward(self, indices, sigma, h_prev=None, return_state=False):
  x = self.vocab_embed(indices)
  if h_prev is not None:
    assert self.relay_enabled
    delta = self.relay_ln(h_prev.to(x.dtype))                   # fp32, outside the bf16 autocast
    x = torch.where((indices == self.mask_index)[..., None], x + delta, x)
  c = F.silu(self.sigma_map(sigma))
  rotary_cos_sin = self.rotary_emb(x)
  h_next = None
  with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    for i in range(len(self.blocks)):
      x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      if return_state and i == self.relay_layer:
        h_next = x                                              # residual stream, pre output_layer
    logits = self.output_layer(x, c)
  return (logits, h_next) if return_state else logits
```
`Diffusion.__init__` passes `mask_index=self.mask_index` to `models.dit.DIT(...)`.

**`diffusion.py` `forward`** (312-327):
```python
def forward(self, x, sigma, h_prev=None, return_state=False):
  sigma = self._process_sigma(sigma)
  with torch.cuda.amp.autocast(dtype=torch.float32):
    if h_prev is not None or return_state:
      logits, h_next = self.backbone(x, sigma, h_prev=h_prev, return_state=True)
    else:
      logits, h_next = self.backbone(x, sigma), None
  # existing subs / sedd / d3pm branches -> out
  return (out, h_next) if return_state else out
```
SUBS handling (`_subs_parameterization`, 261-277) is untouched: mask logit → −∞, visible tokens
→ one-hot. The relay state is read *before* any of that.

**Verify:** parity check still passes with `h_prev=randn` (still zero delta); `h_next.shape ==
(B, L, 768)`, dtype fp32; with `relay_ln.weight` set to ones the outputs change only at mask
positions' downstream logits (sanity that the guard works).

### M2 — (folded into M3) the reveal rule, as a helper in `relay_loss.py`

Not a separate milestone: `_ddpm_update` samples token-or-MASK jointly, so training needs the
reveal decision made explicit (selected positions get ground truth). The helper below is exactly
MDLM's marginal, and the relay sampler (M5) calls the same function.

```python
class SchedulePolicy:
  def __init__(self, noise, num_steps, eps=1e-5):
    self.noise, self.T = noise, num_steps
    self.timesteps = torch.linspace(1, eps, num_steps + 1)      # same grid as _sample (669-671)
    self.dt = (1 - eps) / num_steps
  def t_of(self, step):                                          # step: [B] long, 0..T
    return self.timesteps.to(step.device)[step.clamp(max=self.T)]
  def select(self, log_p_x0, masked_mutable, step):              # -> bool [B, L]
    t = self.t_of(step)
    mc_t = 1 - torch.exp(-self.noise(t)[0])
    mc_s = 1 - torch.exp(-self.noise((t - self.dt).clamp_min(0))[0])
    p = ((mc_t - mc_s) / mc_t)[:, None]                          # loglinear: dt / t
    reveal = torch.rand(masked_mutable.shape, device=step.device) < p
    reveal |= (step >= self.T)[:, None]                          # noise-removal step: reveal the rest
    return reveal & masked_mutable
```
Confidence-threshold selection is deliberately out of scope: the base MDLM schedule is the only
policy, so NFE is always `T+1` and the sampler stays a controlled comparison against `ddpm`.

`log_p_x0` is passed `.detach()`ed; selection is discrete and never in the graph.

### M3 — streaming buffer + K=2 rollout loss (`streaming_batch.py`, `relay_loss.py`, new)

**`StreamingBatch`** (port of the Fast-dLLM one; storage per rank, never in autograd, not saved
in checkpoints — cold-starts on resume, as in both references):

| field | shape / dtype | meaning |
|---|---|---|
| `x0` | `[C, L]` long | clean target (`batch['input_ids']`) |
| `x` | `[C, L]` long | current partially masked tokens |
| `h` | `[C, L, D]` fp32 | relay state carried into the next call (detached) |
| `mutable` | `[C, L]` bool | `batch['attention_mask'].bool()` — all ones for wrapped OWT; hook for prefix-conditioning later |
| `step` | `[C]` long | per-row schedule position (0..T) |
| `ready` | `[C]` bool | no masked-mutable positions left, or `step > T` |

- `evict_and_fill(batch)`: cold start → `C = B`, fill every row (with the stagger of decision 7 if
  `relay.cold_start_stagger`); later → fill `ready` rows from the first `n` rows of the incoming
  batch: `x = where(mutable, mask_index, x0)`, `h = 0`, `step = 0`. Returns `n`.
- `persist(x, h, step)`: `detach().clone()` writes; recompute `ready`.
- `reset()` for the validation buffer.

**`RelayRolloutLoss.__call__(batch, training)`** (module-level, holds `train_buf`/`val_buf`):
```python
buf = self.train_buf if training else self.val_buf
if not training: buf.reset()
n_new = buf.evict_and_fill(batch)
x0, x, h, mutable, step = buf.x0, buf.x.clone(), buf.h.clone(), buf.mutable, buf.step.clone()
losses = []
for k in range(K):                                              # K = relay.num_rollout_steps (2)
  masked = (x == self.mask_index) & mutable
  sigma = self.noise(self.policy.t_of(step))[0]                 # [B]; zeroed by _process_sigma when time_conditioning=False
  h_in = h if self.use_state else None
  log_p, h_next = self.module.forward(x, sigma, h_prev=h_in, return_state=True)
  nll = -torch.gather(log_p, -1, x0[..., None]).squeeze(-1)
  L_k = (nll * masked).sum() / masked.sum().clamp_min(1) + 0.0 * log_p.sum()   # anchor keeps DDP buckets in sync
  losses.append(L_k)
  reveal = self.policy.select(log_p.detach(), masked, step)
  x = torch.where(reveal, x0, x)                                # teacher forcing
  step = step + 1
  last = (k == K - 1)
  h = h_next.detach() if (last or self.stop_grad_state) else h_next   # h1 attached (BPTT), h2 detached
buf.persist(x, h, step)                                         # finished rows evicted on the next call
return torch.stack(losses).sum(), metrics
```
Rows that finish after step 1 simply contribute zero masked positions in step 2 (masked-mean
handles it) and are replaced at the next call — "replace completed rows at the next window
boundary". Diagnostics to return (from `block_bptt_loss._collect_metrics`): `L1`, `L2`,
`reveal_frac`, `mask_ratio_step1/2`, `h1_norm`, `inject_delta_norm`, `relay_ln_weight_norm`,
`relay_ln_bias_norm`, `buf_evicted`, `buf_mean_mask_count`, `buf_mean_step`.

**Verify (the "working two-step backward" milestone):** on one batch, `loss.backward()` gives
non-zero `relay_ln.weight.grad` (it can only come from forward 2 through `h1`); with
`relay.stop_grad_state=true` the grad on forward-1-only paths through `h1` is gone (check that
`h1.requires_grad` and `torch.autograd.grad(L2, h1)` is non-zero in the default and raises/zero under
sg); buffer invariants: `ready` rows get refilled, `step` increments by 2 per call, `h` rows for
fresh slots are zero; a full trajectory of `(T+1)/2` calls on a fixed batch drives every row to
`ready`.

### M4 — Lightning wiring (`diffusion.py`, `configs/config.yaml`)

- `training_step` (390-397): `if self.relay_enabled: loss, m = self.relay_loss(batch, training=True)`
  and `self.log_dict({f'train/relay_{k}': v ...}, on_step=True)`; else the existing
  `_compute_loss`. `trainer/loss` logging unchanged.
- `validation_step` (412-413): keep `_compute_loss(batch, 'val')` (MDLM ELBO → `val/nll`, `val/ppl`,
  the "is it still an MDLM" check; `checkpoint_monitor` watches `val/nll`). With relay enabled pass
  `h_prev=zeros` inside `_forward_pass_diffusion` (add an optional `h_prev` arg there — the only
  touch to that function). Additionally run the rollout loss on the val buffer (reset per batch, 2
  steps from all-mask) → `val/relay_L1`, `val/relay_L2`. Optional flag `relay.val_full_trajectory`:
  full teacher-forced trajectory on the first val batch, CE bucketed by step decile.
- `on_validation_epoch_end` (415-447) needs no change: `_sample()` → new sampler (M5) → existing
  gen-ppl under `gpt2-large`.
- `on_after_backward`: every `log_every_n_steps`, relay-only grad norm vs. rest (the
  `bptt_trainer` ratio) — the single most useful "is the relay learning" signal.
- Optimizer/EMA/checkpoint: nothing to do — `relay_ln` lives inside `backbone`, so
  `configure_optimizers` (454-461), `optimizer_step` EMA update (254-259), `on_save/load_checkpoint`
  (168-215) and `ModelCheckpoint` all pick it up. `find_unused_parameters: false`
  (`configs/strategy/ddp.yaml`) is fine because forward 2 always runs and always touches `relay_ln`;
  with `relay.use_state=false` the LN is not created, so there is no unused parameter.
- `hydra` config additions (in `configs/config.yaml`):
```yaml
relay:
  enabled: False            # rollout-buffer training loop on/off
  use_state: True           # False = rollout-only ablation (no relay_ln, h_prev=None)
  layer: -1                 # DDiT block whose output is the relay state
  num_rollout_steps: 2      # K
  stop_grad_state: False    # True = RELAY (sg)
  num_steps: ${sampling.steps}   # T of the training trajectory schedule
  cold_start_stagger: True
  val_full_trajectory: False
training:
  init_from_pretrained: null
sampling:
  predictor: ddpm_cache     # + new value: relay_ddpm
```
Adaptation hyper-parameters to start from (all overridable): `optim.lr=5e-5`,
`lr_scheduler.num_warmup_steps=500` (default 2500 is for from-scratch, `configs/lr_scheduler/constant_warmup.yaml`),
`training.ema=0.999` for short runs (0.9999 has a ~10k-step horizon; or evaluate with
`eval.disable_ema=true` as well), `trainer.val_check_interval=2000`, `loader.global_batch_size=64`.

### M5 — sampler that carries the state

```python
@torch.no_grad()
def _relay_sample(self, num_steps=None, eps=1e-5):
  B, L, D = self.config.loader.eval_batch_size, self.config.model.length, self.config.model.hidden_size
  x = self._sample_prior(B, L).to(self.device)
  h = torch.zeros(B, L, D, device=self.device) if self.relay_use_state else None   # decision 2
  policy = SchedulePolicy(self.noise, num_steps, eps)
  step = torch.zeros(B, dtype=torch.long, device=self.device)
  self.last_nfe = 0
  for i in range(num_steps + 1):                                 # i == num_steps: noise removal
    sigma = self.noise(policy.t_of(step))[0]
    log_p, h = self.forward(x, sigma, h_prev=h, return_state=True); self.last_nfe += 1
    masked = x == self.mask_index
    reveal = policy.select(log_p, masked, step)
    x_hat = _sample_categorical(log_p.exp()) if i < num_steps else log_p.argmax(-1)
    x = torch.where(reveal, x_hat, x)
    step += 1
  return x
```
Hook: `_sample` (657-698) → `if self.sampler == 'relay_ddpm': return self._relay_sample(...)`.
Semi-AR (`sample_subs_guidance`, 957-997) stays on the old path for now. K=2 only bounds the
training gradient window; the sampler carries `h` for all `T+1` steps.

**Verify:** at the freshly-initialised relay model (M0 weights), `relay_ddpm` and `ddpm` produce
samples with the same gen-ppl within seed noise (they are the same Markov chain, decision 4); NFE
= `T+1`; samples decode to text; all positions non-mask at the end.

### M6 — comparison protocol

Runs (same data, seed set, `T`, decoding rule):

| run | training | sampler |
|---|---|---|
| A. original ckpt | none | `ddpm` (and `ddpm_cache`, reporting real NFE) |
| B. rollout-only adaptation | `relay.enabled=true relay.use_state=false` | `relay_ddpm` (= explicit-select `ddpm`, no cache) |
| C. RELAY | `relay.enabled=true` | `relay_ddpm` with carry |
| C-sg. RELAY (sg) | `+ relay.stop_grad_state=true` | same |
| D. C with `h` zeroed every step | — | separates "adapted weights" from "carried state" |

Metrics: `val/ppl` (ELBO, single forward, `h=zeros`), gen-ppl under `gpt2-large` at matched NFE
∈ {32, 64, 128, 256, 1024}, plus a diversity guard (unigram entropy / distinct-n of samples — gen-ppl
alone is gameable by repetitive text), 3 seeds, NFE logged per run, and the relay diagnostics
(`relay_ln_weight_norm`, carry-grad ratio) to confirm the gate actually opens.

## 3. File map

| file | change |
|---|---|
| `models/dit.py` | `mask_index` ctor arg, `relay_ln`, `relay_layer`, `h_prev` injection, `return_state` |
| `diffusion.py` | `forward(h_prev, return_state)`, `load_pretrained_backbone`, `training_step` dispatch, val relay metrics, `on_after_backward` diagnostics, `_relay_sample` + `_sample` hook, config asserts |
| `streaming_batch.py` (new) | per-row buffer |
| `relay_loss.py` (new) | `SchedulePolicy` (shared by training and the sampler) + `RelayRolloutLoss` (K-step rollout, metrics) |
| `main.py` | `training.init_from_pretrained` hook in `_train`, `mode=parity_check` |
| `configs/config.yaml` | `relay:` block, `training.init_from_pretrained`, `sampling.predictor: relay_ddpm` |
| `scripts/adapt_owt_relay.sh` (new) | slurm launcher for B / C / C-sg |
| `tests/test_relay.py` (new) | tiny-config tests: zero-init parity, buffer lifecycle, grad through `h1`, policy marginal vs `_ddpm_update`, sampler termination/NFE |

## 4. Commands

All GPU work runs on Modal through `modal_app.py` (image: torch 2.2.2+cu121, flash-attn 2.5.6,
the pinned MDLM deps; persistent volumes `mdlm-relay-hf-cache` → `/root/.cache/huggingface` and
`mdlm-relay-outputs` → `/root/outputs`). `--args` is passed verbatim to `python main.py`; always
set `hydra.run.dir=/root/outputs/<run>` so Hydra writes to the volume. `MDLM_MODAL_GPU` selects
the GPU (default `A10G`; training wants `A100-80GB`/`H100`). On Windows prefix the command with
`PYTHONUTF8=1` (the Modal CLI prints non-cp1252 characters).

```bash
# M0 parity
PYTHONUTF8=1 modal run modal_app.py --args "mode=parity_check backbone=dit model=small \
  data=openwebtext-split model.length=1024 relay.enabled=true \
  training.init_from_pretrained=kuleshov-group/mdlm-owt hydra.run.dir=/root/outputs/parity_check"

# M3/M4 adaptation (C)
python main.py model=small data=openwebtext-split parameterization=subs backbone=dit model.length=1024 \
  relay.enabled=true relay.num_steps=128 training.init_from_pretrained=kuleshov-group/mdlm-owt \
  optim.lr=5e-5 lr_scheduler.num_warmup_steps=500 training.ema=0.999 \
  trainer.max_steps=20000 trainer.val_check_interval=2000 \
  loader.global_batch_size=64 loader.batch_size=8 loader.eval_batch_size=8 \
  sampling.predictor=relay_ddpm sampling.steps=128 eval.compute_generative_perplexity=True \
  wandb.name=mdlm-owt-relay
#   B:    ... relay.use_state=false
#   C-sg: ... relay.stop_grad_state=true

# M5/M6 sampling from an adapted checkpoint
python main.py mode=sample_eval backbone=dit relay.enabled=true \
  eval.checkpoint_path=<run_dir>/checkpoints/last.ckpt data=openwebtext-split model.length=1024 \
  sampling.predictor=relay_ddpm sampling.steps=128 loader.eval_batch_size=8 \
  sampling.num_sample_batches=16 eval.compute_generative_perplexity=True
```

## 5. Gotchas / risks

- **Memory.** `L1 + L2` keeps two full DIT graphs alive (forward 1 through `h1`). Expect ~2× the
  activation memory of a normal MDLM step at the same per-GPU batch; start at `loader.batch_size=8`
  on 40–80 GB cards. Per-block `torch.utils.checkpoint` is the fallback (not in `DDiTBlock` today).
- **Throughput and data exposure.** One row needs `(T+1)/K ≈ 65` calls at `T=128`; a buffer of 8
  rows therefore admits ~0.12 fresh sequences per step. Fine for OWT (8M docs) but plan
  `max_steps` in *calls*, not sequences. Smaller training `T` (32/64) or sampling `T` per row from a
  set trades trajectory fidelity for diversity — keep as a later knob.
- **Incoming rows are dropped** when no slot is ready (both references do this). Don't queue them.
- **Nothing runs locally.** The Windows dev box has no `flash_attn`; every run (parity, training,
  sampling, tests) goes through `modal_app.py`. No CPU/Windows fallbacks are kept in the code.
- **Dropout 0.1 is on in both forwards** (as in the references). Keep it; it is part of the
  pretrained recipe.
- **`time_conditioning` must stay `False`** to match the checkpoint (`sigma_map` was trained with
  σ = 0).
- **Do not copy `_shift_h_s`/`_shift_logits`** from the Fast-dLLM loss; they compensate for
  next-token prediction, which MDLM does not do.
- **Resume vs. init:** with `checkpointing.resume_from_ckpt=true` a run dir that already has
  `checkpoints/last.ckpt` resumes (overriding the pretrained init and the rebuilt EMA — correct).
