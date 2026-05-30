# torchagentic.nn — API Reference

Composable, compile-optimized PyTorch primitives for agentic AI.

## Memory

### `nn.Memory`
Differentiable external memory with pluggable addressing.

```python
mem = Memory(num_slots=128, slot_size=64, num_reads=4, backend="dnc")
out, state = mem(x)          # read + write, returns (B, num_reads, D)
out, state = mem.read(x)     # read only
state = mem.write(key, val)  # write only
state = mem.reset(B)         # fresh state
```

**Reference:** Graves et al., "Neural Turing Machines", 2014; Graves et al., "Hybrid computing using a neural network with dynamic external memory", Nature 2016.

### `nn.MemoryState`
Pure-tensor dataclass holding all memory state. Every field is a tensor — no `None` fields, compatible with `torch.compile()`.

- `memory: (num_slots, slot_size)` — storage matrix
- `usage: (num_slots,)` — how recently each slot was used
- `link_matrix: (num_slots, num_slots)` — temporal write-order links
- `read_weights: (B, num_reads, num_slots)` — attention over slots for reading
- `write_weights: (B, num_writes, num_slots)` — attention over slots for writing

### `nn.NTMBank`
NTM-style content + location addressing (interpolation, circular shift, sharpening).

### `nn.DNCBank`
DNC-style content + dynamic allocation + temporal link addressing.

### `nn.SlidingWindowBank`
Fixed sliding window (FIFO) — no learned parameters. Baseline/ablation.

---

## Credit Assignment

### `nn.GAE`
Generalized Advantage Estimation (Schulman et al., 2015). Standard for PPO.

```python
gae = GAE(gamma=0.99, lam=0.95)
returns, advantages = gae(rewards, values, dones)
# rewards: (T, B), values: (T+1, B), dones: (T, B)
```

**Reference:** https://arxiv.org/abs/1506.02438

### `nn.TDLambda`
TD(λ) — classic temporal-difference return. λ=0 → 1-step TD, λ=1 → Monte Carlo.

**Reference:** Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed.

### `nn.VTrace`
Off-policy correction for IMPALA (Espeholt et al., 2018). Clips importance weights.

```python
vt = VTrace(rho_bar=1.0, c_bar=1.0)
vs, adv = vt(rewards, values, dones, log_probs, target_log_probs)
```

**Reference:** https://arxiv.org/abs/1802.01561

### `nn.Retrace`
Safe off-policy TD(λ) with truncated importance weights (Munos et al., 2016).

**Reference:** https://arxiv.org/abs/1606.01247

### `nn.TDLambdaNet`
TD(λ) with a *learned* λ parameter. λ is a `nn.Parameter` trained via gradient descent.

---

## Planning

### `nn.ValueIteration`
Differentiable value iteration — unrolls a fixed number of VI steps.

```python
vi = ValueIteration(num_states=32, num_actions=8, gamma=0.99, num_iters=20)
values, q_values = vi(reward, kernel)
# reward: (B, S, A), kernel: (B, S, A, S) — transition probabilities
```

Gradients flow through all iterations, shaping the reward/transition models.

**Reference:** Classic dynamic programming; adapted for differentiable computation graphs.

### `nn.MCTSPlanner`
Differentiable Monte-Carlo Tree Search (simplified soft-search relaxation).

```python
mcts = MCTSPlanner(num_simulations=50, c_puct=1.25)
action_probs, value = mcts(prior_logits, value)
```

**Reference:** Silver et al., "Mastering the game of Go with deep neural networks and tree search", Nature 2016.

### `nn.SuccessorRepresentation`
Successor representation (Dayan, 1993). Decouples rewards from transitions.

```python
sr = SuccessorRepresentation(num_states=5, feature_dim=64)
values = sr(features, reward=reward)  # (B,)
sr_matrix = sr(features)               # (B, num_states)
```

**Reference:** Dayan, "Improving Generalization for Temporal Difference Learning", 1993.

### `nn.LearnedPrior`
Autoregressive LSTM that outputs a distribution over action sequences (plans).

```python
lp = LearnedPrior(latent_dim=64, num_actions=8, plan_length=10)
plan_logits = lp(latent)  # (B, plan_length, num_actions)
```

---

## Exploration

### `nn.RandomNetworkDistillation`
Fixed random target network + trainable predictor. Prediction error = novelty.

```python
rnd = RandomNetworkDistillation(state_dim=64, reward_scale=1.0)
intrinsic_reward = rnd(states)  # (B,)
```

**Reference:** Burda et al., "Exploration by Random Network Distillation", 2018. https://arxiv.org/abs/1810.12894

### `nn.ICM`
Intrinsic Curiosity Module — forward dynamics prediction error.

```python
icm = ICM(state_dim=64, action_dim=8)
intrinsic_reward = icm(states, actions, next_states)  # (B,)
```

**Reference:** Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", 2017. https://arxiv.org/abs/1705.05363

### `nn.CountBonus`
Count-based exploration via learned density model.

```python
cb = CountBonus(state_dim=64)
intrinsic_reward = cb(states)  # (B,)
```

### `nn.DisagreementEnsemble`
Ensemble of forward dynamics models — disagreement = epistemic uncertainty.

```python
de = DisagreementEnsemble(state_dim=64, action_dim=8, ensemble_size=5)
intrinsic_reward = de(states, actions)  # (B,)
```
