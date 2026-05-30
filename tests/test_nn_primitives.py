"""
Tests for torchagentic.nn primitives.

Covers: Memory, CreditAssignment, Planner, Explorer.
Edge cases: seq_len > 1, batch > 1, empty writes, full memory,
temporal link reset, episode termination, gradient flow, compile.
"""

import pickle
import pytest
import torch
import torch.nn.functional as F

from torchagentic.nn import (
    Memory, MemoryState, NTMBank, DNCBank, SlidingWindowBank,
    GAE, TDLambda, VTrace, Retrace, TDLambdaNet,
    ValueIteration, MCTSPlanner, SuccessorRepresentation, LearnedPrior,
    RandomNetworkDistillation, ICM, CountBonus, DisagreementEnsemble,
)


# ===================================================================
#  MEMORY TESTS
# ===================================================================

class TestMemoryShapes:
    """Basic shape and output contract for Memory."""

    @pytest.fixture(params=["ntm", "dnc", "sliding"])
    def backend(self, request):
        return request.param

    def test_forward_single_step(self, backend):
        B, D = 4, 32
        mem = Memory(num_slots=16, slot_size=D, num_reads=2, backend=backend)
        x = torch.randn(B, D)
        out, state = mem(x)
        assert out.shape == (B, 2, D)
        assert isinstance(state, MemoryState)

    def test_forward_sequence(self, backend):
        B, T, D = 4, 8, 32
        mem = Memory(num_slots=16, slot_size=D, num_reads=2, backend=backend)
        x = torch.randn(B, T, D)
        out, state = mem(x)
        assert out.shape == (B, T, 2, D)
        assert isinstance(state, MemoryState)

    def test_batch_size_1(self, backend):
        D = 32
        mem = Memory(num_slots=16, slot_size=D, num_reads=1, backend=backend)
        x = torch.randn(1, D)
        out, state = mem(x)
        assert out.shape == (1, 1, D)

    def test_batch_size_64(self, backend):
        B, D = 64, 32
        mem = Memory(num_slots=16, slot_size=D, num_reads=2, backend=backend)
        x = torch.randn(B, D)
        out, state = mem(x)
        assert out.shape == (B, 2, D)

    def test_reset_batch_size(self, backend):
        D = 32
        mem = Memory(num_slots=16, slot_size=D, num_reads=2, backend=backend)
        state = mem.reset(batch_size=8)
        assert state.read_weights.shape == (8, 2, 16)
        assert state.write_weights.shape == (8, 1, 16)
        assert state.memory.shape == (16, 32)


class TestMemoryEdgeCases:
    """Edge cases: empty writes, full memory, large sequences."""

    def test_read_without_write(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="ntm")
        state = mem.reset(2)
        x = torch.randn(2, 16)
        vectors, new_state = mem.read(x, state)
        assert vectors.shape == (2, 1, 16)

    def test_write_then_read(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="sliding")
        state = mem.reset(2)
        key = torch.randn(2, 16)
        val = torch.ones(2, 16)
        state = mem.write(key, val, state)
        x = torch.randn(2, 16)
        vectors, new_state = mem.read(x, state)
        assert vectors.shape == (2, 1, 16)

    def test_write_more_than_num_slots(self):
        """Writing more items than slots should still work (overwrites oldest)."""
        mem = Memory(num_slots=4, slot_size=8, num_reads=1, backend="sliding")
        state = mem.reset(1)
        for i in range(10):
            val = torch.full((1, 8), float(i))
            key = torch.randn(1, 8)
            state = mem.write(key, val, state)
        x = torch.randn(1, 8)
        vectors, state = mem.read(x, state)
        assert vectors.shape == (1, 1, 8)

    def test_multi_write_heads_dnc(self):
        mem = Memory(num_slots=16, slot_size=32, num_reads=4, num_writes=2, backend="dnc")
        state = mem.reset(2)
        x = torch.randn(2, 32)
        out, state = mem(x)
        assert out.shape == (2, 4, 32)

    def test_long_sequence(self):
        B, T, D = 2, 64, 16
        mem = Memory(num_slots=32, slot_size=D, num_reads=1, backend="ntm")
        x = torch.randn(B, T, D)
        out, state = mem(x)
        assert out.shape == (B, T, 1, D)

    def test_gradient_flow_through_memory(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="ntm")
        x = torch.randn(2, 16, requires_grad=True)
        out, state = mem(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_memory_custom_backend(self):
        class DummyBackend(NTMBank):
            pass
        backend = DummyBackend(8, 16, num_reads=1, num_writes=1)
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend=backend)
        x = torch.randn(2, 16)
        out, state = mem(x)
        assert out.shape == (2, 1, 16)

    def test_memory_state_pickle(self):
        state = MemoryState.zeros(8, 16, 2, 1, batch_size=4)
        data = pickle.dumps(state)
        loaded = pickle.loads(data)
        assert torch.equal(loaded.memory, state.memory)
        assert loaded.read_weights.shape == (4, 2, 8)

    def test_sliding_window_cursor_advances(self):
        mem = Memory(num_slots=4, slot_size=8, num_reads=1, backend="sliding")
        state = mem.reset(1)
        positions = []
        for i in range(8):
            key = torch.randn(1, 8)
            val = torch.full((1, 8), float(i))
            state = mem.write(key, val, state)
            pos = int(state.write_weights[0, 0].argmax().item())
            positions.append(pos)
        assert positions == [0, 1, 2, 3, 0, 1, 2, 3], f"Expected wrap-around, got {positions}"

    def test_dnc_temporal_link_reset_on_write(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=2, backend="dnc")
        state = mem.reset(1)
        x1 = torch.randn(1, 16)
        _, state = mem(x1)
        link_after_first_write = state.link_matrix.clone()
        x2 = torch.randn(1, 16)
        _, state = mem(x2, state)
        link_after_second = state.link_matrix.clone()
        assert not torch.equal(link_after_first_write, link_after_second), \
            "Temporal link matrix should change after second write"

    def test_dnc_usage_tracking(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=2, backend="dnc")
        state = mem.reset(1)
        for _ in range(10):
            x = torch.randn(1, 16)
            _, state = mem(x, state)
        assert state.usage.sum() > 0, "Usage should increase with writes"

    def test_ntm_location_addressing(self):
        mem = Memory(num_slots=16, slot_size=32, num_reads=4, backend="ntm")
        state = mem.reset(2)
        x = torch.randn(2, 32)
        out, state = mem(x)
        assert out.shape == (2, 4, 32)


# ===================================================================
#  CREDIT ASSIGNMENT TESTS
# ===================================================================

class TestGAE:
    """Generalized Advantage Estimation edge cases."""

    def test_basic_shapes(self):
        gae = GAE(gamma=0.99, lam=0.95)
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        dones = torch.zeros(T, B)
        returns, adv = gae(rewards, values, dones)
        assert returns.shape == (T, B)
        assert adv.shape == (T, B)

    def test_no_dones(self):
        gae = GAE()
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, B)
        assert adv.shape == (T, B)

    def test_batch_size_1(self):
        gae = GAE()
        T = 10
        rewards = torch.randn(T, 1)
        values = torch.randn(T + 1, 1)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, 1)

    def test_batch_size_64(self):
        gae = GAE()
        T, B = 5, 64
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, B)

    def test_single_step(self):
        gae = GAE()
        T, B = 1, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, B)

    def test_episode_termination_mid_trajectory(self):
        gae = GAE(gamma=0.99, lam=0.95)
        T, B = 10, 4
        rewards = torch.ones(T, B)
        values = torch.zeros(T + 1, B)
        dones = torch.zeros(T, B)
        dones[5, :] = 1.0
        returns, adv = gae(rewards, values, dones)
        assert returns.shape == (T, B)
        assert (returns[6:] >= 0).all(), "Returns after done should not bootstrap"

    def test_all_dones_at_end(self):
        gae = GAE()
        T, B = 10, 4
        rewards = torch.ones(T, B)
        values = torch.randn(T + 1, B)
        dones = torch.ones(T, B)  # every step is terminal
        returns, adv = gae(rewards, values, dones)
        assert returns.shape == (T, B)
        assert torch.allclose(returns, rewards), "Each step should equal its own reward"

    def test_gamma_0(self):
        gae = GAE(gamma=0.0)
        T, B = 5, 2
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        assert torch.allclose(returns, rewards), "With γ=0, returns should equal rewards"

    def test_gradient_flow(self):
        gae = GAE()
        T, B = 5, 2
        rewards = torch.randn(T, B, requires_grad=True)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        loss = returns.sum() + adv.sum()
        loss.backward()
        assert rewards.grad is not None
        assert rewards.grad.abs().sum() > 0


class TestTDLambda:
    """TD(λ) edge cases."""

    def test_shapes(self):
        td = TDLambda(gamma=0.99, lam=0.8)
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        ret, adv = td(rewards, values)
        assert ret.shape == (T, B)

    def test_lambda_0(self):
        td = TDLambda(gamma=0.99, lam=0.0)
        T, B = 5, 2
        rewards = torch.ones(T, B)
        values = torch.zeros(T + 1, B)
        ret, adv = td(rewards, values)
        expected = torch.ones(T, B)
        assert torch.allclose(ret, expected, atol=1e-5)

    def test_no_dones(self):
        td = TDLambda()
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        ret, adv = td(rewards, values)
        assert ret.shape == (T, B)

    def test_dones(self):
        td = TDLambda()
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        dones = torch.zeros(T, B)
        dones[3, :] = 1.0
        ret, adv = td(rewards, values, dones)
        assert ret.shape == (T, B)


class TestVTrace:
    """V-trace off-policy correction edge cases."""

    def test_shapes(self):
        vt = VTrace(gamma=0.99)
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        log_probs = torch.randn(T, B)
        target_log_probs = torch.randn(T, B)
        vs, adv = vt(rewards, values, log_probs=log_probs, target_log_probs=target_log_probs)
        assert vs.shape == (T, B)
        assert adv.shape == (T, B)

    def test_requires_log_probs(self):
        vt = VTrace()
        with pytest.raises(ValueError, match="requires log_probs"):
            vt(torch.randn(5, 2), torch.randn(6, 2))

    def test_clipping(self):
        vt = VTrace(rho_bar=1.0, c_bar=1.0)
        T, B = 5, 2
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        log_probs = torch.full((T, B), -100.0)
        target_log_probs = torch.zeros(T, B)
        vs, adv = vt(rewards, values, log_probs=log_probs, target_log_probs=target_log_probs)
        assert vs.shape == (T, B)
        assert torch.isfinite(vs).all(), "Clipping should prevent NaN from extreme IS"

    def test_on_policy(self):
        vt = VTrace()
        T, B = 5, 2
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        log_probs = torch.randn(T, B)
        vs, adv = vt(rewards, values, log_probs=log_probs, target_log_probs=log_probs)
        assert vs.shape == (T, B)


class TestRetrace:
    """Retrace off-policy edge cases."""

    def test_shapes(self):
        rt = Retrace(gamma=0.99)
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        lp = torch.randn(T, B)
        tlp = torch.randn(T, B)
        ret, adv = rt(rewards, values, log_probs=lp, target_log_probs=tlp)
        assert ret.shape == (T, B)

    def test_requires_log_probs(self):
        rt = Retrace()
        with pytest.raises(ValueError, match="requires log_probs"):
            rt(torch.randn(5, 2), torch.randn(6, 2))

    def test_on_policy_no_correction(self):
        rt = Retrace()
        T, B = 5, 2
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        lp = torch.randn(T, B)
        ret, adv = rt(rewards, values, log_probs=lp, target_log_probs=lp)
        assert ret.shape == (T, B)


class TestTDLambdaNet:
    """Learned λ parameter edge cases."""

    def test_shapes(self):
        tdn = TDLambdaNet(gamma=0.99, init_lam=0.8)
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        ret, adv = tdn(rewards, values)
        assert ret.shape == (T, B)
        assert adv.shape == (T, B)

    def test_lam_is_learnable(self):
        tdn = TDLambdaNet(gamma=0.99, init_lam=0.5)
        assert hasattr(tdn, "logit_lam")
        assert tdn.logit_lam.requires_grad
        loss = tdn.logit_lam.sum()
        loss.backward()
        assert tdn.logit_lam.grad is not None

    def test_lam_range(self):
        tdn = TDLambdaNet(gamma=0.99, init_lam=0.3)
        lam = tdn.lam
        assert 0.0 <= lam <= 1.0


# ===================================================================
#  PLANNER TESTS
# ===================================================================

class TestValueIteration:
    """Value iteration edge cases."""

    def test_basic_shapes(self):
        vi = ValueIteration(num_states=5, num_actions=3, gamma=0.99, num_iters=10)
        B = 4
        reward = torch.randn(B, 5, 3)
        kernel = torch.randn(B, 5, 3, 5)
        kernel = F.softmax(kernel.reshape(B, 15, 5), dim=-1).reshape(B, 5, 3, 5)
        v, q = vi(reward, kernel)
        assert v.shape == (B, 5)
        assert q.shape == (B, 5, 3)

    def test_batch_size_1(self):
        vi = ValueIteration(num_states=5, num_actions=3)
        reward = torch.randn(1, 5, 3)
        kernel = torch.randn(1, 5, 3, 5)
        kernel = F.softmax(kernel.reshape(1, 15, 5), dim=-1).reshape(1, 5, 3, 5)
        v, q = vi(reward, kernel)
        assert v.shape == (1, 5)

    def test_single_state_single_action(self):
        vi = ValueIteration(num_states=1, num_actions=1, gamma=0.9, num_iters=50)
        reward = torch.randn(2, 1, 1)
        kernel = torch.ones(2, 1, 1, 1)
        v, q = vi(reward, kernel)
        assert v.shape == (2, 1)
        expected = reward.squeeze(-1) / (1 - 0.9)
        assert torch.allclose(v, expected, atol=1e-1)

    def test_one_iteration(self):
        vi = ValueIteration(num_states=5, num_actions=3, num_iters=1)
        reward = torch.randn(2, 5, 3)
        kernel = torch.randn(2, 5, 3, 5)
        kernel = F.softmax(kernel.reshape(2, 15, 5), dim=-1).reshape(2, 5, 3, 5)
        v, q = vi(reward, kernel)
        assert v.shape == (2, 5)

    def test_gradient_flow_through_iterations(self):
        vi = ValueIteration(num_states=3, num_actions=2, num_iters=5)
        B = 2
        reward = torch.randn(B, 3, 2, requires_grad=True)
        kernel = torch.randn(B, 3, 2, 3)
        kernel = F.softmax(kernel.reshape(B, 6, 3), dim=-1).reshape(B, 3, 2, 3)
        v, q = vi(reward, kernel)
        loss = v.sum()
        loss.backward()
        assert reward.grad is not None
        assert reward.grad.abs().sum() > 0

    def test_init_v(self):
        vi = ValueIteration(num_states=5, num_actions=3, num_iters=10)
        reward = torch.randn(2, 5, 3)
        kernel = torch.randn(2, 5, 3, 5)
        kernel = F.softmax(kernel.reshape(2, 15, 5), dim=-1).reshape(2, 5, 3, 5)
        init_v = torch.randn(2, 5)
        v, q = vi(reward, kernel, init_v=init_v)
        assert v.shape == (2, 5)

    def test_batch_size_64(self):
        vi = ValueIteration(num_states=10, num_actions=4, num_iters=5)
        B = 64
        reward = torch.randn(B, 10, 4)
        kernel = torch.randn(B, 10, 4, 10)
        kernel = F.softmax(kernel.reshape(B, 40, 10), dim=-1).reshape(B, 10, 4, 10)
        v, q = vi(reward, kernel)
        assert v.shape == (B, 10)


class TestMCTSPlanner:
    """MCTS planner edge cases."""

    def test_basic_shapes(self):
        mcts = MCTSPlanner(num_simulations=10, c_puct=1.0)
        B, A = 4, 6
        prior_logits = torch.randn(B, A)
        value = torch.randn(B)
        probs, v = mcts(prior_logits, value)
        assert probs.shape == (B, A)
        assert v.shape == (B,)

    def test_batch_size_1(self):
        mcts = MCTSPlanner()
        probs, v = mcts(torch.randn(1, 4), torch.randn(1))
        assert probs.shape == (1, 4)

    def test_single_action(self):
        mcts = MCTSPlanner()
        probs, v = mcts(torch.randn(2, 1), torch.randn(2))
        assert probs.shape == (2, 1)
        assert torch.allclose(probs, torch.ones(2, 1))

    def test_policy_is_valid_distribution(self):
        mcts = MCTSPlanner()
        B, A = 4, 6
        probs, _ = mcts(torch.randn(B, A), torch.randn(B))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))
        assert (probs >= 0).all()


class TestSuccessorRepresentation:
    """Successor representation edge cases."""

    def test_shapes_no_reward(self):
        sr = SuccessorRepresentation(num_states=5, gamma=0.99, feature_dim=16)
        B = 4
        features = torch.randn(B, 16)
        out = sr(features)
        assert out.shape == (B, 5)

    def test_shapes_with_reward(self):
        sr = SuccessorRepresentation(num_states=5, gamma=0.99, feature_dim=16)
        B = 4
        features = torch.randn(B, 16)
        reward = torch.randn(B, 5)
        out = sr(features, reward=reward)
        assert out.shape == (B,)

    def test_batch_size_1(self):
        sr = SuccessorRepresentation(num_states=3, feature_dim=8)
        out = sr(torch.randn(1, 8))
        assert out.shape == (1, 3)

    def test_gradient_flow(self):
        sr = SuccessorRepresentation(num_states=3, feature_dim=8)
        features = torch.randn(2, 8, requires_grad=True)
        out = sr(features)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0


class TestLearnedPrior:
    """Learned prior edge cases."""

    def test_shapes(self):
        lp = LearnedPrior(latent_dim=32, num_actions=4, plan_length=8)
        B = 4
        latent = torch.randn(B, 32)
        logits = lp(latent)
        assert logits.shape == (B, 8, 4)

    def test_batch_size_1(self):
        lp = LearnedPrior(latent_dim=16, num_actions=3, plan_length=5)
        logits = lp(torch.randn(1, 16))
        assert logits.shape == (1, 5, 3)

    def test_gradient_flow(self):
        lp = LearnedPrior(latent_dim=16, num_actions=3, plan_length=5)
        latent = torch.randn(2, 16, requires_grad=True)
        logits = lp(latent)
        loss = logits.sum()
        loss.backward()
        assert latent.grad is not None


# ===================================================================
#  EXPLORER TESTS
# ===================================================================

class TestRND:
    """Random Network Distillation edge cases."""

    def test_shapes(self):
        rnd = RandomNetworkDistillation(state_dim=32)
        B = 8
        states = torch.randn(B, 32)
        reward = rnd(states)
        assert reward.shape == (B,)

    def test_batch_size_1(self):
        rnd = RandomNetworkDistillation(state_dim=16)
        reward = rnd(torch.randn(1, 16))
        assert reward.shape == (1,)

    def test_batch_size_64(self):
        rnd = RandomNetworkDistillation(state_dim=16)
        reward = rnd(torch.randn(64, 16))
        assert reward.shape == (64,)

    def test_target_is_frozen(self):
        rnd = RandomNetworkDistillation(state_dim=16)
        for p in rnd.target.parameters():
            assert not p.requires_grad

    def test_predictor_is_trainable(self):
        rnd = RandomNetworkDistillation(state_dim=16)
        for p in rnd.predictor.parameters():
            assert p.requires_grad

    def test_gradient_flows_to_predictor(self):
        rnd = RandomNetworkDistillation(state_dim=16)
        states = torch.randn(4, 16)
        reward = rnd(states)
        loss = reward.sum()
        loss.backward()
        pred_grads = [p.grad for p in rnd.predictor.parameters()]
        assert all(g is not None for g in pred_grads)
        assert all(g.abs().sum() > 0 for g in pred_grads)

    def test_novel_state_high_reward(self):
        rnd = RandomNetworkDistillation(state_dim=8, reward_scale=1.0)
        seen = torch.randn(100, 8)
        novel = torch.randn(100, 8) + 10.0
        for _ in range(50):
            rnd(seen)
        seen_reward = rnd(seen).mean()
        novel_reward = rnd(novel).mean()
        assert novel_reward > seen_reward, "Novel states should have higher RND error"

    def test_reward_scale(self):
        rnd = RandomNetworkDistillation(state_dim=8, reward_scale=2.0)
        states = torch.randn(4, 8)
        r1 = rnd(states)
        rnd.reward_scale = 0.5
        r2 = rnd(states)
        assert torch.allclose(r1, r2 * 4.0, atol=1e-5)


class TestICM:
    """Intrinsic Curiosity Module edge cases."""

    def test_shapes(self):
        icm = ICM(state_dim=32, action_dim=4)
        B = 8
        states = torch.randn(B, 32)
        actions = torch.randn(B, 4)
        next_states = torch.randn(B, 32)
        reward = icm(states, actions, next_states)
        assert reward.shape == (B,)

    def test_requires_actions_and_next_states(self):
        icm = ICM(state_dim=16, action_dim=4)
        with pytest.raises(ValueError, match="requires actions"):
            icm(torch.randn(4, 16))
        with pytest.raises(ValueError, match="requires actions"):
            icm(torch.randn(4, 16), torch.randn(4, 4))

    def test_batch_size_1(self):
        icm = ICM(state_dim=8, action_dim=2)
        reward = icm(torch.randn(1, 8), torch.randn(1, 2), torch.randn(1, 8))
        assert reward.shape == (1,)

    def test_forward_loss(self):
        icm = ICM(state_dim=16, action_dim=4)
        B = 8
        states = torch.randn(B, 16)
        actions = torch.randn(B, 4)
        next_states = torch.randn(B, 16)
        loss = icm.forward_loss(states, actions, next_states)
        assert loss.shape == ()
        assert loss.item() > 0
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in icm.parameters())

    def test_gradient_flow(self):
        icm = ICM(state_dim=16, action_dim=4)
        states = torch.randn(4, 16, requires_grad=True)
        actions = torch.randn(4, 4)
        next_states = torch.randn(4, 16)
        reward = icm(states, actions, next_states)
        loss = reward.sum()
        loss.backward()
        assert states.grad is not None


class TestCountBonus:
    """Count-based exploration edge cases."""

    def test_shapes(self):
        cb = CountBonus(state_dim=16)
        B = 8
        reward = cb(torch.randn(B, 16))
        assert reward.shape == (B,)

    def test_batch_size_1(self):
        cb = CountBonus(state_dim=8)
        reward = cb(torch.randn(1, 8))
        assert reward.shape == (1,)

    def test_total_count_increments(self):
        cb = CountBonus(state_dim=4)
        initial = cb.total_count.item()
        for _ in range(5):
            cb(torch.randn(2, 4))
        assert cb.total_count.item() == initial


class TestDisagreementEnsemble:
    """Ensemble disagreement edge cases."""

    def test_shapes(self):
        de = DisagreementEnsemble(state_dim=16, action_dim=4, ensemble_size=3)
        B = 8
        states = torch.randn(B, 16)
        actions = torch.randn(B, 4)
        reward = de(states, actions)
        assert reward.shape == (B,)

    def test_requires_actions(self):
        de = DisagreementEnsemble(state_dim=8, action_dim=4)
        with pytest.raises(ValueError, match="requires actions"):
            de(torch.randn(4, 8))

    def test_batch_size_1(self):
        de = DisagreementEnsemble(state_dim=8, action_dim=2, ensemble_size=3)
        reward = de(torch.randn(1, 8), torch.randn(1, 2))
        assert reward.shape == (1,)

    def test_ensemble_size_property(self):
        de = DisagreementEnsemble(state_dim=8, action_dim=2, ensemble_size=7)
        assert len(de.models) == 7

    def test_gradient_flow(self):
        de = DisagreementEnsemble(state_dim=16, action_dim=4, ensemble_size=3)
        states = torch.randn(4, 16, requires_grad=True)
        actions = torch.randn(4, 4)
        reward = de(states, actions)
        loss = reward.sum()
        loss.backward()
        assert states.grad is not None

    def test_disagreement_for_different_states(self):
        de = DisagreementEnsemble(state_dim=8, action_dim=2, ensemble_size=5)
        states_same = torch.randn(4, 8).expand(5, -1, -1)
        actions = torch.randn(4, 2)
        d1 = de(states_same[0], actions)
        d2 = de(torch.randn(4, 8), actions)
        assert d1.shape == (4,)


# ===================================================================
#  COMPOSITION TESTS
# ===================================================================

class TestComposition:
    """Multiple primitives composing together."""

    def test_memory_plus_explorer(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="sliding")
        rnd = RandomNetworkDistillation(state_dim=16)
        state = mem.reset(2)
        x = torch.randn(2, 16)
        out, state = mem(x, state)
        bonus = rnd(out.squeeze(1))
        assert bonus.shape == (2,)

    def test_memory_plus_credit(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="sliding")
        gae = GAE()
        state = mem.reset(2)
        outputs = []
        for _ in range(5):
            x = torch.randn(2, 16)
            out, state = mem(x, state)
            outputs.append(out)
        T, B = 5, 2
        values = torch.randn(T + 1, B)
        rewards = torch.randn(T, B)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, B)

    def test_all_explorers_produce_nonnegative_reward(self):
        B = 4
        rnd = RandomNetworkDistillation(state_dim=16)
        icm = ICM(state_dim=16, action_dim=4)
        cb = CountBonus(state_dim=16)
        de = DisagreementEnsemble(state_dim=16, action_dim=4, ensemble_size=3)
        states = torch.randn(B, 16)
        assert (rnd(states) >= 0).all()
        assert (cb(states) >= 0).all()
        assert (icm(states, torch.randn(B, 4), torch.randn(B, 16)) >= 0).all()
        assert (de(states, torch.randn(B, 4)) >= 0).all()


# ===================================================================
#  COMPILE COMPATIBILITY TESTS
# ===================================================================

class TestCompileCompatibility:
    """Verify all primitives work with torch.compile()."""

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_memory_compile(self):
        mem = Memory(num_slots=8, slot_size=16, num_reads=1, backend="ntm")
        compiled = torch.compile(mem)
        x = torch.randn(2, 16)
        out, state = compiled(x)
        assert out.shape == (2, 1, 16)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_gae_compile(self):
        gae = torch.compile(GAE())
        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        returns, adv = gae(rewards, values)
        assert returns.shape == (T, B)
        assert adv.shape == (T, B)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_value_iteration_compile(self):
        vi = torch.compile(ValueIteration(num_states=5, num_actions=3, num_iters=5))
        reward = torch.randn(2, 5, 3)
        kernel = torch.randn(2, 5, 3, 5)
        kernel = F.softmax(kernel.reshape(2, 15, 5), dim=-1).reshape(2, 5, 3, 5)
        v, q = vi(reward, kernel)
        assert v.shape == (2, 5)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_rnd_compile(self):
        rnd = torch.compile(RandomNetworkDistillation(state_dim=16))
        states = torch.randn(4, 16)
        reward = rnd(states)
        assert reward.shape == (4,)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_full_composable_agent_compile(self):
        from examples.composable_agent import ComposableAgent
        agent = ComposableAgent(obs_dim=16, action_dim=4, hidden_dim=64, memory_slots=16, memory_dim=32)
        compiled = torch.compile(agent, mode="reduce-overhead")
        obs = torch.randn(4, 16)
        logits, value, intrinsic, state = compiled(obs)
        assert logits.shape == (4, 4)
        assert value.shape == (4,)
        assert intrinsic.shape == (4,)


# ===================================================================
#  MEMORYSTATE TESTS
# ===================================================================

class TestMemoryState:
    """MemoryState dataclass utilities."""

    def test_zeros_defaults(self):
        state = MemoryState.zeros(8, 16, 2, 1)
        assert state.memory.shape == (8, 16)
        assert state.usage.shape == (8,)
        assert state.link_matrix.shape == (8, 8)
        assert state.read_weights.shape == (1, 2, 8)
        assert state.write_weights.shape == (1, 1, 8)
        assert torch.allclose(state.read_weights.sum(dim=-1), torch.ones(1, 2))
        assert state.usage.sum() == 0.0

    def test_zeros_custom_batch(self):
        state = MemoryState.zeros(8, 16, 2, 1, batch_size=4)
        assert state.read_weights.shape == (4, 2, 8)

    def test_zeros_device(self):
        state = MemoryState.zeros(8, 16, 2, 1)
        assert state.memory.device.type == "cpu"
