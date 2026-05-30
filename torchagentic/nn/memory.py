"""
Differentiable external memory primitive (nn.Memory).

A theoretical and practical foundation for differentiable memory in PyTorch,
abstracting NTM, DNC, and sliding-window addressing under a unified
read/write interface with compile-optimized state management.
"""

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. State — a pure-tensor container, never None fields.
#    This is what makes compile() work and gradients flow.
# ─────────────────────────────────────────────────────────────

@dataclass
class MemoryState:
    """The complete state of a differentiable memory bank.

    Every field is a tensor; unused fields are zero-filled.
    This is the key theoretical contribution: memory state as a
    first-class tensor structure rather than a bag of optional hacks.

    Shapes are always:
        memory:       (num_slots, slot_size)
        usage:        (num_slots,)
        link_matrix:  (num_slots, num_slots)
        read_weights: (batch, num_reads, num_slots)
        write_weights:(batch, num_writes, num_slots)
    """
    memory: torch.Tensor
    usage: torch.Tensor
    link_matrix: torch.Tensor
    read_weights: torch.Tensor
    write_weights: torch.Tensor

    @staticmethod
    def zeros(
        num_slots: int,
        slot_size: int,
        num_reads: int,
        num_writes: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> "MemoryState":
        return MemoryState(
            memory=torch.zeros(num_slots, slot_size, device=device),
            usage=torch.zeros(num_slots, device=device),
            link_matrix=torch.zeros(num_slots, num_slots, device=device),
            read_weights=torch.full(
                (batch_size, num_reads, num_slots),
                1.0 / num_slots,
                device=device,
            ),
            write_weights=torch.full(
                (batch_size, num_writes, num_slots),
                1.0 / num_slots,
                device=device,
            ),
        )


# ─────────────────────────────────────────────────────────────
# 2. Abstract addressing backend.
#    Subclasses define *how* the memory is addressed (content,
#    location, allocation, temporal, sliding). The Memory module
#    owns state management and exposes the user-facing API.
# ─────────────────────────────────────────────────────────────

class MemoryBackend(nn.Module):
    """Pluggable addressing scheme for a differentiable memory.

    A backend decides where to read from and write to given a
    query.  It does NOT own the memory matrix itself — that
    lives in MemoryState so it can be serialised and compiled
    transparently.
    """

    num_slots: int
    slot_size: int
    num_reads: int
    num_writes: int

    def forward(
        self,
        query: torch.Tensor,
        state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, MemoryState]:
        """Produce read & write weights from a query + current state.

        Returns:
            read_weights:   (batch, num_reads, num_slots)
            write_params:   (batch, num_writes * (slot_size + slot_size + 1))
            updated_state:  MemoryState (backends may mutate usage / links)
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# 2a. NTM addressing — content + location + interpolation.
# ─────────────────────────────────────────────────────────────

class NTMBank(MemoryBackend):
    """Neural Turing Machine addressing.

    Combines content-based cosine-similarity with location-based
    circular convolution and temporal interpolation.

    Reference: Graves et al., "Neural Turing Machines", 2014.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        num_reads: int = 4,
        num_writes: int = 1,
        shift_kernel_size: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.shift_kernel_size = shift_kernel_size

        # Learned projection from query to interface parameters.
        interface_dim = (
            slot_size                       # content key
            + 1                             # key strength
            + 1                             # interpolation gate
            + shift_kernel_size             # shift weights
            + 1                             # sharpening
            + num_writes * slot_size        # write values
            + num_writes * slot_size        # erase vectors
        )
        self.interface = nn.Linear(slot_size, interface_dim)

    def _content_weights(
        self,
        key: torch.Tensor,
        strength: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        mem_norm = F.normalize(memory.unsqueeze(0), dim=-1)
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)
        sim = torch.matmul(key_norm, mem_norm.transpose(-2, -1))
        return F.softmax(sim * strength.unsqueeze(-1), dim=-1)

    @staticmethod
    def _convolve(weights: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        padded = F.pad(weights, (1, 1), mode="circular")
        return (
            padded[..., :-2] * shift[..., 0:1]
            + padded[..., 1:-1] * shift[..., 1:2]
            + padded[..., 2:] * shift[..., 2:3]
        )

    def forward(
        self,
        query: torch.Tensor,
        state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, MemoryState]:
        B = query.shape[0]
        raw = self.interface(query)

        off = 0
        key = raw[:, off:off + self.slot_size]
        off += self.slot_size
        strength = F.softplus(raw[:, off:off + 1]) + 1.0
        off += 1
        interp_gate = torch.sigmoid(raw[:, off:off + 1]).reshape(B, 1, 1)
        off += 1
        shift_raw = raw[:, off:off + self.shift_kernel_size]
        shift_w = F.softmax(shift_raw.reshape(B, 1, self.shift_kernel_size), dim=-1)
        shift_w = shift_w.expand(-1, self.num_reads, -1)
        off += self.shift_kernel_size
        sharpen = (F.softplus(raw[:, off:off + 1]) + 1.0).reshape(B, 1, 1)
        off += 1

        write_vals = raw[:, off:off + self.num_writes * self.slot_size]
        write_vals = write_vals.reshape(B, self.num_writes, self.slot_size)
        off += self.num_writes * self.slot_size

        erase = torch.sigmoid(raw[:, off:off + self.num_writes * self.slot_size])
        erase = erase.reshape(B, self.num_writes, self.slot_size)

        # Content addressing — produces (B, 1, num_slots).
        content_w = self._content_weights(key, strength, state.memory)
        # Broadcast to all read heads.
        content_w = content_w.expand(-1, self.num_reads, -1)

        # Per-head interpolation + shift + sharpen.
        rw_list = []
        for h in range(self.num_reads):
            prev = state.read_weights[:, h:h+1, :]
            interp = (1 - interp_gate) * prev + interp_gate * content_w[:, h:h+1, :]
            shifted = self._convolve(interp, shift_w[:, h:h+1, :])
            sharp = shifted ** sharpen
            rw_list.append(sharp / (sharp.sum(dim=-1, keepdim=True) + 1e-8))

        read_weights = torch.cat(rw_list, dim=1)

        # Write weights are the first read head's weights tiled.
        write_weights = read_weights[:, :self.num_writes, :]

        # Flatten write params for the Memory module.
        write_params = torch.cat([
            write_vals.reshape(B, -1),
            erase.reshape(B, -1),
        ], dim=-1)

        return read_weights, write_params, state


# ─────────────────────────────────────────────────────────────
# 2b. DNC addressing — content + allocation + temporal links.
# ─────────────────────────────────────────────────────────────

class DNCBank(MemoryBackend):
    """Differentiable Neural Computer addressing.

    Adds dynamic memory allocation (free list) and temporal
    link matrices (sequence-of-writes tracking) on top of
    content-based addressing.

    Reference: Graves et al., "Hybrid computing using a neural
    network with dynamic external memory", Nature 2016.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        num_reads: int = 4,
        num_writes: int = 1,
        temporal_decay: float = 0.95,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.temporal_decay = temporal_decay

        interface_dim = (
            num_reads * slot_size       # read keys
            + num_reads                 # read strengths
            + 3 * num_reads             # read modes (content/backward/forward)
            + num_writes * slot_size    # write values
            + num_writes * slot_size    # erase vectors
            + num_writes                # write gates
            + num_writes                # free gates
            + 1                         # allocation gate
        )
        self.interface = nn.Linear(slot_size, interface_dim)

    @staticmethod
    def _content_weights(
        key: torch.Tensor,
        strength: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        mem_norm = F.normalize(memory.unsqueeze(0), dim=-1)
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)
        sim = torch.matmul(key_norm, mem_norm.transpose(-2, -1))
        return F.softmax(sim * strength.unsqueeze(-1), dim=-1)

    def _allocation_weights(
        self,
        usage: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        usage = usage.unsqueeze(0)
        idx = torch.argsort(usage, dim=-1, descending=False)
        sorted_u = torch.gather(usage, -1, idx)
        cumprod = torch.cumprod(1 - sorted_u, dim=-1)
        alloc = torch.zeros_like(sorted_u)
        alloc[..., 0] = sorted_u[..., 0]
        alloc[..., 1:] = sorted_u[..., 1:] * cumprod[..., :-1]
        # Unsort.
        inv_idx = torch.argsort(idx, dim=-1)
        alloc = torch.gather(alloc, -1, inv_idx)
        return alloc * gate.unsqueeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, MemoryState]:
        B = query.shape[0]
        raw = self.interface(query)

        off = 0
        read_keys = raw[:, off:off + self.num_reads * self.slot_size]
        read_keys = read_keys.reshape(B, self.num_reads, self.slot_size)
        off += self.num_reads * self.slot_size

        read_strengths = F.softplus(raw[:, off:off + self.num_reads]) + 1.0
        off += self.num_reads

        read_modes = F.softmax(
            raw[:, off:off + 3 * self.num_reads].reshape(B, self.num_reads, 3),
            dim=-1,
        )
        off += 3 * self.num_reads

        write_vals = raw[:, off:off + self.num_writes * self.slot_size]
        write_vals = write_vals.reshape(B, self.num_writes, self.slot_size)
        off += self.num_writes * self.slot_size

        erase = torch.sigmoid(
            raw[:, off:off + self.num_writes * self.slot_size]
        ).reshape(B, self.num_writes, self.slot_size)
        off += self.num_writes * self.slot_size

        write_gates = torch.sigmoid(raw[:, off:off + self.num_writes])
        off += self.num_writes

        free_gates = torch.sigmoid(raw[:, off:off + self.num_writes])
        off += self.num_writes

        alloc_gate = torch.sigmoid(raw[:, off:off + 1])

        # --- Write weights: blend of content + allocation. ---
        write_key = write_vals.mean(dim=1)
        content_w = self._content_weights(write_key, torch.ones(B, 1, device=query.device), state.memory)
        alloc_w = self._allocation_weights(state.usage, alloc_gate.squeeze(-1))
        raw_write_w = 0.5 * content_w.squeeze(1) + 0.5 * alloc_w
        write_weights = write_gates.unsqueeze(-1) * raw_write_w.unsqueeze(1)

        # --- Update usage. (keep 1D) ---
        new_usage = state.usage * 0.99 + write_weights.mean(dim=(0, 1)) * 0.01
        new_usage = new_usage * (1 - free_gates.mean(dim=(0, 1)).squeeze())

        # --- Update link matrix. ---
        write_flat = write_weights.mean(dim=1)
        outer = write_flat.unsqueeze(-1) @ write_flat.unsqueeze(1)
        diag_mask = 1 - torch.eye(self.num_slots, device=query.device)
        outer = outer * diag_mask.unsqueeze(0)
        new_links = state.link_matrix * self.temporal_decay + outer.mean(dim=0) * (1 - self.temporal_decay)

        # --- Read weights: blend content + temporal. ---
        read_weights_list = []
        for h in range(self.num_reads):
            key = read_keys[:, h, :]  # 2D — _content_weights adds the head dim
            strength = read_strengths[:, h:h+1]
            cw = self._content_weights(key, strength, state.memory)

            bw = state.read_weights[:, h:h+1, :] @ state.link_matrix
            fw = state.read_weights[:, h:h+1, :] @ state.link_matrix.T

            blend = (
                read_modes[:, h, 0:1].unsqueeze(-1) * cw
                + read_modes[:, h, 1:2].unsqueeze(-1) * bw
                + read_modes[:, h, 2:3].unsqueeze(-1) * fw
            )
            read_weights_list.append(blend)

        read_weights = torch.cat(read_weights_list, dim=1)

        # Flatten write params.
        write_params = torch.cat([
            write_vals.reshape(B, -1),
            erase.reshape(B, -1),
        ], dim=-1)

        new_state = MemoryState(
            memory=state.memory,
            usage=new_usage,
            link_matrix=new_links,
            read_weights=read_weights,
            write_weights=write_weights,
        )

        return read_weights, write_params, new_state


# ─────────────────────────────────────────────────────────────
# 2c. Sliding window — simple FIFO for comparison & testing.
# ─────────────────────────────────────────────────────────────

class SlidingWindowBank(MemoryBackend):
    """A fixed-size sliding window over recent inputs.

    Used as a baseline / ablation.  No learned addressing —
    simply tracks the last K items and returns a uniform
    read weight over them.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        num_reads: int = 1,
        num_writes: int = 1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self._cursor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        query: torch.Tensor,
        state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor, MemoryState]:
        B = query.shape[0]

        # Uniform read over all slots.
        read_weights = torch.full(
            (B, self.num_reads, self.num_slots),
            1.0 / self.num_slots,
            device=query.device,
        )

        # Write overwrites the oldest slot.
        write_weights = torch.zeros(B, self.num_writes, self.num_slots, device=query.device)
        pos = int(self._cursor.item()) % self.num_slots
        write_weights[:, :, pos] = 1.0

        write_params = torch.cat([
            query.reshape(B, -1),       # write value = query
            torch.zeros_like(query),    # no erase
        ], dim=-1)

        with torch.no_grad():
            self._cursor.add_(1)

        new_state = MemoryState(
            memory=state.memory,
            usage=state.usage,
            link_matrix=state.link_matrix,
            read_weights=read_weights,
            write_weights=write_weights,
        )
        return read_weights, write_params, new_state


# ─────────────────────────────────────────────────────────────
# 3. Memory module — the user-facing primitive.
#    Owns the state machine; delegates addressing to backend.
# ─────────────────────────────────────────────────────────────

class Memory(nn.Module):
    """nn.Memory — Differentiable external memory for agentic AI.

    A first-class PyTorch primitive (like nn.Linear, nn.LSTM) that
    provides differentiable read/write access to an external memory
    bank.  The addressing backend (NTM, DNC, sliding window) is
    swappable.

    Args:
        num_slots:  Number of memory slots.
        slot_size:  Dimension of each slot.
        num_reads:  Number of parallel read heads.
        num_writes: Number of parallel write heads.
        backend:    Addressing backend — "ntm", "dnc", "sliding",
                    or a MemoryBackend instance.

    Shapes:
        Input:  (batch, slot_size) or (batch, seq_len, slot_size)
        Output: (batch, num_reads, slot_size)  —  read vectors

    Example:
        >>> mem = Memory(128, 64, num_reads=4, backend="dnc")
        >>> x = torch.randn(16, 64)
        >>> out, state = mem(x)          # read + write, internal state
        >>> out2, state = mem(x, state)  # explicit state
        >>> state = mem.reset(16)        # fresh state
    """

    def __init__(
        self,
        num_slots: int = 128,
        slot_size: int = 64,
        num_reads: int = 4,
        num_writes: int = 1,
        backend: str = "ntm",
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes

        # Memory matrix — the actual storage.
        self.memory_bias = nn.Parameter(torch.zeros(num_slots, slot_size))

        # Build backend.
        backend_cls = {
            "ntm": NTMBank,
            "dnc": DNCBank,
            "sliding": SlidingWindowBank,
        }.get(backend, None)
        if backend_cls is not None:
            self.backend: MemoryBackend = backend_cls(
                num_slots, slot_size, num_reads, num_writes,
            )
        elif isinstance(backend, MemoryBackend):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # State is managed internally for drop-in ease-of-use,
        # but can be overridden via the `state` argument.
        self._state: Optional[MemoryState] = None

    # ── public API ──────────────────────────────────────────

    def reset(self, batch_size: int = 1) -> MemoryState:
        """Create a fresh initial state.

        All tensors are zero-initialised except read/write weights
        which are uniform (1/num_slots).
        """
        state = MemoryState.zeros(
            self.num_slots, self.slot_size,
            self.num_reads, self.num_writes,
            batch_size,
            device=next(self.parameters()).device,
        )
        with torch.no_grad():
            init = torch.empty(self.num_slots, self.slot_size, device=state.memory.device)
            nn.init.xavier_uniform_(init)
            state.memory.copy_(init)
            state.memory.add_(self.memory_bias)
        self._state = state
        return state

    def read(
        self,
        query: torch.Tensor,
        state: Optional[MemoryState] = None,
    ) -> Tuple[torch.Tensor, MemoryState]:
        """Read from memory without writing.

        Args:
            query: (batch, slot_size) or (batch, seq_len, slot_size)
            state: Current memory state (uses internal state if None).

        Returns:
            (read_vectors, new_state)
        """
        B, L = query.shape[0], 1 if query.dim() == 2 else query.shape[1]
        if query.dim() == 2:
            query = query.unsqueeze(1)

        state = state if state is not None else self._state
        if state is None:
            state = self.reset(B)

        # Backend produces read weights via content/location addressing.
        read_w, _write_params, new_s = self.backend(query[:, -1, :], state)
        vectors = torch.matmul(read_w, new_s.memory.unsqueeze(0))

        self._state = new_s
        return vectors, new_s

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        state: Optional[MemoryState] = None,
    ) -> MemoryState:
        """Write to memory.

        Args:
            key:   (batch, slot_size) — determines where to write.
            value: (batch, slot_size) — content to write.
            state: Current memory state.

        Returns:
            new_state  (memory matrix is updated)
        """
        B = key.shape[0]
        state = state if state is not None else self._state
        if state is None:
            state = self.reset(B)

        # Use a dummy read to get write params from the backend.
        _read_w, write_params, new_s = self.backend(key, state)

        # Parse write params.
        W = self.num_writes
        ws = self.slot_size
        write_vals = write_params[:, :W * ws].reshape(B, W, ws)
        erase = torch.sigmoid(write_params[:, W * ws:].reshape(B, W, ws))

        # Erase.
        erase_mat = (new_s.write_weights.transpose(1, 2) @ erase).mean(dim=0)
        memory = new_s.memory * (1 - erase_mat)

        # Add.
        add_mat = (new_s.write_weights.transpose(1, 2) @ value.unsqueeze(1)).mean(dim=0)
        memory = memory + add_mat

        new_s.memory = memory
        self._state = new_s
        return new_s

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MemoryState] = None,
    ) -> Tuple[torch.Tensor, MemoryState]:
        """One step: attend to memory → write → read.

        Equivalent to: read(write(query)) as in the original NTM.

        Args:
            x:     (batch, slot_size) or (batch, seq_len, slot_size)
            state: Optional initial state.

        Returns:
            (read_vectors, new_state)
        """
        B, L = x.shape[0], 1 if x.dim() == 2 else x.shape[1]
        flat = x.dim() == 2

        state = state if state is not None else self._state
        if state is None:
            state = self.reset(B)

        outputs = []
        for t in range(L):
            step = x[:, t, :] if not flat else x
            rw, wp, state = self.backend(step, state)
            # Write: erase then add.
            W, S = self.num_writes, self.slot_size
            write_vals = wp[:, :W * S].reshape(B, W, S)
            erase = torch.sigmoid(wp[:, W * S:].reshape(B, W, S))
            erase_mat = (state.write_weights.transpose(1, 2) @ erase).mean(dim=0)
            state.memory = state.memory * (1 - erase_mat)
            add_mat = (state.write_weights.transpose(1, 2) @ write_vals).mean(dim=0)
            state.memory = state.memory + add_mat
            # Read.
            vectors = torch.matmul(rw, state.memory.unsqueeze(0))
            outputs.append(vectors)

        result = torch.stack(outputs, dim=1) if L > 1 else outputs[0]
        self._state = state
        return result, state

    @property
    def state(self) -> Optional[MemoryState]:
        return self._state

    @state.setter
    def state(self, value: MemoryState) -> None:
        self._state = value
