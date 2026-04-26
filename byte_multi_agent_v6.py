# %% [markdown]
# # Byte-Multi-Agent v6.0: Continuous Life, Offspring-as-Fitness
#
# ## Design philosophy
#
# v5.x systems treated emergent communication as a supervised-learning-style
# problem with explicit fitness functions, curriculum stages, and speaker/listener
# role assignments. That got us to ~78% eat rate on a trivial Lewis game but not
# any further, because ES on hand-crafted fitness doesn't produce the kind of
# open-ended optimization that evolution uses to produce intelligence.
#
# v6.0 uses a fundamentally different approach:
#
#   1. NO fitness function.
#      Agents have energy. Energy depletes with time and actions. Eating food
#      restores energy. Agents with enough energy can reproduce. Agents at zero
#      energy die. The "fitness function" is just: did you reproduce? — and even
#      that isn't computed anywhere. The population IS the optimizer.
#
#   2. NO curriculum.
#      One world, one set of dynamics, millions of ticks. If the agents can't
#      solve it, we see extinction. If they can, we see dynasties.
#
#   3. NO speaker/listener roles.
#      Every agent can move, eat, reproduce, speak, and listen. Whether
#      communication emerges is determined by selection, not our assignment.
#
#   4. NO ES gradient, NO PPO.
#      Reproduction = clone parent params + Gaussian mutation. The only
#      learning signal is "did my lineage persist?".
#
#   5. NO shared brain.
#      Each agent slot has its own parameter tensor. We have 256 slots per
#      world, so 256 distinct agents evolving in parallel.
#
# ## TRC scaling: island model
#
# Each TPU device runs its own independent world. Worlds are connected in a
# ring: every MIGRATION_INTERVAL ticks, each world exports its top-K
# reproducers to the next world in the ring. This is a standard island-model
# evolutionary technique that prevents local extinction while allowing
# regional divergence (dialects).
#
# ## Measurement
#
# Every MEASURE_INTERVAL ticks, we pause and read out state to CPU to track:
#   - Population size per world
#   - Generation depth of living agents
#   - Speech usage rate
#   - MI between agent speech and food-in-its-vicinity
#   - Lineage diversity
#   - Cross-world dialect divergence
#
# These measurements are for our observation only — they never feed back into
# the sim. The sim's only feedback is energy → reproduction → survival.

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import flax.linen as nn
from typing import NamedTuple
from functools import partial
import time
import numpy as np
import os
import json

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")


# ==========================================
# WORLD CONSTANTS
# ==========================================

# Cell types (visible to agents)
EMPTY = 0
WALL = 1
FOOD = 2          # food cell — agents can SEE food but can't tell good from bad
                  # without being near it
AGENT_MARK = 3
NUM_CELL_TYPES = 4  # for obs normalization

# Food TYPE codes (stored in parallel food_type_grid; NOT visible in vision)
# 0 = no food (cell is not food), 1 = good (energy boost), 2 = bad (energy hit)
FOOD_TYPE_NONE = 0
FOOD_TYPE_GOOD = 1
FOOD_TYPE_BAD = 2

# Of all spawned food, this fraction is good. Bad food looks identical from
# vision but you only learn its type by getting close (sniffing).
GOOD_FOOD_FRACTION = 0.85       # was 0.70 — bad food is a real but uncommon
                                # hazard; random foragers shouldn't die just
                                # from eating typical food

# World geometry
WORLD_SIZE = 64              # 64x64 grid per world
VISION_RADIUS = 4            # agents see 9x9 local region
VISION_SIZE = 2 * VISION_RADIUS + 1
SNIFF_RADIUS = 1             # agents automatically know food types of cells
                             # within this Chebyshev radius (Moore neighborhood
                             # of size 3x3 around the agent). To know about
                             # food further away, you must move closer OR
                             # listen to a speaker who's been there.

# Agent population per world
N_SLOTS = 256                # max simultaneous agents per world
# Initial alive count: we seed ~1/4 of slots so there's room for reproduction
INITIAL_POP = 64

# Speech channel
VOCAB_SIZE = 16              # tokens agents can emit (0 = silence)
SPEECH_RADIUS = 16           # agents hear within this radius (4× vision radius)
                             # so speech has real utility beyond vision

# Hidden state size (small; larger comes in v6.1 if v6.0 works)
HIDDEN_SIZE = 32

# Action spaces
NUM_MOVE_ACTIONS = 5         # stay, N, S, E, W
# Speak action: token index (0 means silence)
# Reproduce action: 0=no, 1=yes. Agent reproduces only if energy>=repro threshold
# AND it chose to reproduce.

# ==========================================
# ENERGY ECONOMICS
# ==========================================
# v6.2: v6.1 was too generous — population capped instantly and stopped
# evolving. Now: random agents survive but rarely reproduce; good agents
# thrive; ALL agents have a hard lifespan limit so the population always
# turns over.

INITIAL_ENERGY = 100.0        # was 150 — back to a real starting buffer
TICK_ENERGY_COST = 0.4        # was 0.2 — restored mortality pressure
MOVE_ENERGY_COST = 0.05       # unchanged
SPEAK_ENERGY_COST = 0.1       # was 0.02 — speech is now a real investment;
                              # silence is the default unless speech provides
                              # offsetting value via better foraging
FOOD_ENERGY_GOOD = 100.0      # red food: nourishing
FOOD_ENERGY_BAD = -20.0       # was -60 — a single bad meal at low energy
                              # was killing random foragers before selection
                              # could operate. -20 still creates a real
                              # selection gradient (smart=+100, dumb avg=+82)
                              # but doesn't outright kill on first encounter.
REPRODUCE_ENERGY_COST = 150.0 # was 100 — bigger investment in reproduction
REPRODUCE_THRESHOLD = 200.0   # was 130 — can only reproduce after sustained success
MAX_ENERGY = 400.0            # unchanged

# Food regeneration: less abundant world; with mortality the population should
# self-regulate well below the slot cap.
TARGET_FOOD_COUNT = 150       # was 200 — moderate abundance
FOOD_REGEN_PER_TICK = 5       # max food spawned per tick. With slow regen,
                              # heavy local foraging creates real scarcity.

# Lifespan: agents die of old age regardless of energy. This is the key
# selection-pressure restoration: even "perfect" agents die, freeing slots
# for descendants. Without this, the first random agent to find food fills
# the population and never lets anyone else evolve past it.
MAX_LIFESPAN = 1500           # ticks. Roughly 4× the time to reach repro threshold.

# ==========================================
# MUTATION
# ==========================================

# When an agent reproduces, child params = parent + noise * sigma.
# sigma ITSELF is a heritable trait — each agent carries its own mutation rate.
# This lets the population tune its own exploration rate.
MUTATION_SIGMA_INIT = 0.02    # starting mutation rate
MUTATION_SIGMA_MIN = 0.001
MUTATION_SIGMA_MAX = 0.1
MUTATION_META_SIGMA = 0.1     # how much mutation_sigma itself mutates
                              # (multiplicatively; log-normal noise)

# ==========================================
# SIM SCHEDULE
# ==========================================

TOTAL_TICKS = 2_000_000        # total sim ticks per run
MEASURE_INTERVAL = 2_000       # measure every N ticks (for JSON log)
PRINT_INTERVAL = 20_000        # PRINT to stdout every N ticks (10x sparser
                               # than measurement to keep output manageable
                               # on TRC pods with 32+ devices)
MIGRATION_INTERVAL = 25_000    # was 100K — much more frequent, since lineage
                               # collapse to 1-2 per world was happening fast
MIGRATION_K = 24               # was 8 — more agents per event. With N_SLOTS=256
                               # and 24 migrating, ~9% of each world's
                               # population turns over per migration.
CHECKPOINT_INTERVAL = 200_000  # was 100K — disk writes are slow, less often is
                               # fine since JSON log preserves trajectory

# ==========================================
# ACTION TEMPERATURE
# ==========================================
ACTION_TEMP = 0.8              # slightly stochastic to prevent determinism


# ==========================================
# WORLD STATE
# ==========================================

class WorldState(NamedTuple):
    # Grid: (WORLD_SIZE, WORLD_SIZE) — EMPTY/WALL/FOOD/AGENT_MARK
    # FOOD cells are visually identical regardless of type; the type is
    # stored in food_type_grid (a parallel array).
    grid: jnp.ndarray
    food_type_grid: jnp.ndarray       # (WORLD_SIZE, WORLD_SIZE) int32
                                       # 0=none, 1=good, 2=bad. Only meaningful
                                       # where grid==FOOD.

    # Per-slot agent data. Dead slots have alive=False but other fields may
    # still hold old values — they are IGNORED for dead slots.
    alive: jnp.ndarray                # (N_SLOTS,) bool
    pos: jnp.ndarray                  # (N_SLOTS, 2) int32
    energy: jnp.ndarray               # (N_SLOTS,) float32
    hidden: jnp.ndarray               # (N_SLOTS, HIDDEN_SIZE) float32

    # Per-slot speech emitted last tick (0 means silence)
    last_speak: jnp.ndarray           # (N_SLOTS,) int32

    # Per-slot network params (flat)
    params_flat: jnp.ndarray          # (N_SLOTS, num_params) float32

    # Per-slot heritable traits
    mutation_sigma: jnp.ndarray       # (N_SLOTS,) float32

    # Lineage tracking
    lineage_id: jnp.ndarray           # (N_SLOTS,) int32 — unique per founder
    generation: jnp.ndarray           # (N_SLOTS,) int32 — depth from founder
    age: jnp.ndarray                  # (N_SLOTS,) int32 — ticks alive
    offspring_count: jnp.ndarray      # (N_SLOTS,) int32 — children produced

    # Per-slot lifetime food stats — used to compute "mean energy per food
    # eaten" as a signal of foraging skill (random=52, perfect=100, worst=-60)
    food_eaten_count: jnp.ndarray     # (N_SLOTS,) int32
    food_energy_total: jnp.ndarray    # (N_SLOTS,) float32 — sum of energy
                                       # gains/losses from eating

    # Global sim state
    tick: jnp.ndarray                 # int32 scalar
    next_lineage_id: jnp.ndarray      # int32 scalar (counter for new founders)
    rng: jnp.ndarray                  # (2,) uint32 for in-sim randomness


MOVE_DIRS = jnp.array([
    [0, 0],     # stay
    [-1, 0],    # N
    [1, 0],     # S
    [0, 1],     # E
    [0, -1],    # W
], dtype=jnp.int32)


# ==========================================
# AGENT NETWORK
# ==========================================

class AgentNet(nn.Module):
    """Compact network. Input: vision grid + speech heard + sniff + proprio.
    Outputs: move logits (5), speak logits (VOCAB_SIZE+1)."""

    @nn.compact
    def __call__(self, obs_grid, obs_speech, obs_sniff, obs_proprio, hidden):
        grid_bf = obs_grid.astype(jnp.bfloat16)
        speech_bf = obs_speech.astype(jnp.bfloat16)
        sniff_bf = obs_sniff.astype(jnp.bfloat16)
        proprio_bf = obs_proprio.astype(jnp.bfloat16)
        hidden_bf = hidden.astype(jnp.bfloat16)

        # Vision: small conv
        x = grid_bf[None, :, :, None]
        x = nn.Conv(features=8, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        # Strided to reduce size
        x = x[::2, ::2]
        x_flat = x.reshape(-1)

        # Speech: small dense
        s = nn.Dense(16, dtype=jnp.bfloat16, param_dtype=jnp.float32)(speech_bf)
        s = nn.relu(s)

        # Sniff: small dense
        sn = nn.Dense(16, dtype=jnp.bfloat16, param_dtype=jnp.float32)(sniff_bf)
        sn = nn.relu(sn)

        # Combine
        feat = jnp.concatenate([x_flat, s, sn, proprio_bf], axis=-1)
        feat = nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(feat)
        feat = nn.relu(feat)

        # GRU-lite update
        zr = nn.Dense(2 * HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(
            jnp.concatenate([feat, hidden_bf], axis=-1))
        z = nn.sigmoid(zr[:HIDDEN_SIZE])
        r = nn.sigmoid(zr[HIDDEN_SIZE:])
        cand = nn.tanh(
            nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(
                jnp.concatenate([feat, r * hidden_bf], axis=-1))
        )
        new_hidden = (1.0 - z) * hidden_bf + z * cand
        h32 = new_hidden.astype(jnp.float32)

        # Heads
        move_logits = nn.Dense(NUM_MOVE_ACTIONS, param_dtype=jnp.float32)(h32)
        speak_logits = nn.Dense(VOCAB_SIZE + 1, param_dtype=jnp.float32)(h32)

        return move_logits, speak_logits, h32


# ==========================================
# PARAM UTILS
# ==========================================

def flatten_params(params):
    leaves = jax.tree.leaves(params)
    return jnp.concatenate([l.reshape(-1) for l in leaves])


def unflatten_params(flat, params_template):
    leaves = jax.tree.leaves(params_template)
    shapes = [l.shape for l in leaves]
    sizes = [l.size for l in leaves]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    new_leaves = [
        lax.dynamic_slice(flat, (offsets[i],), (sizes[i],)).reshape(shapes[i])
        for i in range(len(leaves))
    ]
    return jax.tree.unflatten(jax.tree.structure(params_template), new_leaves)


# ==========================================
# OBSERVATIONS
# ==========================================

# Proprio dims:
PROPRIO_DIM = 1 + 1 + 2  # energy (normalized), age (normalized), pos_norm (2)
# Speech obs: we hear all agents within SPEECH_RADIUS. We summarize by
# (a) counting tokens per type within radius, and (b) one-hot for our own last
# speech. This keeps speech obs size bounded regardless of population.
SPEECH_OBS_DIM = (VOCAB_SIZE + 1) + (VOCAB_SIZE + 1)  # counts + own-last
# Sniff obs: 3x3 grid of food-type info around the agent. Each cell:
#   0 = empty/wall (not food)
#   1 = good food
#   2 = bad food
# Encoded as 3 channels (one-hot per cell), 3x3 spatial = 27 dims.
SNIFF_GRID_SIZE = 2 * SNIFF_RADIUS + 1
SNIFF_OBS_DIM = SNIFF_GRID_SIZE * SNIFF_GRID_SIZE * 3


def compute_shared_obs_context(state):
    """Compute world-level obs context ONCE per tick (shared across agents)."""
    # Draw alive agents on the grid
    def add_agent(g, i):
        ar, ac = state.pos[i, 0], state.pos[i, 1]
        return jnp.where(state.alive[i], g.at[ar, ac].set(AGENT_MARK), g), None

    grid_drawn, _ = lax.scan(add_agent, state.grid, jnp.arange(N_SLOTS))
    padded_grid = jnp.pad(grid_drawn, VISION_RADIUS, constant_values=WALL)

    # Also pad food_type_grid for sniff observations. Pad with FOOD_TYPE_NONE.
    padded_food_type = jnp.pad(state.food_type_grid, SNIFF_RADIUS,
                                 constant_values=FOOD_TYPE_NONE)

    return padded_grid, padded_food_type


def get_agent_obs_shared(state, slot_idx, padded_grid, padded_food_type):
    """Return (obs_grid, obs_speech, obs_sniff, obs_proprio) for one agent."""
    r, c = state.pos[slot_idx, 0], state.pos[slot_idx, 1]

    # Vision is a simple slice from padded_grid
    vision = lax.dynamic_slice(padded_grid, (r, c), (VISION_SIZE, VISION_SIZE))
    obs_grid = vision.astype(jnp.float32) / (NUM_CELL_TYPES - 1.0)

    # Sniff: 3x3 around agent gives food types
    sniff_slice = lax.dynamic_slice(padded_food_type, (r, c),
                                      (SNIFF_GRID_SIZE, SNIFF_GRID_SIZE))
    # One-hot encode: 3 channels (none/good/bad)
    sniff_oh = jax.nn.one_hot(sniff_slice, 3)  # (3, 3, 3)
    obs_sniff = sniff_oh.reshape(-1)

    # Speech
    all_pos = state.pos
    diff = jnp.abs(all_pos - jnp.array([r, c]))
    within = (diff[:, 0] <= SPEECH_RADIUS) & (diff[:, 1] <= SPEECH_RADIUS)
    valid = state.alive & within & (jnp.arange(N_SLOTS) != slot_idx) & \
            (state.last_speak > 0)

    token_oh = jax.nn.one_hot(state.last_speak, VOCAB_SIZE + 1)
    counts = jnp.sum(token_oh * valid.astype(jnp.float32)[:, None], axis=0)
    counts = counts / (jnp.sum(counts) + 1.0)

    own_last_oh = jax.nn.one_hot(state.last_speak[slot_idx], VOCAB_SIZE + 1)
    obs_speech = jnp.concatenate([counts, own_last_oh], axis=-1)

    # Proprio
    energy_norm = state.energy[slot_idx] / MAX_ENERGY
    age_norm = jnp.minimum(state.age[slot_idx].astype(jnp.float32) / 1000.0, 1.0)
    pos_norm = state.pos[slot_idx].astype(jnp.float32) / float(WORLD_SIZE)
    obs_proprio = jnp.concatenate([
        jnp.array([energy_norm, age_norm]), pos_norm])

    return obs_grid, obs_speech, obs_sniff, obs_proprio


# ==========================================
# STEP
# ==========================================

def step_world(state, apply_fn, params_template, step_key):
    """Advance the world one tick."""
    k_act, k_repro, k_food, k_meta, k_food_type, k_next = random.split(step_key, 6)

    # --- Shared obs context (compute once, not per agent) ---
    padded_grid, padded_food_type = compute_shared_obs_context(state)

    # --- 1. All agents compute actions ---
    slot_keys = random.split(k_act, N_SLOTS)

    def act_one(idx):
        og, os_, osn, op = get_agent_obs_shared(
            state, idx, padded_grid, padded_food_type)
        params = unflatten_params(state.params_flat[idx], params_template)
        move_logits, speak_logits, new_hidden = apply_fn(
            params, og, os_, osn, op, state.hidden[idx])
        k_m, k_s = random.split(slot_keys[idx], 2)

        def sample_cat(logits, k, temp):
            g = random.gumbel(k, logits.shape)
            return jnp.argmax(logits / temp + g)

        move = sample_cat(move_logits, k_m, ACTION_TEMP)
        speak = sample_cat(speak_logits, k_s, ACTION_TEMP)
        return move, speak, new_hidden

    moves, speaks, new_hiddens = vmap(act_one)(jnp.arange(N_SLOTS))
    moves = jnp.where(state.alive, moves, 0)
    speaks = jnp.where(state.alive, speaks, 0)

    # --- 2. Movement resolution ---
    curr_pos = state.pos
    dp = MOVE_DIRS[moves]
    want_pos = curr_pos + dp
    want_pos = jnp.clip(want_pos, 0, WORLD_SIZE - 1)
    want_pos = jnp.where(state.alive[:, None], want_pos, curr_pos)

    # Wall check
    wall_mask = (state.grid[want_pos[:, 0], want_pos[:, 1]] == WALL)
    want_pos = jnp.where(wall_mask[:, None], curr_pos, want_pos)

    # Collision: two alive agents want the same cell → neither moves
    same_target = jnp.all(
        want_pos[:, None, :] == want_pos[None, :, :], axis=-1)
    eye = jnp.eye(N_SLOTS, dtype=jnp.bool_)
    alive_mat = state.alive[:, None] & state.alive[None, :]
    has_conflict = jnp.any(same_target & alive_mat & ~eye, axis=1)
    final_pos = jnp.where(has_conflict[:, None], curr_pos, want_pos)

    # Also block moving onto another alive agent's current position
    target_occupied = jnp.all(
        final_pos[:, None, :] == curr_pos[None, :, :], axis=-1)
    target_occupied = target_occupied & state.alive[None, :] & ~eye
    target_occupied_by_other = jnp.any(target_occupied, axis=1)
    final_pos = jnp.where(target_occupied_by_other[:, None], curr_pos, final_pos)

    # --- 3. Food eating (vectorized, type-aware) ---
    on_food = state.alive & (state.grid[final_pos[:, 0], final_pos[:, 1]] == FOOD)
    # Get the food type at each agent's final position (for those on food)
    food_type_at_agent = state.food_type_grid[final_pos[:, 0], final_pos[:, 1]]
    is_good = on_food & (food_type_at_agent == FOOD_TYPE_GOOD)
    is_bad = on_food & (food_type_at_agent == FOOD_TYPE_BAD)
    # Energy delta from eating (good = +100, bad = -60, none = 0)
    food_delta = (is_good.astype(jnp.float32) * FOOD_ENERGY_GOOD +
                  is_bad.astype(jnp.float32) * FOOD_ENERGY_BAD)

    # Remove eaten food from grid AND food_type_grid
    eaten_flat_idx = final_pos[:, 0] * WORLD_SIZE + final_pos[:, 1]
    eaten_flat_idx = jnp.where(on_food, eaten_flat_idx, WORLD_SIZE * WORLD_SIZE)
    grid_flat = state.grid.reshape(-1)
    type_flat = state.food_type_grid.reshape(-1)
    padded_grid_flat = jnp.concatenate([grid_flat, jnp.array([EMPTY])])
    padded_type_flat = jnp.concatenate([type_flat, jnp.array([FOOD_TYPE_NONE])])
    grid_after_eat_flat = padded_grid_flat.at[eaten_flat_idx].set(EMPTY)[:-1]
    type_after_eat_flat = padded_type_flat.at[eaten_flat_idx].set(FOOD_TYPE_NONE)[:-1]
    grid_after_eat = grid_after_eat_flat.reshape(WORLD_SIZE, WORLD_SIZE)
    food_type_after_eat = type_after_eat_flat.reshape(WORLD_SIZE, WORLD_SIZE)

    # Update per-slot food stats (counts events, sums energy delta)
    new_food_eaten_count = state.food_eaten_count + on_food.astype(jnp.int32)
    new_food_energy_total = state.food_energy_total + food_delta

    # --- 4. Energy update ---
    move_happened = jnp.any(dp != 0, axis=-1) & state.alive
    spoke = (speaks > 0) & state.alive

    energy = state.energy
    energy = energy - jnp.where(state.alive, TICK_ENERGY_COST, 0.0)
    energy = energy - jnp.where(move_happened, MOVE_ENERGY_COST, 0.0)
    energy = energy - jnp.where(spoke, SPEAK_ENERGY_COST, 0.0)
    energy = energy + food_delta  # +100 for good, -60 for bad, 0 otherwise
    energy = jnp.minimum(energy, MAX_ENERGY)

    # --- 5. Reproduction (AUTOMATIC — no learned action required) ---
    wants_repro = state.alive & (energy >= REPRODUCE_THRESHOLD)
    empty_slot = ~state.alive

    # Rank parents by random priority (so reproduction order isn't slot-based)
    parent_priority_key = random.fold_in(k_repro, jnp.int32(0))
    parent_priority = random.uniform(parent_priority_key, (N_SLOTS,))
    parent_priority = jnp.where(wants_repro, parent_priority, -1.0)
    parent_order = jnp.argsort(-parent_priority)  # (N_SLOTS,)

    # Empty slots by index
    empty_order_key = jnp.where(empty_slot, jnp.arange(N_SLOTS),
                                  N_SLOTS + 1)
    empty_order = jnp.argsort(empty_order_key)

    # How many valid births? The k-th pair is valid iff:
    #   parent_order[k] is actually a wants_repro slot
    #   AND empty_order[k] is actually an empty slot
    # We compute per-pair validity.
    parent_k_wants = wants_repro[parent_order]       # (N_SLOTS,)
    empty_k_isempty = empty_slot[empty_order]        # (N_SLOTS,)
    pair_valid = parent_k_wants & empty_k_isempty    # (N_SLOTS,)

    # For vectorized application, we want:
    #   For each child slot c, if there's a pair that produces this child, find
    #   the pair's parent. Then child receives mutated copy of parent's stuff.
    #
    # Equivalently: for each PAIR k, the parent is parent_order[k] and child
    # is empty_order[k]. We produce child arrays indexed by pair k, then
    # scatter into child slots. Unused pair slots get masked.

    # Gather per-pair parent info
    parent_indices = parent_order  # (N_SLOTS,) — parent slot at each pair
    child_indices = empty_order    # (N_SLOTS,) — child slot at each pair

    parent_params_per_pair = state.params_flat[parent_indices]  # (N_SLOTS, num_params)
    parent_mut_per_pair = state.mutation_sigma[parent_indices]  # (N_SLOTS,)
    parent_lineage_per_pair = state.lineage_id[parent_indices]
    parent_gen_per_pair = state.generation[parent_indices]
    parent_pos_per_pair = final_pos[parent_indices]             # (N_SLOTS, 2)

    # Per-pair mutation noise
    mut_keys = random.split(k_meta, N_SLOTS * 2).reshape(N_SLOTS, 2, 2)
    def mutate(k_pair, parent_p, parent_mut):
        param_noise = random.normal(k_pair[0], parent_p.shape) * parent_mut
        meta_noise = random.normal(k_pair[1], ()) * MUTATION_META_SIGMA
        child_p = parent_p + param_noise
        child_mut = jnp.clip(parent_mut * jnp.exp(meta_noise),
                               MUTATION_SIGMA_MIN, MUTATION_SIGMA_MAX)
        return child_p, child_mut

    child_params_per_pair, child_mut_per_pair = vmap(mutate)(
        mut_keys, parent_params_per_pair, parent_mut_per_pair)

    # Now scatter per-pair child info into their child slots.
    # We only scatter for valid pairs; invalid pairs write to "scratch" slots
    # that don't get used. Trick: use pair_valid to route invalid pairs to
    # their OWN slot (which is an empty slot with nothing important in it),
    # and then just mask the final alive bit.
    # Actually simpler: we do a scatter for all N_SLOTS pairs, but we only
    # SET alive=True where pair_valid is True. Other slots' alive stays as-is
    # (which is False, since child_indices points at empty slots).

    # Build new per-slot arrays
    new_alive = state.alive
    new_pos = final_pos
    new_energy = energy
    new_hidden_arr = jnp.where(state.alive[:, None], new_hiddens,
                                 state.hidden)
    new_params = state.params_flat
    new_mut_sigma = state.mutation_sigma
    new_lineage = state.lineage_id
    new_generation = state.generation
    new_age = state.age + state.alive.astype(jnp.int32)
    new_offspring = state.offspring_count
    # Food stats already updated during eating step above; carry forward
    # the post-eating values, then reset for children below.
    # (new_food_eaten_count and new_food_energy_total were computed above)

    # Parent energy deductions: for each parent_index of a valid pair, deduct
    # REPRODUCE_ENERGY_COST.
    # Use segment_sum: per parent slot, count number of valid pairs with that
    # parent (should be 0 or 1 since each parent appears at most once in
    # parent_order).
    energy_deduct = jnp.where(pair_valid, REPRODUCE_ENERGY_COST, 0.0)
    parent_deductions = jnp.zeros(N_SLOTS).at[parent_indices].add(energy_deduct)
    new_energy = new_energy - parent_deductions

    # Offspring count bumps for parents
    off_bump = pair_valid.astype(jnp.int32)
    parent_bumps = jnp.zeros(N_SLOTS, dtype=jnp.int32).at[parent_indices].add(off_bump)
    new_offspring = new_offspring + parent_bumps

    # Scatter child state into child slots (only where pair_valid)
    # For each pair k: if pair_valid[k], then:
    #   new_alive[child_indices[k]] = True
    #   new_pos[child_indices[k]] = parent_pos_per_pair[k]
    #   new_energy[child_indices[k]] = REPRODUCE_ENERGY_COST * 0.5
    #   new_hidden_arr[child_indices[k]] = zeros
    #   new_params[child_indices[k]] = child_params_per_pair[k]
    #   ...
    #
    # Trick: we do these scatters unconditionally but for invalid pairs, the
    # destination (child_indices[k]) is ALSO an "empty" slot, and we'd like
    # it to REMAIN empty. We accomplish this by using masked values:
    # for invalid pairs, we write alive=False (current state), and other
    # fields don't matter (we overwrite with current state values).
    #
    # For VALID pairs: we write alive=True, child params, etc.

    child_new_alive = pair_valid  # True for valid, False for invalid
    child_new_pos = parent_pos_per_pair
    child_new_energy = jnp.where(pair_valid, REPRODUCE_ENERGY_COST * 0.5, 0.0)
    child_new_hidden = jnp.zeros((N_SLOTS, HIDDEN_SIZE), dtype=jnp.float32)
    child_new_params = child_params_per_pair
    child_new_mut = jnp.where(pair_valid, child_mut_per_pair, MUTATION_SIGMA_INIT)
    child_new_lineage = jnp.where(pair_valid, parent_lineage_per_pair, -1)
    child_new_generation = jnp.where(pair_valid, parent_gen_per_pair + 1, 0)

    # Scatter into child_indices. For INVALID pairs, child_indices[k] points
    # to an empty slot (still !alive). We preserve "not alive" by OR with
    # current alive — actually no, we just overwrite. For invalid pairs,
    # we write False, and the slot stays False.
    # Before scatter, save current state at child_indices so we can preserve
    # non-written fields for invalid pairs.
    # Actually simpler approach: ONLY scatter where pair_valid. The masks
    # above already ensure invalid pairs write "no-op-equivalent" values.

    # Alive: for valid pairs write True; for invalid pairs, child_indices[k]
    # is !alive anyway so writing False is fine.
    new_alive = new_alive.at[child_indices].set(
        jnp.where(pair_valid, True, new_alive[child_indices]))
    new_pos = new_pos.at[child_indices].set(
        jnp.where(pair_valid[:, None], child_new_pos, new_pos[child_indices]))
    new_energy = new_energy.at[child_indices].set(
        jnp.where(pair_valid, child_new_energy, new_energy[child_indices]))
    new_hidden_arr = new_hidden_arr.at[child_indices].set(
        jnp.where(pair_valid[:, None], child_new_hidden, new_hidden_arr[child_indices]))
    new_params = new_params.at[child_indices].set(
        jnp.where(pair_valid[:, None], child_new_params, new_params[child_indices]))
    new_mut_sigma = new_mut_sigma.at[child_indices].set(
        jnp.where(pair_valid, child_new_mut, new_mut_sigma[child_indices]))
    new_lineage = new_lineage.at[child_indices].set(
        jnp.where(pair_valid, child_new_lineage, new_lineage[child_indices]))
    new_generation = new_generation.at[child_indices].set(
        jnp.where(pair_valid, child_new_generation, new_generation[child_indices]))
    # Child age starts at 0
    new_age = new_age.at[child_indices].set(
        jnp.where(pair_valid, jnp.int32(0), new_age[child_indices]))
    # Child offspring starts at 0
    new_offspring = new_offspring.at[child_indices].set(
        jnp.where(pair_valid, jnp.int32(0), new_offspring[child_indices]))
    # Child food stats start at 0
    new_food_eaten_count = new_food_eaten_count.at[child_indices].set(
        jnp.where(pair_valid, jnp.int32(0), new_food_eaten_count[child_indices]))
    new_food_energy_total = new_food_energy_total.at[child_indices].set(
        jnp.where(pair_valid, jnp.float32(0.0),
                    new_food_energy_total[child_indices]))

    # --- 6. Death ---
    # Two ways to die: starvation (energy <= 0) OR old age (age >= MAX_LIFESPAN).
    # Old-age death is critical: without it, the first agents to find food
    # fill the slots and never die, blocking selection.
    starved = new_alive & (new_energy <= 0.0)
    old_age = new_alive & (new_age >= MAX_LIFESPAN)
    died = starved | old_age
    new_alive = new_alive & ~died

    # --- 7. Food regeneration (SLOW, with type assignment) ---
    current_food_count = jnp.sum((grid_after_eat == FOOD).astype(jnp.int32))
    food_deficit = jnp.maximum(TARGET_FOOD_COUNT - current_food_count, 0)
    spawn_count = jnp.minimum(food_deficit, FOOD_REGEN_PER_TICK)

    grid_flat = grid_after_eat.reshape(-1)
    type_flat = food_type_after_eat.reshape(-1)
    n_cells = WORLD_SIZE * WORLD_SIZE
    spawn_noise = random.uniform(k_food, (n_cells,))
    empty_mask = (grid_flat == EMPTY)

    agent_pos_flat = new_pos[:, 0] * WORLD_SIZE + new_pos[:, 1]
    agent_occ = jnp.zeros(n_cells, dtype=jnp.bool_).at[agent_pos_flat].max(
        new_alive)

    spawn_eligible = empty_mask & ~agent_occ
    spawn_score = jnp.where(spawn_eligible, spawn_noise, -1.0)
    top_idx = jnp.argsort(-spawn_score)
    rank = jnp.argsort(top_idx)
    spawn_mask = rank < spawn_count

    # Assign types: each newly-spawned food is GOOD with probability
    # GOOD_FOOD_FRACTION, else BAD.
    type_noise = random.uniform(k_food_type, (n_cells,))
    new_type = jnp.where(type_noise < GOOD_FOOD_FRACTION,
                          FOOD_TYPE_GOOD, FOOD_TYPE_BAD).astype(jnp.int32)

    grid_final = jnp.where(spawn_mask, FOOD, grid_flat).reshape(
        WORLD_SIZE, WORLD_SIZE)
    food_type_final = jnp.where(spawn_mask, new_type, type_flat).reshape(
        WORLD_SIZE, WORLD_SIZE)

    # --- 8. Update state ---
    return WorldState(
        grid=grid_final,
        food_type_grid=food_type_final,
        alive=new_alive,
        pos=new_pos,
        energy=new_energy,
        hidden=new_hidden_arr,
        last_speak=speaks,
        params_flat=new_params,
        mutation_sigma=new_mut_sigma,
        lineage_id=new_lineage,
        generation=new_generation,
        age=new_age,
        offspring_count=new_offspring,
        food_eaten_count=new_food_eaten_count,
        food_energy_total=new_food_energy_total,
        tick=state.tick + 1,
        next_lineage_id=state.next_lineage_id,
        rng=k_next,
    )


# ==========================================
# WORLD INIT
# ==========================================

def init_world(key, params_template, num_params, world_id):
    """Create a fresh world with INITIAL_POP seeded agents."""
    k_grid, k_walls, k_food, k_food_type, k_agents, k_params, key = random.split(key, 7)

    # Grid: border walls, interior empty
    rows = jnp.arange(WORLD_SIZE)[:, None]
    cols = jnp.arange(WORLD_SIZE)[None, :]
    on_border = (rows == 0) | (rows == WORLD_SIZE - 1) | \
                (cols == 0) | (cols == WORLD_SIZE - 1)
    grid = jnp.where(on_border, WALL, EMPTY)

    # Initial food: place TARGET_FOOD_COUNT at random empty cells
    food_noise = random.uniform(k_food, (WORLD_SIZE * WORLD_SIZE,))
    grid_flat = grid.reshape(-1)
    food_score = jnp.where(grid_flat == EMPTY, food_noise, -1.0)
    top_food = jnp.argsort(-food_score)
    food_rank = jnp.argsort(top_food)
    food_mask = food_rank < TARGET_FOOD_COUNT
    grid_flat = jnp.where(food_mask, FOOD, grid_flat)
    grid = grid_flat.reshape(WORLD_SIZE, WORLD_SIZE)

    # Assign types to initial food
    type_noise = random.uniform(k_food_type, (WORLD_SIZE * WORLD_SIZE,))
    food_type_flat = jnp.where(food_mask,
                                 jnp.where(type_noise < GOOD_FOOD_FRACTION,
                                            FOOD_TYPE_GOOD, FOOD_TYPE_BAD),
                                 FOOD_TYPE_NONE).astype(jnp.int32)
    food_type_grid = food_type_flat.reshape(WORLD_SIZE, WORLD_SIZE)

    # Place initial agents at random empty cells
    agent_noise = random.uniform(k_agents, (WORLD_SIZE * WORLD_SIZE,))
    free_score = jnp.where(grid_flat == EMPTY, agent_noise, -1.0)
    top_agents = jnp.argsort(-free_score)[:INITIAL_POP]
    agent_rows = top_agents // WORLD_SIZE
    agent_cols = top_agents % WORLD_SIZE

    # Build per-slot arrays
    alive = jnp.arange(N_SLOTS) < INITIAL_POP
    pos = jnp.zeros((N_SLOTS, 2), dtype=jnp.int32)
    pos = pos.at[:INITIAL_POP, 0].set(agent_rows)
    pos = pos.at[:INITIAL_POP, 1].set(agent_cols)
    energy = jnp.where(alive, INITIAL_ENERGY, 0.0)
    hidden = jnp.zeros((N_SLOTS, HIDDEN_SIZE), dtype=jnp.float32)
    last_speak = jnp.zeros((N_SLOTS,), dtype=jnp.int32)

    # Initial params: each agent gets an INDEPENDENT random init
    param_keys = random.split(k_params, N_SLOTS)

    def gen_params(k):
        dummy_grid = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
        dummy_speech = jnp.zeros((SPEECH_OBS_DIM,), dtype=jnp.float32)
        dummy_sniff = jnp.zeros((SNIFF_OBS_DIM,), dtype=jnp.float32)
        dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
        dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
        p = AgentNet().init(k, dummy_grid, dummy_speech, dummy_sniff,
                              dummy_proprio, dummy_hidden)
        return flatten_params(p)

    params_flat = vmap(gen_params)(param_keys)
    params_flat = jnp.where(alive[:, None], params_flat,
                             jnp.zeros_like(params_flat))

    mutation_sigma = jnp.full((N_SLOTS,), MUTATION_SIGMA_INIT, dtype=jnp.float32)

    lineage_base = jnp.int32(world_id * N_SLOTS)
    lineage_id = lineage_base + jnp.arange(N_SLOTS, dtype=jnp.int32)
    lineage_id = jnp.where(alive, lineage_id, -1)

    generation = jnp.zeros((N_SLOTS,), dtype=jnp.int32)
    age = jnp.zeros((N_SLOTS,), dtype=jnp.int32)
    offspring_count = jnp.zeros((N_SLOTS,), dtype=jnp.int32)
    food_eaten_count = jnp.zeros((N_SLOTS,), dtype=jnp.int32)
    food_energy_total = jnp.zeros((N_SLOTS,), dtype=jnp.float32)

    next_lineage_id = lineage_base + jnp.int32(N_SLOTS)

    return WorldState(
        grid=grid,
        food_type_grid=food_type_grid,
        alive=alive,
        pos=pos,
        energy=energy,
        hidden=hidden,
        last_speak=last_speak,
        params_flat=params_flat,
        mutation_sigma=mutation_sigma,
        lineage_id=lineage_id,
        generation=generation,
        age=age,
        offspring_count=offspring_count,
        food_eaten_count=food_eaten_count,
        food_energy_total=food_energy_total,
        tick=jnp.int32(0),
        next_lineage_id=next_lineage_id,
        rng=key,
    )


# ==========================================
# ISLAND MIGRATION
# ==========================================

def migrate(worlds_state, k_mig):
    """Each world sends its top-K alive agents (by offspring_count) to the next
    world in the ring. Receiving world replaces its bottom-K alive agents with
    the incoming ones.
    """
    num_worlds = worlds_state.alive.shape[0]

    # Per-world: top-K indices by offspring_count (alive-only)
    def top_k_per_world(off, alive):
        score = jnp.where(alive, off.astype(jnp.float32), -1.0)
        return jnp.argsort(-score)[:MIGRATION_K]

    top_per_world = vmap(top_k_per_world)(
        worlds_state.offspring_count, worlds_state.alive)  # (num_worlds, K)

    # Per-world: bottom-K indices — dead slots first, then low-offspring alive
    def bottom_k_per_world(off, alive):
        # Rank: dead slots have score -2 (replaced first),
        # alive slots have score = offspring_count (lower = worse)
        score = jnp.where(alive, off.astype(jnp.float32), -2.0)
        return jnp.argsort(score)[:MIGRATION_K]

    bottom_per_world = vmap(bottom_k_per_world)(
        worlds_state.offspring_count, worlds_state.alive)

    # Helper: gather emigrants from each world using top indices
    def gather(field):
        # field: (num_worlds, N_SLOTS, ...)
        # Want: (num_worlds, MIGRATION_K, ...)
        return vmap(lambda f, idx: f[idx])(field, top_per_world)

    emig_alive = gather(worlds_state.alive)
    emig_pos = gather(worlds_state.pos)
    emig_energy = gather(worlds_state.energy)
    emig_hidden = gather(worlds_state.hidden)
    emig_last_speak = gather(worlds_state.last_speak)
    emig_params = gather(worlds_state.params_flat)
    emig_mut = gather(worlds_state.mutation_sigma)
    emig_lineage = gather(worlds_state.lineage_id)
    emig_gen = gather(worlds_state.generation)
    emig_age = gather(worlds_state.age)
    emig_off = gather(worlds_state.offspring_count)
    emig_food_count = gather(worlds_state.food_eaten_count)
    emig_food_energy = gather(worlds_state.food_energy_total)

    # Rotate: world i sends emigrants to world (i+1) mod num_worlds
    def rot(x):
        return jnp.roll(x, shift=1, axis=0)
    shifted_alive = rot(emig_alive)
    shifted_pos = rot(emig_pos)
    shifted_energy = rot(emig_energy)
    shifted_hidden = rot(emig_hidden)
    shifted_last_speak = rot(emig_last_speak)
    shifted_params = rot(emig_params)
    shifted_mut = rot(emig_mut)
    shifted_lineage = rot(emig_lineage)
    shifted_gen = rot(emig_gen)
    shifted_age = rot(emig_age)
    shifted_off = rot(emig_off)
    shifted_food_count = rot(emig_food_count)
    shifted_food_energy = rot(emig_food_energy)

    # Scatter: for each world, write the incoming emigrants into the bottom indices.
    def scatter_field(field, bottom_idx, incoming):
        return field.at[bottom_idx].set(incoming)

    new_alive = vmap(scatter_field)(worlds_state.alive, bottom_per_world, shifted_alive)
    new_pos = vmap(scatter_field)(worlds_state.pos, bottom_per_world, shifted_pos)
    new_energy = vmap(scatter_field)(worlds_state.energy, bottom_per_world, shifted_energy)
    new_hidden = vmap(scatter_field)(worlds_state.hidden, bottom_per_world, shifted_hidden)
    new_last_speak = vmap(scatter_field)(worlds_state.last_speak, bottom_per_world, shifted_last_speak)
    new_params = vmap(scatter_field)(worlds_state.params_flat, bottom_per_world, shifted_params)
    new_mut = vmap(scatter_field)(worlds_state.mutation_sigma, bottom_per_world, shifted_mut)
    new_lineage = vmap(scatter_field)(worlds_state.lineage_id, bottom_per_world, shifted_lineage)
    new_gen = vmap(scatter_field)(worlds_state.generation, bottom_per_world, shifted_gen)
    new_age = vmap(scatter_field)(worlds_state.age, bottom_per_world, shifted_age)
    new_off = vmap(scatter_field)(worlds_state.offspring_count, bottom_per_world, shifted_off)
    new_food_count = vmap(scatter_field)(worlds_state.food_eaten_count, bottom_per_world, shifted_food_count)
    new_food_energy = vmap(scatter_field)(worlds_state.food_energy_total, bottom_per_world, shifted_food_energy)

    return worlds_state._replace(
        alive=new_alive, pos=new_pos, energy=new_energy, hidden=new_hidden,
        last_speak=new_last_speak, params_flat=new_params,
        mutation_sigma=new_mut, lineage_id=new_lineage,
        generation=new_gen, age=new_age, offspring_count=new_off,
        food_eaten_count=new_food_count, food_energy_total=new_food_energy)


# ==========================================
# MEASUREMENT (CPU-side)
# ==========================================

def measure(worlds_state, tick):
    """Read out state to CPU and compute summary metrics."""
    alive = np.array(jax.device_get(worlds_state.alive))
    energy = np.array(jax.device_get(worlds_state.energy))
    lineage = np.array(jax.device_get(worlds_state.lineage_id))
    generation = np.array(jax.device_get(worlds_state.generation))
    age = np.array(jax.device_get(worlds_state.age))
    offspring = np.array(jax.device_get(worlds_state.offspring_count))
    mut_sigma = np.array(jax.device_get(worlds_state.mutation_sigma))
    last_speak = np.array(jax.device_get(worlds_state.last_speak))
    food_eaten = np.array(jax.device_get(worlds_state.food_eaten_count))
    food_energy = np.array(jax.device_get(worlds_state.food_energy_total))

    num_worlds = alive.shape[0]
    report = {"tick": int(tick), "per_world": [], "aggregate": {}}

    for w in range(num_worlds):
        w_alive = alive[w]
        w_count = int(w_alive.sum())
        if w_count == 0:
            report["per_world"].append({
                "world": w, "pop": 0, "status": "extinct"})
            continue
        w_energy = energy[w][w_alive]
        w_lineage = lineage[w][w_alive]
        w_gen = generation[w][w_alive]
        w_age = age[w][w_alive]
        w_off = offspring[w][w_alive]
        w_mut = mut_sigma[w][w_alive]
        w_speak = last_speak[w][w_alive]
        w_food_eaten = food_eaten[w][w_alive]
        w_food_energy = food_energy[w][w_alive]
        # Energy per food eaten: agents that ate at least 1 food
        ate_at_least_one = w_food_eaten > 0
        if ate_at_least_one.any():
            energy_per_food_arr = (w_food_energy[ate_at_least_one] /
                                    w_food_eaten[ate_at_least_one].astype(float))
            mean_energy_per_food = float(energy_per_food_arr.mean())
        else:
            mean_energy_per_food = 0.0

        report["per_world"].append({
            "world": w,
            "pop": w_count,
            "energy_mean": float(w_energy.mean()),
            "energy_std": float(w_energy.std()),
            "unique_lineages": int(len(np.unique(w_lineage))),
            "max_generation": int(w_gen.max()),
            "mean_generation": float(w_gen.mean()),
            "oldest_age": int(w_age.max()),
            "max_offspring": int(w_off.max()),
            "mean_offspring": float(w_off.mean()),
            "mean_mutation_sigma": float(w_mut.mean()),
            "speech_rate": float((w_speak > 0).mean()),
            "mean_energy_per_food": mean_energy_per_food,
        })

    # Aggregate
    total_pop = int(alive.sum())
    living_energy = energy[alive]
    living_gen = generation[alive]
    living_off = offspring[alive]
    living_mut = mut_sigma[alive]
    living_speak = last_speak[alive]
    living_lineage = lineage[alive]
    living_food_eaten = food_eaten[alive]
    living_food_energy = food_energy[alive]

    # Energy-per-food across all alive agents that have eaten
    has_eaten = living_food_eaten > 0
    if total_pop > 0 and has_eaten.any():
        epf = (living_food_energy[has_eaten] /
               living_food_eaten[has_eaten].astype(float))
        mean_energy_per_food_global = float(epf.mean())
    else:
        mean_energy_per_food_global = 0.0

    report["aggregate"] = {
        "total_pop": total_pop,
        "worlds_alive": int(sum(1 for w in report["per_world"]
                                  if w.get("pop", 0) > 0)),
        "mean_energy": float(living_energy.mean()) if total_pop > 0 else 0.0,
        "max_generation_global": int(living_gen.max()) if total_pop > 0 else 0,
        "mean_generation_global": float(living_gen.mean()) if total_pop > 0 else 0.0,
        "total_unique_lineages": int(len(np.unique(living_lineage))) if total_pop > 0 else 0,
        "mean_offspring_global": float(living_off.mean()) if total_pop > 0 else 0.0,
        "mean_mutation_sigma_global": float(living_mut.mean()) if total_pop > 0 else 0.0,
        "speech_rate_global": float((living_speak > 0).mean()) if total_pop > 0 else 0.0,
        # The KEY metric: energy gained per food eaten.
        # Random forager (no discrimination): ~82 (0.85*100 + 0.15*-20)
        # Perfect discriminator (only good food): 100
        # Worst case: -20 (eats only bad food)
        "mean_energy_per_food_global": mean_energy_per_food_global,
    }

    return report


def print_report(report):
    """One-line aggregate summary. Per-world detail only on anomalies."""
    agg = report["aggregate"]
    n_worlds = len(report["per_world"])

    # One-line summary
    # epf = energy per food: random forager ~82, perfect discriminator 100,
    # worst case -20. Climbing epf above 82 is the key emergence signal.
    print(f"[t={report['tick']:>9,d}] "
          f"pop={agg['total_pop']:>4d}/{N_SLOTS * n_worlds:<5d} "
          f"worlds={agg['worlds_alive']:>2d}/{n_worlds:<2d} "
          f"gen={agg['max_generation_global']:>4d} "
          f"lin={agg['total_unique_lineages']:>4d} "
          f"off={agg['mean_offspring_global']:.2f} "
          f"speech={agg['speech_rate_global']:.2f} "
          f"epf={agg.get('mean_energy_per_food_global', 0.0):>6.1f} "
          f"sigma={agg['mean_mutation_sigma_global']:.4f}")

    # Anomaly detail: any extinct worlds?
    extinct = [w["world"] for w in report["per_world"]
               if w.get("pop", 0) == 0]
    if extinct:
        print(f"   EXTINCT WORLDS: {extinct}")


# ==========================================
# TOP-LEVEL STEP (across all worlds)
# ==========================================

def make_multi_world_step(apply_fn, params_template):
    """Return a jit-compiled function that advances all worlds by one tick."""

    def step_one(state, k):
        return step_world(state, apply_fn, params_template, k), None

    @jit
    def stepper(worlds_state, k):
        # Split one key per world
        num_worlds = worlds_state.alive.shape[0]
        keys = random.split(k, num_worlds)

        def advance_world(ws, wk):
            return step_world(ws, apply_fn, params_template, wk)

        new_state = vmap(advance_world)(worlds_state, keys)
        return new_state

    return stepper


def make_batched_stepper(apply_fn, params_template, n_ticks_per_call):
    """Return a jit-compiled function that advances all worlds by N ticks.

    Batching ticks inside a lax.scan reduces Python overhead substantially.
    """

    def step_one(carry, _):
        state, k = carry
        k_step, k_next = random.split(k)
        num_worlds = state.alive.shape[0]
        keys = random.split(k_step, num_worlds)
        new_state = vmap(lambda ws, wk: step_world(ws, apply_fn, params_template, wk))(
            state, keys)
        return (new_state, k_next), None

    @jit
    def batched(worlds_state, k):
        (final_state, _), _ = lax.scan(step_one, (worlds_state, k), None,
                                         length=n_ticks_per_call)
        return final_state

    return batched


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 70)
    print("  BYTE-MULTI-AGENT v6.5: SOFTENED FOOD HAZARDS")
    print("  (red food +100, blue food -20, 85% good)")
    print("  (random forager survives but smart forager wins)")
    print("  (key metric: epf — random=82, perfect=100)")
    print("=" * 70)

    num_devices = jax.device_count()
    num_worlds = num_devices  # one world per device
    print(f"Running {num_worlds} worlds on {num_devices} devices "
          f"({WORLD_SIZE}x{WORLD_SIZE}, {N_SLOTS} slots/world, "
          f"init_pop={INITIAL_POP})")
    print(f"Schedule: {TOTAL_TICKS:,} ticks, "
          f"measure={MEASURE_INTERVAL:,}, print={PRINT_INTERVAL:,}, "
          f"migrate={MIGRATION_INTERVAL:,} (K={MIGRATION_K}), "
          f"ckpt={CHECKPOINT_INTERVAL:,}")

    # --- Determine checkpoint dir ---
    for cand in ['/kaggle/working', '/tmp', '.']:
        if os.path.exists(cand):
            CKPT_DIR = cand
            break

    # --- Build template for params ---
    key = random.PRNGKey(42)
    k_init, key = random.split(key)
    net = AgentNet()
    dummy_grid = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_speech = jnp.zeros((SPEECH_OBS_DIM,), dtype=jnp.float32)
    dummy_sniff = jnp.zeros((SNIFF_OBS_DIM,), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
    dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
    params_template = net.init(k_init, dummy_grid, dummy_speech, dummy_sniff,
                                 dummy_proprio, dummy_hidden)
    num_params = flatten_params(params_template).shape[0]
    print(f"Per-agent params: {num_params:,} | "
          f"Total: {num_params * N_SLOTS * num_worlds:,} | "
          f"Ckpt dir: {CKPT_DIR}")

    apply_fn = net.apply

    # --- Initialize all worlds ---
    t0 = time.time()
    world_keys = random.split(key, num_worlds + 1)
    key = world_keys[0]
    init_keys = world_keys[1:]

    # vmap init across worlds
    def init_one(k, wid):
        return init_world(k, params_template, num_params, wid)

    worlds_state = vmap(init_one)(init_keys, jnp.arange(num_worlds))
    # Force computation
    _ = jax.block_until_ready(worlds_state.alive)
    print(f"Initialized {num_worlds} worlds in {time.time() - t0:.1f}s")

    # --- Build the stepper ---
    # Batched stepper: advance N ticks per JIT call to reduce Python overhead.
    # We still want to pause for measurements, so N should divide MEASURE_INTERVAL.
    TICKS_PER_CALL = 500
    assert MEASURE_INTERVAL % TICKS_PER_CALL == 0
    batched_stepper = make_batched_stepper(apply_fn, params_template, TICKS_PER_CALL)

    # --- Initial measurement ---
    report = measure(worlds_state, 0)
    print_report(report)

    reports_history = [report]

    # --- Main loop ---
    total_ticks = 0
    t_start = time.time()
    last_checkpoint_tick = 0
    last_migration_tick = 0
    last_print_tick = 0
    pending_events = []  # Queue of strings to print with next status line

    while total_ticks < TOTAL_TICKS:
        calls_to_next_measure = MEASURE_INTERVAL // TICKS_PER_CALL

        for _ in range(calls_to_next_measure):
            k_tick, key = random.split(key)
            worlds_state = batched_stepper(worlds_state, k_tick)
            total_ticks += TICKS_PER_CALL

        # Migration check (silent — accumulated and printed at next status line)
        if total_ticks - last_migration_tick >= MIGRATION_INTERVAL:
            k_mig, key = random.split(key)
            worlds_state = migrate(worlds_state, k_mig)
            last_migration_tick = total_ticks
            pending_events.append(f"migrate@{total_ticks:,}")

        # Measure (always, but cheap — populates JSON log only)
        _ = jax.block_until_ready(worlds_state.alive)
        report = measure(worlds_state, total_ticks)
        reports_history.append(report)

        # Checkpoint (silent unless it's a print tick)
        if total_ticks - last_checkpoint_tick >= CHECKPOINT_INTERVAL:
            ckpt_path = os.path.join(
                CKPT_DIR, f'v6_5_state_tick_{total_ticks}.npz')
            try:
                state_np = jax.tree.map(
                    lambda x: np.array(jax.device_get(x)), worlds_state)
                np.savez(ckpt_path, **{
                    k: v for k, v in state_np._asdict().items()})
                pending_events.append(f"ckpt@{total_ticks:,}")
            except Exception as e:
                pending_events.append(f"ckpt_FAIL: {e}")

            log_path = os.path.join(CKPT_DIR, 'v6_5_reports.json')
            try:
                with open(log_path, 'w') as f:
                    json.dump(reports_history, f, indent=1)
            except Exception as e:
                pending_events.append(f"log_FAIL: {e}")

            last_checkpoint_tick = total_ticks

        # Print only at PRINT_INTERVAL boundaries
        if total_ticks - last_print_tick >= PRINT_INTERVAL:
            elapsed = time.time() - t_start
            ticks_per_sec = total_ticks / max(elapsed, 1.0)
            eta_min = (TOTAL_TICKS - total_ticks) / max(ticks_per_sec, 1.0) / 60
            print_report(report)
            # Compact pace + queued events on a single follow-up line
            event_str = (" | " + " ".join(pending_events)) if pending_events else ""
            print(f"   [{ticks_per_sec:,.0f} ticks/s, eta {eta_min:.0f}m]{event_str}")
            pending_events = []
            last_print_tick = total_ticks

        # Sanity: if all worlds extinct, stop early
        total_pop = report["aggregate"]["total_pop"]
        if total_pop == 0:
            print("\n*** ALL WORLDS EXTINCT. Halting simulation. ***")
            break

    # Final save
    final_ckpt = os.path.join(CKPT_DIR, 'v6_5_final_state.npz')
    try:
        state_np = jax.tree.map(lambda x: np.array(jax.device_get(x)), worlds_state)
        np.savez(final_ckpt, **{k: v for k, v in state_np._asdict().items()})
        print(f"\nFinal state saved: {final_ckpt}")
    except Exception as e:
        print(f"\n!! final save failed: {e}")

    log_path = os.path.join(CKPT_DIR, 'v6_5_reports.json')
    try:
        with open(log_path, 'w') as f:
            json.dump(reports_history, f, indent=1)
        print(f"Reports log: {log_path}")
    except Exception as e:
        print(f"!! log save failed: {e}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
