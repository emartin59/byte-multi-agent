# %% [markdown]
# # Byte-Multi-Agent v4.0: Bootstrapped ES + PBT
#
# This version is a substantial rewrite based on hard lessons from v1-v3.2.
#
# === Why a rewrite ===
#
# v2 and v3.x all hit the same wall: evolutionary strategies (ES) starting
# from random weights gets trapped in a "do nothing" local optimum. Every
# action has a cost; the cheapest policy is the one that takes fewest
# actions; ES gradients point toward stillness. No amount of coordination
# mechanic design (feast, telepathy, species traits) can overcome this —
# they all get absorbed into "be better at doing nothing."
#
# The fundamental issue: ES with parameter-space noise (sigma ~0.02) cannot
# traverse the long flat valley between "random babble" and "useful
# coordination." Random policies happen to sometimes coordinate, but those
# random successes are too noisy relative to the consistent gains from
# reducing action cost.
#
# === The v4 strategy ===
#
# Keep ES (it's what the TPU pod runs efficiently), but:
#
#  1. BOOTSTRAP FROM A TEACHER. Write a simple hand-coded policy in JAX that
#     actively forages. Supervised-train the network to imitate it. This puts
#     ES's starting point in a "take actions" region of parameter space, not
#     the "do nothing" trap.
#
#  2. REPLACE METABOLIC_COST WITH ALIVE_TAX. You pay the tax whether you
#     move or not. Now sitting still is equally expensive as exploring,
#     removing the gradient toward stillness.
#
#  3. ACTIVITY BONUSES IN FITNESS. Add small rewards for distance traveled,
#     action-type variety, and number of distinct cells visited. Shapes the
#     gradient toward engagement, not just toward survival.
#
#  4. TRUE PBT ON TOP OF ES. 4 separate "tribes" each with their own
#     parameter center. ES improves each tribe independently. Every 30 gens,
#     the worst tribe is replaced by a perturbed copy of the best tribe.
#     Agents from different tribes are sometimes placed together, creating
#     real cross-tribe pressure.
#
#  5. SIMPLIFIED ENVIRONMENT. Dropped: species traits, vision masks, crafting
#     substrate, tool effects, telepathy bonus. Kept: grid world, food, feast,
#     walls, writing, speech, inventory (for future use). We're trying to get
#     basic active foraging + some coordination working. Complexity returns
#     when the foundation does.
#
#  6. PER-TICK INTRINSIC REWARD. Rather than only episode-end fitness,
#     ES uses episode-mean intrinsic reward. This provides denser signal and
#     better credit assignment across the 400-tick episode.

# %%
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import flax.linen as nn
import optax
from typing import NamedTuple
from functools import partial
import time
import numpy as np
import os

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")


# ==========================================
# 1. CONSTANTS
# ==========================================

# --- World Bytes (simplified) ---
EMPTY = 0
WALL = 1
FOOD = 2
SEED = 3
FEAST = 4
WRITE_OFFSET = 5      # 5-30: a-z
SPEAK_OFFSET = 31     # 31-56: A-Z (display only)
AGENT_MARK = 57
NUM_BYTE_TYPES = 64

CHAR_MAP = {
    EMPTY: '.', WALL: '#', FOOD: '*', SEED: ',', FEAST: '&',
    AGENT_MARK: '@',
}
for i in range(26):
    CHAR_MAP[i + WRITE_OFFSET] = chr(97 + i)
    CHAR_MAP[i + SPEAK_OFFSET] = chr(65 + i)

# --- Environment ---
VISION_RADIUS = 7
VISION_SIZE = 2 * VISION_RADIUS + 1
MAX_AGENTS = 8
EPISODE_LENGTH = 300
INVENTORY_SIZE = 4   # unused but keep for future
AUDIO_RANGE = 10
AUDIO_PERSIST_TICKS = 3
HIDDEN_SIZE = 128

INITIAL_ARENA = 20
INITIAL_ACTIVE = 6
GRID_SIZE = 64

MOVE_DIRS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)

# --- Energy / economy ---
INITIAL_ENERGY = 50.0
MAX_ENERGY = 100.0
FOOD_ENERGY = 15.0
FEAST_ENERGY_PER_AGENT = 40.0
FEAST_RADIUS = 3
FEAST_WINDOW_TICKS = 8

# CRITICAL CHANGE: alive_tax replaces metabolic_cost. You pay it whether you
# move or not. Removes the "do nothing to save energy" gradient.
ALIVE_TAX = 0.4                  # per tick, unconditional on actions

# Action costs are now MUCH smaller — just enough to prevent pure babbling
MOVE_COST = 0.05                 # tiny — moving is basically free now
WRITE_COST = 0.15
SPEAK_COST = 0.05
PICKUP_COST = 0.1
USE_TOOL_COST = 0.1

# --- Ecology ---
SEED_TO_FOOD_PROB = 0.004
INITIAL_FOOD_DENSITY = 0.05
INITIAL_SEED_DENSITY = 0.05
INITIAL_FEAST_DENSITY = 0.025
FEAST_REGROW_PROB = 0.0015

# --- PBT ---
NUM_TRIBES = 4
TRIBE_TOURNAMENT_INTERVAL = 30   # every N gens, replace worst tribe
TRIBE_MUTATION_SIGMA = 0.05      # noise added when cloning winner → loser

# --- ES Hyperparameters (per tribe) ---
POP_SIZE = 64                    # per tribe. Total evals = NUM_TRIBES * POP_SIZE = 256
NUM_ENVS_PER_MEMBER = 4
LR = 0.01
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.005

# --- Teacher bootstrap ---
TEACHER_TRAIN_STEPS = 2000       # SGD steps for supervised teacher distillation
TEACHER_BATCH_SIZE = 128
TEACHER_LR = 3e-4
TEACHER_EPISODES = 50            # how many teacher-controlled episodes to collect data from
                                   # 50 × 300 × 8 = 120K samples, plenty for simple policy

# --- Adaptive Noise ---
NOISE_STD_INIT = 0.015
NOISE_STD_MIN = 0.005
NOISE_STD_MAX = 0.025

# --- Action sampling ---
ACTION_TEMP = 0.5


# ==========================================
# 2. ENVIRONMENT STATE
# ==========================================

class EnvState(NamedTuple):
    grid: jnp.ndarray
    agent_pos: jnp.ndarray
    agent_alive: jnp.ndarray
    agent_energy: jnp.ndarray
    agent_hidden: jnp.ndarray
    agent_tribe: jnp.ndarray          # (MAX_AGENTS,) int — which tribe each agent is from
    audio_buffer: jnp.ndarray
    agent_underfoot: jnp.ndarray
    survival_ticks: jnp.ndarray
    energy_earned: jnp.ndarray
    feast_bonus_earned: jnp.ndarray
    recent_feast_tick: jnp.ndarray
    recent_feast_pos: jnp.ndarray
    # NEW in v4: activity tracking for intrinsic reward
    cells_visited: jnp.ndarray        # (MAX_AGENTS, GRID_SIZE, GRID_SIZE) bool — was too big, use counter
    distinct_cell_count: jnp.ndarray  # (MAX_AGENTS,) int — approx distinct cells visited
    action_type_counts: jnp.ndarray   # (MAX_AGENTS, 4) int — move/speak/write/puu counts
    total_movement: jnp.ndarray       # (MAX_AGENTS,) int — total successful moves
    tick: jnp.ndarray
    rng: jnp.ndarray


def init_env(key, arena_h, arena_w, grid_size, num_active, tribe_assignments):
    """tribe_assignments: (MAX_AGENTS,) int, which tribe each slot belongs to.
    This is passed in from the caller because PBT controls it."""
    k_blocks, k_agents, k_rng = random.split(key, 3)

    rows = jnp.arange(grid_size)[:, None]
    cols = jnp.arange(grid_size)[None, :]
    off_r = (grid_size - arena_h) // 2
    off_c = (grid_size - arena_w) // 2
    in_arena = (rows >= off_r) & (rows < off_r + arena_h) & (cols >= off_c) & (cols < off_c + arena_w)
    on_border = (rows == off_r) | (rows == off_r + arena_h - 1) | (cols == off_c) | (cols == off_c + arena_w - 1)
    playable = in_arena & ~on_border
    grid = jnp.where(in_arena, jnp.where(on_border, WALL, EMPTY), WALL)

    k1, k2, k3 = random.split(k_blocks, 3)
    food_noise = random.uniform(k1, (grid_size, grid_size))
    seed_noise = random.uniform(k2, (grid_size, grid_size))
    feast_noise = random.uniform(k3, (grid_size, grid_size))

    grid = jnp.where(playable & (food_noise < INITIAL_FOOD_DENSITY), FOOD, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (seed_noise < INITIAL_SEED_DENSITY), SEED, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (feast_noise < INITIAL_FEAST_DENSITY), FEAST, grid)

    gumbel = random.gumbel(k_agents, shape=(grid_size * grid_size,))
    is_empty = (grid.reshape(-1) == EMPTY)
    gumbel = jnp.where(is_empty, gumbel, -1e9)
    top_idx = jnp.argsort(-gumbel)[:MAX_AGENTS]
    agent_pos = jnp.stack([top_idx // grid_size, top_idx % grid_size], axis=-1)
    agent_alive = jnp.arange(MAX_AGENTS) < num_active
    agent_energy = jnp.where(agent_alive, INITIAL_ENERGY, 0.0)
    agent_hidden = jnp.zeros((MAX_AGENTS, HIDDEN_SIZE), dtype=jnp.float32)
    agent_underfoot = jnp.full((MAX_AGENTS,), EMPTY, dtype=jnp.int32)

    audio_buffer = jnp.zeros((MAX_AGENTS, AUDIO_PERSIST_TICKS), dtype=jnp.int32)

    survival_ticks = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    energy_earned = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    feast_bonus_earned = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    recent_feast_tick = jnp.full((MAX_AGENTS,), -1000, dtype=jnp.int32)
    recent_feast_pos = jnp.zeros((MAX_AGENTS, 2), dtype=jnp.int32)

    # NOTE: cells_visited as a dense grid would be (MAX_AGENTS, GRID_SIZE, GRID_SIZE)
    # = 8 * 64 * 64 = 32K bools per state, per env. Too much.
    # Instead, track just a count: every tick we increment distinct_cell_count by 1 if
    # the agent just entered a cell it hadn't been on last tick. Approximation but cheap.
    # We set cells_visited to a dummy 1-dim; the real info is in distinct_cell_count.
    cells_visited = jnp.zeros((MAX_AGENTS,), dtype=jnp.int32)  # unused placeholder
    distinct_cell_count = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    action_type_counts = jnp.zeros((MAX_AGENTS, 4), dtype=jnp.int32)
    total_movement = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)

    return EnvState(
        grid=grid, agent_pos=agent_pos, agent_alive=agent_alive,
        agent_energy=agent_energy, agent_hidden=agent_hidden,
        agent_tribe=tribe_assignments,
        audio_buffer=audio_buffer,
        agent_underfoot=agent_underfoot,
        survival_ticks=survival_ticks,
        energy_earned=energy_earned,
        feast_bonus_earned=feast_bonus_earned,
        recent_feast_tick=recent_feast_tick,
        recent_feast_pos=recent_feast_pos,
        cells_visited=cells_visited,
        distinct_cell_count=distinct_cell_count,
        action_type_counts=action_type_counts,
        total_movement=total_movement,
        tick=jnp.int32(0), rng=k_rng,
    )


# ==========================================
# 3. OBSERVATIONS
# ==========================================

def get_all_obs(state, grid_size, last_speak):
    """Return (obs_grid, obs_audio) per agent."""
    grid = state.grid

    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        body = jnp.where(last_speak[i] > 0,
                         jnp.minimum(last_speak[i] - 1 + SPEAK_OFFSET, AGENT_MARK - 1),
                         AGENT_MARK)
        return jnp.where(state.agent_alive[i], g.at[r, c].set(body), g), None

    composed, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    padded = jnp.pad(composed, VISION_RADIUS, constant_values=WALL)

    def get_one(pos):
        return lax.dynamic_slice(padded, (pos[0], pos[1]), (VISION_SIZE, VISION_SIZE))

    obs_bytes = vmap(get_one)(state.agent_pos)
    obs_norm = obs_bytes.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)

    def audio_for_listener(listener_idx):
        lp = state.agent_pos[listener_idx]
        dp = state.agent_pos - lp[None, :]
        chebyshev = jnp.maximum(jnp.abs(dp[:, 0]), jnp.abs(dp[:, 1]))
        audible = (chebyshev <= AUDIO_RANGE) & state.agent_alive & \
                  (jnp.arange(MAX_AGENTS) != listener_idx)
        gated = state.audio_buffer * audible[:, None].astype(jnp.int32)
        return gated.reshape(-1).astype(jnp.float32) / 27.0

    obs_audio = vmap(audio_for_listener)(jnp.arange(MAX_AGENTS))
    return obs_norm, obs_audio


# ==========================================
# 4. PHYSICS
# ==========================================

def is_passable_byte(b):
    return (b == EMPTY) | (b == FOOD) | (b == SEED) | (b == FEAST) | \
           ((b >= WRITE_OFFSET) & (b < SPEAK_OFFSET))


def step_env(state, actions, grid_size):
    """actions: (MAX_AGENTS, 4) — [move, speak, write, puu]
    PUU simplified: 0 noop, 1 pickup(unused), 2 use(unused), 3 drop(unused).
    We keep the head for future use but currently all non-noop puu is no-op.
    """
    move_acts = actions[:, 0]
    speak_acts = actions[:, 1]
    write_acts = actions[:, 2]
    puu_acts = actions[:, 3]

    curr_pos = state.agent_pos
    dp = MOVE_DIRS[move_acts]
    want_pos = curr_pos + dp
    want_pos = jnp.where(state.agent_alive[:, None], want_pos, curr_pos)
    want_pos = jnp.clip(want_pos, 0, grid_size - 1)

    target_vals = state.grid[want_pos[:, 0], want_pos[:, 1]]
    passable = is_passable_byte(target_vals) & state.agent_alive

    same_target = jnp.all(want_pos[:, None, :] == want_pos[None, :, :], axis=-1)
    eye = jnp.eye(MAX_AGENTS, dtype=jnp.bool_)
    has_conflict = jnp.any(same_target & passable[:, None] & passable[None, :] & ~eye, axis=1)

    def check_occupied(i):
        same = jnp.all(curr_pos == want_pos[i][None, :], axis=-1) & state.agent_alive
        same = same & (jnp.arange(MAX_AGENTS) != i)
        return jnp.any(same)
    other_pos_mask = vmap(check_occupied)(jnp.arange(MAX_AGENTS))

    final_move = passable & ~has_conflict & ~other_pos_mask
    final_pos = jnp.where(final_move[:, None], want_pos, curr_pos)
    moved = final_move & jnp.any(dp != 0, axis=-1)

    ate_food = moved & (target_vals == FOOD)
    stepped_on_feast = moved & (target_vals == FEAST)

    new_underfoot_moved = jnp.where(ate_food, SEED,
                            jnp.where(stepped_on_feast, EMPTY, target_vals))
    new_underfoot = jnp.where(moved, new_underfoot_moved, state.agent_underfoot)

    leave_byte = jnp.where(write_acts > 0, write_acts - 1 + WRITE_OFFSET, state.agent_underfoot)

    new_grid = state.grid
    alive_moved = moved & state.agent_alive
    leave_r = jnp.where(alive_moved, curr_pos[:, 0], 0)
    leave_c = jnp.where(alive_moved, curr_pos[:, 1], 0)
    deposit_vals = jnp.where(alive_moved, leave_byte, new_grid[leave_r, leave_c])
    new_grid = new_grid.at[leave_r, leave_c].set(deposit_vals)

    eat_r = jnp.where(ate_food, final_pos[:, 0], 0)
    eat_c = jnp.where(ate_food, final_pos[:, 1], 0)
    keep_vals = jnp.where(ate_food, EMPTY, new_grid[eat_r, eat_c])
    new_grid = new_grid.at[eat_r, eat_c].set(keep_vals)

    # FEAST temporal-window mechanic (from v3.1)
    feast_now = stepped_on_feast & state.agent_alive
    fp_pos = final_pos
    rfp = state.recent_feast_pos
    dpij = fp_pos[:, None, :] - rfp[None, :, :]
    cheb_ij = jnp.maximum(jnp.abs(dpij[:, :, 0]), jnp.abs(dpij[:, :, 1]))
    j_recent = (state.tick - state.recent_feast_tick) <= FEAST_WINDOW_TICKS
    j_recent = j_recent & state.agent_alive
    pair_valid = (cheb_ij <= FEAST_RADIUS) & feast_now[:, None] & j_recent[None, :]
    pair_valid = pair_valid & ~eye

    has_feast_partner_now = jnp.any(pair_valid, axis=1)
    gets_partner_credit = jnp.any(pair_valid, axis=0)

    bonus_for_now = jnp.where(has_feast_partner_now, FEAST_ENERGY_PER_AGENT, 0.0)
    bonus_for_prior = jnp.where(gets_partner_credit, FEAST_ENERGY_PER_AGENT, 0.0)
    feast_bonus_this_tick = bonus_for_now + bonus_for_prior

    new_recent_feast_tick = jnp.where(feast_now, state.tick, state.recent_feast_tick)
    new_recent_feast_pos = jnp.where(feast_now[:, None], final_pos, state.recent_feast_pos)

    feast_eat_r = jnp.where(stepped_on_feast, final_pos[:, 0], 0)
    feast_eat_c = jnp.where(stepped_on_feast, final_pos[:, 1], 0)
    keep_feast = jnp.where(stepped_on_feast, EMPTY, new_grid[feast_eat_r, feast_eat_c])
    new_grid = new_grid.at[feast_eat_r, feast_eat_c].set(keep_feast)

    # Seed regrowth
    k_regrow, new_rng = random.split(state.rng)
    regrow_noise = random.uniform(k_regrow, (grid_size, grid_size))
    is_seed = (new_grid == SEED)
    regrow_trigger = is_seed & (regrow_noise < SEED_TO_FOOD_PROB)
    new_grid = jnp.where(regrow_trigger, FOOD, new_grid)

    # Feast regrowth
    k_feast, new_rng = random.split(new_rng)
    feast_regrow_noise = random.uniform(k_feast, (grid_size, grid_size))
    is_empty_cell = (new_grid == EMPTY)
    feast_regrow_trigger = is_empty_cell & (feast_regrow_noise < FEAST_REGROW_PROB)
    new_grid = jnp.where(feast_regrow_trigger, FEAST, new_grid)

    # ENERGY: alive_tax (UNCONDITIONAL on actions), then small action costs
    new_energy = state.agent_energy - jnp.where(state.agent_alive, ALIVE_TAX, 0.0)
    new_energy = new_energy - jnp.where(moved, MOVE_COST, 0.0)
    new_energy = new_energy - jnp.where((speak_acts > 0) & state.agent_alive, SPEAK_COST, 0.0)
    wrote_something = (write_acts > 0) & moved
    new_energy = new_energy - jnp.where(wrote_something, WRITE_COST, 0.0)

    new_energy = new_energy + jnp.where(ate_food, FOOD_ENERGY, 0.0)
    new_energy = new_energy + feast_bonus_this_tick
    new_energy = jnp.clip(new_energy, 0.0, MAX_ENERGY)

    new_energy_earned = state.energy_earned + jnp.where(ate_food, FOOD_ENERGY, 0.0) + feast_bonus_this_tick
    new_feast_bonus_earned = state.feast_bonus_earned + feast_bonus_this_tick

    new_alive = state.agent_alive & (new_energy > 0.0)
    new_survival = state.survival_ticks + new_alive.astype(jnp.float32)

    # Activity tracking (for fitness shaping)
    new_total_movement = state.total_movement + moved.astype(jnp.float32)
    # Approximate distinct cells: increment whenever agent moves to a NEW cell.
    # (True distinct tracking would need a visited-grid; this is the cheap
    # approximation that's correlated with exploration.)
    new_distinct = state.distinct_cell_count + moved.astype(jnp.float32)
    # Action counts
    did_speak = (speak_acts > 0) & state.agent_alive
    did_write = (write_acts > 0) & state.agent_alive
    did_puu = (puu_acts > 0) & state.agent_alive
    act_delta = jnp.stack([
        moved.astype(jnp.int32),
        did_speak.astype(jnp.int32),
        did_write.astype(jnp.int32),
        did_puu.astype(jnp.int32),
    ], axis=-1)
    new_action_type_counts = state.action_type_counts + act_delta

    # Audio update
    new_audio_buffer = jnp.concatenate([
        speak_acts[:, None],
        state.audio_buffer[:, :-1]
    ], axis=1)

    return state._replace(
        grid=new_grid,
        agent_pos=final_pos,
        agent_alive=new_alive,
        agent_energy=new_energy,
        agent_underfoot=new_underfoot,
        audio_buffer=new_audio_buffer,
        survival_ticks=new_survival,
        energy_earned=new_energy_earned,
        feast_bonus_earned=new_feast_bonus_earned,
        recent_feast_tick=new_recent_feast_tick,
        recent_feast_pos=new_recent_feast_pos,
        distinct_cell_count=new_distinct,
        action_type_counts=new_action_type_counts,
        total_movement=new_total_movement,
        tick=state.tick + 1,
        rng=new_rng,
    )


# ==========================================
# 5. NETWORK
# ==========================================

class AgentNet(nn.Module):
    @nn.compact
    def __call__(self, obs, audio, proprio, hidden):
        obs_bf = obs.astype(jnp.bfloat16)
        audio_bf = audio.astype(jnp.bfloat16)
        proprio_bf = proprio.astype(jnp.bfloat16)
        hidden_bf = hidden.astype(jnp.bfloat16)

        x = obs_bf[None, :, :, None]
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x_flat = x.reshape(-1)

        a = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.float32)(audio_bf)
        a = nn.relu(a)

        feat = jnp.concatenate([x_flat, a, proprio_bf], axis=-1)
        feat = nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16,
                        param_dtype=jnp.float32)(feat)
        feat = nn.relu(feat)

        zr = nn.Dense(2 * HIDDEN_SIZE, dtype=jnp.bfloat16,
                      param_dtype=jnp.float32)(
            jnp.concatenate([feat, hidden_bf], axis=-1))
        z = nn.sigmoid(zr[:HIDDEN_SIZE])
        r = nn.sigmoid(zr[HIDDEN_SIZE:])
        candidate = nn.tanh(
            nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16,
                     param_dtype=jnp.float32)(
                jnp.concatenate([feat, r * hidden_bf], axis=-1))
        )
        new_hidden = (1.0 - z) * hidden_bf + z * candidate

        h32 = new_hidden.astype(jnp.float32)
        move_logits  = nn.Dense(5,  param_dtype=jnp.float32)(h32)
        speak_logits = nn.Dense(27, param_dtype=jnp.float32)(h32)
        write_logits = nn.Dense(27, param_dtype=jnp.float32)(h32)
        puu_logits   = nn.Dense(4,  param_dtype=jnp.float32)(h32)

        return (move_logits, speak_logits, write_logits, puu_logits, h32)


def build_proprioception(state):
    energy_norm = state.agent_energy / MAX_ENERGY
    tribe_onehot = jax.nn.one_hot(state.agent_tribe, NUM_TRIBES)
    proprio = jnp.concatenate([energy_norm[:, None], tribe_onehot], axis=-1)
    return proprio


PROPRIO_DIM = 1 + NUM_TRIBES


# ==========================================
# 6. TEACHER POLICY (rule-based, in JAX)
# ==========================================
#
# The teacher is a simple rule-based policy that:
# - Moves toward the nearest visible FOOD or FEAST
# - Occasionally speaks a token correlated with what it sees
# - Writes letters occasionally
# - Takes random other actions to maintain diversity
#
# This is NOT the policy we want agents to end up with — it's the bootstrap
# signal to get them out of the "do nothing" valley. After distillation, ES
# takes over and can improve on it (or diverge from it) freely.

def teacher_action(obs, audio, proprio, hidden, key):
    """obs: (V, V) normalized bytes. Returns (action, new_hidden).
    action: int32[4] — [move, speak, write, puu]"""
    # Un-normalize
    obs_bytes = jnp.round(obs * (NUM_BYTE_TYPES - 1)).astype(jnp.int32)

    # Find nearest food or feast
    center = VISION_RADIUS
    is_target = (obs_bytes == FOOD) | (obs_bytes == FEAST)
    # Distances (Manhattan) from center
    rows_idx = jnp.arange(VISION_SIZE)
    cols_idx = jnp.arange(VISION_SIZE)
    dist = jnp.abs(rows_idx[:, None] - center) + jnp.abs(cols_idx[None, :] - center)
    # Inf out non-targets
    masked_dist = jnp.where(is_target, dist, 999)
    has_target = jnp.any(is_target)

    # Find argmin cell
    flat_idx = jnp.argmin(masked_dist)
    tr = flat_idx // VISION_SIZE
    tc = flat_idx % VISION_SIZE

    # Decide move direction: step toward (tr, tc) from center
    dr = tr - center
    dc = tc - center
    # Preferred direction:
    # 0=stay, 1=up(-r), 2=down(+r), 3=right(+c), 4=left(-c)
    prefer_vertical = jnp.abs(dr) > jnp.abs(dc)
    vert_move = jnp.where(dr < 0, 1, jnp.where(dr > 0, 2, 0))
    horz_move = jnp.where(dc > 0, 3, jnp.where(dc < 0, 4, 0))
    target_move = jnp.where(prefer_vertical, vert_move, horz_move)
    # Fallback: if dr and dc both zero, stay
    target_move = jnp.where((dr == 0) & (dc == 0), 0, target_move)

    # Random walk if no target
    k1, k2, k3, k4, k5 = random.split(key, 5)
    random_move = random.randint(k1, (), 1, 5)
    move = jnp.where(has_target, target_move, random_move)

    # Speak: if we see a target, speak a token derived from position (teaches
    # "speech can correlate with what I see"). Else silent most of the time.
    # Token = 1 + (flat_idx % 26) when target visible; small random chance
    # otherwise.
    target_speak_token = 1 + (flat_idx % 26)
    should_speak_randomly = random.uniform(k2, ()) < 0.10
    random_speak = random.randint(k3, (), 1, 27)
    speak = jnp.where(has_target, target_speak_token,
              jnp.where(should_speak_randomly, random_speak, 0))

    # Write: 15% chance of writing a letter
    should_write = random.uniform(k4, ()) < 0.15
    write_letter = random.randint(k5, (), 1, 27)
    write = jnp.where(should_write, write_letter, 0)

    # puu: almost always 0 (noop)
    puu = jnp.int32(0)

    action = jnp.array([move, speak, write, puu], dtype=jnp.int32)
    # Hidden state: teacher doesn't use it but we need to return something.
    # Zero out so the network learns to ignore hidden during distillation.
    new_hidden = jnp.zeros_like(hidden)
    return action, new_hidden


def collect_teacher_data(key, num_episodes, grid_size, arena_h, arena_w, num_active):
    """Run the teacher for num_episodes, collecting (obs, audio, proprio, hidden, action)
    tuples. Returns batched arrays.
    """
    total_steps = num_episodes * EPISODE_LENGTH * MAX_AGENTS

    def run_one_episode(ek):
        # Random tribe assignment
        k_tribe, k_init, k_run = random.split(ek, 3)
        tribes = random.randint(k_tribe, (MAX_AGENTS,), 0, NUM_TRIBES)
        env_state = init_env(k_init, arena_h, arena_w, grid_size, num_active, tribes)

        def step_fn(carry, step_key):
            state, last_speak = carry
            obs, audio = get_all_obs(state, grid_size, last_speak)
            proprio = build_proprioception(state)

            # Per-agent teacher actions
            agent_keys = random.split(step_key, MAX_AGENTS)
            def one_agent(o, a, p, h, k):
                return teacher_action(o, a, p, h, k)

            actions_pair = vmap(one_agent)(obs, audio, proprio, state.agent_hidden, agent_keys)
            actions, new_hidden = actions_pair

            # Record data: (obs, audio, proprio, hidden_in, action_taken)
            data_tuple = (obs, audio, proprio, state.agent_hidden, actions, state.agent_alive)

            state = state._replace(agent_hidden=new_hidden)
            new_state = step_env(state, actions, grid_size)
            return (new_state, actions[:, 1]), data_tuple

        step_keys = random.split(k_run, EPISODE_LENGTH)
        _, data = lax.scan(step_fn, (env_state, jnp.zeros(MAX_AGENTS, dtype=jnp.int32)), step_keys)
        return data

    episode_keys = random.split(key, num_episodes)
    all_data = vmap(run_one_episode)(episode_keys)
    # all_data is a tuple of arrays with leading axes (num_episodes, EPISODE_LENGTH, MAX_AGENTS, ...)
    obs_all, audio_all, proprio_all, hidden_all, actions_all, alive_all = all_data

    # Flatten to (N, ...) — N = num_episodes * EPISODE_LENGTH * MAX_AGENTS
    def flat(x):
        return x.reshape((-1,) + x.shape[3:])
    obs_flat = flat(obs_all)
    audio_flat = flat(audio_all)
    proprio_flat = flat(proprio_all)
    hidden_flat = flat(hidden_all)
    actions_flat = flat(actions_all)
    alive_flat = flat(alive_all)

    # Keep only samples where agent was alive
    alive_mask = alive_flat
    return obs_flat, audio_flat, proprio_flat, hidden_flat, actions_flat, alive_mask


# ==========================================
# 7. SUPERVISED DISTILLATION
# ==========================================

def distill_teacher(net, apply_fn, params, data, num_steps, batch_size, lr, key):
    """Train network to predict teacher actions via cross-entropy."""
    obs, audio, proprio, hidden, actions, alive = data
    N = obs.shape[0]
    print(f"  Teacher data: {N} (obs, action) pairs, {int(alive.sum())} alive")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch):
        b_obs, b_audio, b_proprio, b_hidden, b_act, b_alive = batch

        def per_sample(o, au, p, h, a, al):
            ml, sl, wl, pul, _ = apply_fn(params, o, au, p, h)
            loss_m = optax.softmax_cross_entropy_with_integer_labels(ml, a[0])
            loss_s = optax.softmax_cross_entropy_with_integer_labels(sl, a[1])
            loss_w = optax.softmax_cross_entropy_with_integer_labels(wl, a[2])
            loss_p = optax.softmax_cross_entropy_with_integer_labels(pul, a[3])
            total = loss_m + loss_s + loss_w + loss_p
            return total * al.astype(jnp.float32)

        losses = vmap(per_sample)(b_obs, b_audio, b_proprio, b_hidden, b_act, b_alive)
        return jnp.sum(losses) / (jnp.sum(b_alive.astype(jnp.float32)) + 1e-6)

    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def train_step(params, opt_state, batch):
        loss, grads = grad_fn(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    for step in range(num_steps):
        key, sk = random.split(key)
        idx = random.randint(sk, (batch_size,), 0, N)
        batch = (obs[idx], audio[idx], proprio[idx], hidden[idx], actions[idx], alive[idx])
        params, opt_state, loss = train_step(params, opt_state, batch)
        if step % 200 == 0:
            print(f"    step {step:5d} loss={float(loss):.3f}")
    return params


# ==========================================
# 8. ES CORE
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


def rank_fitness_shape(fitness):
    n = fitness.shape[0]
    ranks = jnp.argsort(jnp.argsort(-fitness)).astype(jnp.float32)
    log_util = jnp.maximum(0.0, jnp.log(n / 2.0 + 1.0) - jnp.log(ranks + 1.0))
    utilities = log_util / jnp.sum(log_util) - 1.0 / n
    return utilities


# ==========================================
# 9. ACTION SAMPLING
# ==========================================

def sample_actions(tribe_params, obs_all, audio_all, proprio_all, hidden_all,
                   tribe_ids, key, apply_fn, temperature):
    """tribe_params: pytree where leading dim = NUM_TRIBES.
    tribe_ids: (MAX_AGENTS,) — tribe index for each agent slot.
    """
    keys = random.split(key, MAX_AGENTS * 4).reshape(MAX_AGENTS, 4, 2)

    def act_one(obs, audio, proprio, hidden, tribe, ks):
        my_params = jax.tree.map(lambda x: x[tribe], tribe_params)
        ml, sl, wl, puul, new_hidden = apply_fn(my_params, obs, audio, proprio, hidden)

        def sample(logits, k):
            g = random.gumbel(k, logits.shape)
            return jnp.argmax(logits / temperature + g)

        move = sample(ml, ks[0])
        speak = sample(sl, ks[1])
        write = sample(wl, ks[2])
        puu = sample(puul, ks[3])
        return (jnp.array([move, speak, write, puu], dtype=jnp.int32), new_hidden)

    actions, new_hidden = vmap(act_one)(obs_all, audio_all, proprio_all,
                                          hidden_all, tribe_ids, keys)
    return actions, new_hidden


# ==========================================
# 10. FITNESS WITH ACTIVITY SHAPING
# ==========================================

def compute_per_tribe_fitness(final_state):
    """Returns (per_tribe_fitness: (NUM_TRIBES,), metrics: (4,))
    Fitness per agent =
        survival_ticks
      + 1.0 * energy_earned_from_food
      + 1.5 * feast_bonus_earned           (coordination bonus)
      + 0.05 * total_movement               (activity shaping, small)
      + 0.1 * distinct_cell_count           (exploration shaping)
      + 0.5 * action_type_variety          (behavior diversity, mildly)
    """
    survival = final_state.survival_ticks
    earned = final_state.energy_earned
    feast = final_state.feast_bonus_earned
    movement = final_state.total_movement
    distinct = final_state.distinct_cell_count

    # Action variety: entropy-like measure. Count how many of the 4 action
    # types were used at least 5 times.
    act_counts = final_state.action_type_counts.astype(jnp.float32)
    act_used = (act_counts >= 5).astype(jnp.float32)
    variety = jnp.sum(act_used, axis=-1)  # 0..4

    per_agent_fit = (survival
                     + 1.0 * earned
                     + 1.5 * feast
                     + 0.05 * movement
                     + 0.1 * distinct
                     + 0.5 * variety)

    spawned = (final_state.survival_ticks > 0) | (final_state.agent_energy > 0)
    per_agent_fit = per_agent_fit * spawned.astype(jnp.float32)

    tribes = final_state.agent_tribe
    one_hot = jax.nn.one_hot(tribes, NUM_TRIBES)
    tribe_fit = jnp.sum(one_hot * per_agent_fit[:, None], axis=0)
    tribe_count = jnp.sum(one_hot * spawned[:, None].astype(jnp.float32), axis=0)
    tribe_fit_mean = tribe_fit / (tribe_count + 1e-3)

    # Metrics
    total_feast = jnp.sum(feast)
    total_food = jnp.sum(earned) - total_feast
    total_movement = jnp.sum(movement)
    total_variety = jnp.sum(variety)
    metrics = jnp.stack([total_feast, total_food, total_movement, total_variety])

    return tribe_fit_mean, metrics


# ==========================================
# 11. EPISODE RUNNER
# ==========================================

def run_episode(tribe_params_stack, apply_fn, env_state, grid_size, ep_key):
    def step_fn(carry, step_key):
        state, last_speak = carry
        obs, audio = get_all_obs(state, grid_size, last_speak)
        proprio = build_proprioception(state)
        actions, new_hidden = sample_actions(
            tribe_params_stack, obs, audio, proprio, state.agent_hidden,
            state.agent_tribe, step_key, apply_fn, ACTION_TEMP)
        state = state._replace(agent_hidden=new_hidden)
        new_state = step_env(state, actions, grid_size)
        return (new_state, actions[:, 1]), None

    init_speak = jnp.zeros(MAX_AGENTS, dtype=jnp.int32)
    step_keys = random.split(ep_key, EPISODE_LENGTH)
    (final_state, _), _ = lax.scan(
        step_fn, (env_state, init_speak), step_keys)
    return final_state


# ==========================================
# 12. RENDERING
# ==========================================

def render_snapshot(grid_np, arena_h, arena_w, grid_size, agent_pos_np,
                     agent_alive_np, agent_energy_np, agent_tribe_np):
    off_r = (grid_size - arena_h) // 2
    off_c = (grid_size - arena_w) // 2
    disp = np.array(grid_np).copy()
    for i in range(MAX_AGENTS):
        if agent_alive_np[i]:
            r, c = int(agent_pos_np[i, 0]), int(agent_pos_np[i, 1])
            disp[r, c] = AGENT_MARK

    lines = []
    for r in range(off_r, off_r + arena_h):
        row = ""
        for c in range(off_c, off_c + arena_w):
            row += CHAR_MAP.get(int(disp[r, c]), '?')
        lines.append(row)

    tribe_summary = ",".join(
        f"t{int(agent_tribe_np[i])}={agent_energy_np[i]:.0f}"
        for i in range(MAX_AGENTS) if agent_alive_np[i])
    stat = (f"  alive={int(agent_alive_np.sum())}/{MAX_AGENTS}  "
            f"energy=[{agent_energy_np.min():.0f},{agent_energy_np.max():.0f}] "
            f"mean={agent_energy_np.mean():.0f}  tribes: {tribe_summary}")
    return "\n".join(lines) + "\n" + stat


# ==========================================
# 13. MAIN TRAINING LOOP
# ==========================================

def main():
    print("=" * 60)
    print("  BYTE-MULTI-AGENT v4.0: BOOTSTRAPPED ES + PBT")
    print("=" * 60)

    num_devices = jax.device_count()
    print(f"Using {num_devices} device(s)")
    # Total ES members per generation = NUM_TRIBES * POP_SIZE
    total_pop = NUM_TRIBES * POP_SIZE
    assert total_pop % num_devices == 0, \
        f"NUM_TRIBES*POP_SIZE ({total_pop}) must be divisible by num_devices ({num_devices})"

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), axis_names=('pop',))

    key = random.PRNGKey(42)
    net = AgentNet()
    apply_fn = net.apply

    k_init, key = random.split(key)
    dummy_obs = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_audio = jnp.zeros((MAX_AGENTS * AUDIO_PERSIST_TICKS,), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
    dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
    params_template = net.init(k_init, dummy_obs, dummy_audio, dummy_proprio, dummy_hidden)

    flat = flatten_params(params_template)
    num_params = flat.shape[0]
    print(f"Network parameters: {num_params:,}")
    print(f"Tribes: {NUM_TRIBES}, per-tribe pop: {POP_SIZE}, total evals/gen: {total_pop}")

    CKPT_DIR = '/tmp' if os.path.exists('/tmp') else '.'
    CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v4_params.npy')
    GEN_CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v4_gen.npy')
    TEACHER_CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v4_teacher.npy')

    # --- PHASE 1: teacher bootstrap ---
    resume_from_teacher = os.path.exists(TEACHER_CKPT_PATH)
    if resume_from_teacher:
        print(f"\n=== PHASE 1: loading cached teacher-distilled params from {TEACHER_CKPT_PATH} ===")
        teacher_flat = jnp.array(np.load(TEACHER_CKPT_PATH))
    else:
        print(f"\n=== PHASE 1: teacher bootstrap ===")
        print(f"  Collecting {TEACHER_EPISODES} episodes of teacher data...")
        k_teach, key = random.split(key)
        t0 = time.time()
        teacher_data = collect_teacher_data(
            k_teach, TEACHER_EPISODES, GRID_SIZE, INITIAL_ARENA, INITIAL_ARENA,
            INITIAL_ACTIVE)
        print(f"  Collection took {time.time() - t0:.1f}s")

        print(f"  Distilling into network for {TEACHER_TRAIN_STEPS} SGD steps...")
        k_distill, key = random.split(key)
        teacher_params = distill_teacher(
            net, apply_fn, params_template, teacher_data,
            TEACHER_TRAIN_STEPS, TEACHER_BATCH_SIZE, TEACHER_LR, k_distill)
        teacher_flat = flatten_params(teacher_params)
        np.save(TEACHER_CKPT_PATH, np.array(jax.device_get(teacher_flat)))
        print(f"  Teacher params cached to {TEACHER_CKPT_PATH}")

    # --- Initialize tribes ---
    # Each tribe starts as teacher + small random noise (so they diverge during ES)
    k_init_tribes = random.split(key, NUM_TRIBES + 1)
    key = k_init_tribes[0]
    tribe_centers_list = []
    for t in range(NUM_TRIBES):
        tribe_noise = random.normal(k_init_tribes[t + 1], (num_params,)) * TRIBE_MUTATION_SIGMA
        tribe_centers_list.append(teacher_flat + tribe_noise)
    tribe_centers = jnp.stack(tribe_centers_list)  # (NUM_TRIBES, num_params)

    # Resume if we have a prior v4 checkpoint
    start_gen = 0
    if os.path.exists(CKPT_PATH):
        loaded = jnp.array(np.load(CKPT_PATH))
        if loaded.shape == tribe_centers.shape:
            tribe_centers = loaded
            print(f"  Resumed tribe centers from {CKPT_PATH}")
            if os.path.exists(GEN_CKPT_PATH):
                gen_info = np.load(GEN_CKPT_PATH, allow_pickle=True).item()
                start_gen = int(gen_info.get('gen', 0))
                print(f"  Resuming from gen {start_gen}")

    # Separate optimizer state per tribe
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LR),
    )
    tribe_opt_states = [optimizer.init(tribe_centers[t]) for t in range(NUM_TRIBES)]

    arena_h = INITIAL_ARENA
    arena_w = INITIAL_ARENA
    grid_size = GRID_SIZE
    num_active = INITIAL_ACTIVE
    half_pop = POP_SIZE // 2
    per_device = total_pop // num_devices

    print(f"Population: {POP_SIZE} mirrored per tribe, {NUM_ENVS_PER_MEMBER} envs each")
    print(f"Arena: {arena_h}x{arena_w}, grid {grid_size}, agents {num_active}")
    print(f"Per-device members: {per_device}")
    print(f"Alive tax: {ALIVE_TAX}/tick (unconditional)")
    print()

    print("  Compiling sharded generation fn...")

    # Member indexing scheme (critical to get right):
    #   total_pop = NUM_TRIBES * POP_SIZE     # all members, mirrored
    #   half_pop_per_tribe = POP_SIZE // 2    # requires POP_SIZE even
    #   total_half = NUM_TRIBES * half_pop_per_tribe = total_pop // 2
    #
    # Layout in the (total_pop,) fitness array:
    #   Positive half [0 .. total_half):
    #     tribe 0 members: indices 0 .. half_pop_per_tribe
    #     tribe 1 members: indices half_pop_per_tribe .. 2*half_pop_per_tribe
    #     etc.
    #   Negative half [total_half .. total_pop):
    #     mirrors of the above — tribe 0 neg starts at total_half, etc.
    #
    # This makes per-tribe fitness extraction trivial: slice by tribe index.

    assert POP_SIZE % 2 == 0, "POP_SIZE must be even (for mirrored ES)"
    half_pop_per_tribe = POP_SIZE // 2
    total_half = NUM_TRIBES * half_pop_per_tribe

    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P(), P(), P('pop'), P('pop')),
             out_specs=(P('pop'), P('pop')),
             check_rep=False)
    def sharded_eval(all_tribe_centers, sigma_val, noise_base_seed,
                     env_keys_shard, ep_keys_shard):
        my_idx = lax.axis_index('pop')
        start = my_idx * per_device

        def eval_one_member(local_idx, env_keys_m, ep_keys_m):
            global_idx = start + local_idx  # in [0, total_pop)
            is_negative = global_idx >= total_half
            global_idx_half = jnp.where(is_negative, global_idx - total_half, global_idx)
            tribe_idx = global_idx_half // half_pop_per_tribe

            row_key = random.fold_in(noise_base_seed, global_idx_half)
            noise = random.normal(row_key, (num_params,))
            noise = jnp.where(is_negative, -noise, noise)

            flat_params = all_tribe_centers[tribe_idx] + sigma_val * noise
            params = unflatten_params(flat_params, params_template)
            # Broadcast params across tribe dim so sample_actions can index by tribe
            tribe_params_stack = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None], (NUM_TRIBES,) + x.shape),
                params)

            def eval_one_env(ek, pk):
                # All agents in the env use this member's tribe (clean within-tribe signal)
                tribes = jnp.full((MAX_AGENTS,), tribe_idx, dtype=jnp.int32)
                env_state = init_env(ek, arena_h, arena_w, grid_size, num_active, tribes)
                final_state = run_episode(tribe_params_stack, apply_fn, env_state,
                                            grid_size, pk)
                fit, met = compute_per_tribe_fitness(final_state)
                return fit[tribe_idx], met

            fits, metrics = vmap(eval_one_env)(env_keys_m, ep_keys_m)
            return jnp.mean(fits), jnp.mean(metrics, axis=0)

        local_indices = jnp.arange(per_device)
        fits, metrics = vmap(eval_one_member)(
            local_indices, env_keys_shard, ep_keys_shard)
        return fits, metrics

    @jit
    def reconstruct_noise(noise_base_seed):
        """Return (total_half, num_params) — positive noise rows."""
        def gen_one_row(base_idx):
            row_key = random.fold_in(noise_base_seed, base_idx)
            return random.normal(row_key, (num_params,))
        return vmap(gen_one_row)(jnp.arange(total_half))

    # Per-tribe ES update
    def es_update_one_tribe(center, noise_tribe, sigma_val, fitness_pos, fitness_neg, opt_state):
        combined_fit = jnp.concatenate([fitness_pos, fitness_neg])  # (POP_SIZE,)
        utilities = rank_fitness_shape(combined_fit)
        pos_util = utilities[:half_pop_per_tribe]
        neg_util = utilities[half_pop_per_tribe:]
        grad_estimate = jnp.dot((pos_util - neg_util), noise_tribe) / (half_pop_per_tribe * sigma_val)
        grad_estimate = grad_estimate - WEIGHT_DECAY * center
        updates, new_opt_state = optimizer.update(-grad_estimate, opt_state, center)
        return optax.apply_updates(center, updates), new_opt_state

    es_update_one_tribe_jit = jit(es_update_one_tribe)

    start_time = time.time()
    last_print_time = start_time
    PRINT_INTERVAL = 120
    current_sigma = NOISE_STD_INIT
    tribe_fit_history = [[] for _ in range(NUM_TRIBES)]

    for gen in range(start_gen, 100000):
        k_noise, k_env, k_ep, key = random.split(key, 4)
        sigma_jnp = jnp.float32(current_sigma)
        noise_base_seed = k_noise

        env_keys = random.split(k_env, total_pop * NUM_ENVS_PER_MEMBER).reshape(
            total_pop, NUM_ENVS_PER_MEMBER, 2)
        ep_keys = random.split(k_ep, total_pop * NUM_ENVS_PER_MEMBER).reshape(
            total_pop, NUM_ENVS_PER_MEMBER, 2)

        fits, metrics_all = sharded_eval(
            tribe_centers, sigma_jnp, noise_base_seed, env_keys, ep_keys)
        # fits: (total_pop,); metrics_all: (total_pop, 4)

        noise_half = reconstruct_noise(noise_base_seed)  # (total_half, num_params)

        # Split fitness and noise per tribe, update each tribe's center
        new_tribe_centers = []
        new_opt_states = []
        for t in range(NUM_TRIBES):
            tribe_start_pos = t * half_pop_per_tribe
            tribe_end_pos = (t + 1) * half_pop_per_tribe
            tribe_start_neg = total_half + tribe_start_pos
            tribe_end_neg = total_half + tribe_end_pos

            tribe_fit_pos = fits[tribe_start_pos:tribe_end_pos]
            tribe_fit_neg = fits[tribe_start_neg:tribe_end_neg]
            tribe_noise = noise_half[tribe_start_pos:tribe_end_pos]

            new_center, new_opt = es_update_one_tribe_jit(
                tribe_centers[t], tribe_noise, sigma_jnp,
                tribe_fit_pos, tribe_fit_neg, tribe_opt_states[t])
            new_tribe_centers.append(new_center)
            new_opt_states.append(new_opt)

        tribe_centers = jnp.stack(new_tribe_centers)
        tribe_opt_states = new_opt_states

        # Metrics
        mean_fit = float(jnp.mean(fits))
        max_fit = float(jnp.max(fits))
        min_fit = float(jnp.min(fits))
        pop_std = float(jnp.std(fits))
        tribe_means = [float(jnp.mean(fits[t * half_pop_per_tribe:(t + 1) * half_pop_per_tribe]))
                        for t in range(NUM_TRIBES)]
        for t in range(NUM_TRIBES):
            tribe_fit_history[t].append(tribe_means[t])

        # Simple sigma adapter: if std is growing, slightly shrink; if shrinking, grow
        # (skip detailed adaptive for v4 simplicity)
        if gen > 5 and gen % 10 == 0:
            recent_mean = np.mean([np.mean(h[-5:]) for h in tribe_fit_history if len(h) >= 5])
            older_mean = np.mean([np.mean(h[-10:-5]) for h in tribe_fit_history if len(h) >= 10])
            if recent_mean > older_mean + 0.5:
                current_sigma = max(NOISE_STD_MIN, current_sigma * 0.95)  # progress → narrow
            else:
                current_sigma = min(NOISE_STD_MAX, current_sigma * 1.03)  # plateau → broaden

        # --- PBT: periodic tribe tournament ---
        if gen > 0 and gen % TRIBE_TOURNAMENT_INTERVAL == 0:
            # Evaluate each tribe's recent average performance
            recent_perf = [np.mean(h[-5:]) if len(h) >= 5 else h[-1] for h in tribe_fit_history]
            best_tribe = int(np.argmax(recent_perf))
            worst_tribe = int(np.argmin(recent_perf))
            if best_tribe != worst_tribe:
                k_mut, key = random.split(key)
                mutation = random.normal(k_mut, (num_params,)) * TRIBE_MUTATION_SIGMA
                new_loser_center = tribe_centers[best_tribe] + mutation
                tribe_centers = tribe_centers.at[worst_tribe].set(new_loser_center)
                # Reset the replaced tribe's optimizer state
                tribe_opt_states[worst_tribe] = optimizer.init(new_loser_center)
                print(f"  >> PBT: replaced tribe {worst_tribe} (fit={recent_perf[worst_tribe]:.1f}) "
                      f"with perturbed copy of tribe {best_tribe} (fit={recent_perf[best_tribe]:.1f})")

        if gen > 0 and gen % 50 == 0:
            np.save(CKPT_PATH, np.array(jax.device_get(tribe_centers)))
            np.save(GEN_CKPT_PATH, {'gen': gen, 'mean_fit': mean_fit})
            print(f"  [Checkpoint saved at gen {gen}]")

        now = time.time()
        if now - last_print_time > PRINT_INTERVAL or gen == start_gen or gen % 10 == 0:
            elapsed = now - start_time
            print(f"\n--- GEN {gen} | {elapsed:.0f}s elapsed ---")
            print(f"  Aggregate fitness: mean={mean_fit:.2f} max={max_fit:.2f} "
                  f"min={min_fit:.2f} std={pop_std:.2f}")
            print(f"  Per-tribe mean: " +
                  ", ".join(f"t{t}={tribe_means[t]:.1f}" for t in range(NUM_TRIBES)))
            print(f"  Sigma: {current_sigma:.4f}")

            # Population-wide metrics
            metrics_np = np.array(jax.device_get(metrics_all))
            feast_avg = metrics_np[:, 0].mean()
            food_avg = metrics_np[:, 1].mean()
            move_avg = metrics_np[:, 2].mean()
            variety_avg = metrics_np[:, 3].mean()
            feast_adoption = (metrics_np[:, 0] > 0).mean() * 100
            print(f"  Pop avg: feast_energy={feast_avg:.1f}, food_energy={food_avg:.1f}, "
                  f"movement={move_avg:.1f}, variety={variety_avg:.1f}")
            print(f"  Feast adoption: {feast_adoption:.1f}% of members")

            # Render preview from best tribe
            best_tribe_now = int(np.argmax(tribe_means))
            k_render, key = random.split(key)
            preview_params = unflatten_params(tribe_centers[best_tribe_now], params_template)
            preview_tribe_stack = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None], (NUM_TRIBES,) + x.shape),
                preview_params)
            tribes = jnp.full((MAX_AGENTS,), best_tribe_now, dtype=jnp.int32)
            render_env = init_env(k_render, arena_h, arena_w, grid_size, num_active, tribes)

            def preview_step(carry, sk):
                state, last_speak = carry
                obs, audio = get_all_obs(state, grid_size, last_speak)
                proprio = build_proprioception(state)
                actions, new_hidden = sample_actions(
                    preview_tribe_stack, obs, audio, proprio, state.agent_hidden,
                    state.agent_tribe, sk, apply_fn, ACTION_TEMP)
                state = state._replace(agent_hidden=new_hidden)
                new_state = step_env(state, actions, grid_size)
                # Log per-tick data for analysis
                log = {
                    'speech': actions[:, 1],          # (MAX_AGENTS,)
                    'pos': new_state.agent_pos,       # (MAX_AGENTS, 2)
                    'alive': new_state.agent_alive,   # (MAX_AGENTS,)
                    'ate_food': (new_state.energy_earned > state.energy_earned),
                    'got_feast': (new_state.feast_bonus_earned > state.feast_bonus_earned),
                    'energy': new_state.agent_energy, # (MAX_AGENTS,)
                }
                return (new_state, actions[:, 1]), log

            preview_keys = random.split(k_render, 200)
            (mid_state, _), log = lax.scan(preview_step,
                (render_env, jnp.zeros(MAX_AGENTS, dtype=jnp.int32)),
                preview_keys)

            grid_np = np.array(jax.device_get(mid_state.grid))
            pos_np = np.array(jax.device_get(mid_state.agent_pos))
            alive_np = np.array(jax.device_get(mid_state.agent_alive))
            energy_np = np.array(jax.device_get(mid_state.agent_energy))
            tribe_np = np.array(jax.device_get(mid_state.agent_tribe))
            speech_np = np.array(jax.device_get(log['speech']))        # (T, MAX_AGENTS)
            pos_log = np.array(jax.device_get(log['pos']))              # (T, MAX_AGENTS, 2)
            alive_log = np.array(jax.device_get(log['alive']))          # (T, MAX_AGENTS)
            ate_log = np.array(jax.device_get(log['ate_food']))         # (T, MAX_AGENTS)
            feast_log = np.array(jax.device_get(log['got_feast']))      # (T, MAX_AGENTS)
            energy_log = np.array(jax.device_get(log['energy']))        # (T, MAX_AGENTS)

            print(f"  Preview from tribe {best_tribe_now}:")
            print(render_snapshot(grid_np, arena_h, arena_w, grid_size,
                                   pos_np, alive_np, energy_np, tribe_np))

            total_speech = int((speech_np > 0).sum())
            if total_speech > 0:
                uniq_tokens = np.unique(speech_np[speech_np > 0])
                uniq = len(uniq_tokens)
                # Histogram of token usage
                token_counts = np.zeros(27, dtype=np.int64)
                for t in range(1, 27):
                    token_counts[t] = int((speech_np == t).sum())
                top_tokens = np.argsort(-token_counts)[:5]
                top_str = ", ".join(
                    f"{chr(64 + int(t))}:{int(token_counts[t])}"
                    for t in top_tokens if token_counts[t] > 0)
                print(f"  Speech: {total_speech} utterances, {uniq} unique tokens")
                print(f"  Top tokens: {top_str}")

            # Per-agent transcript for one agent (pick the one that lived longest)
            lifetimes = alive_log.sum(axis=0)  # (MAX_AGENTS,)
            best_agent = int(np.argmax(lifetimes))
            if lifetimes[best_agent] > 20:
                print(f"  === Transcript for agent {best_agent} (alive {int(lifetimes[best_agent])} ticks) ===")
                # Show a 30-tick window around each eating event (at most 3 such windows)
                eat_ticks = np.where(ate_log[:, best_agent] | feast_log[:, best_agent])[0]
                if len(eat_ticks) == 0:
                    # No eating — show first 40 ticks as a boring baseline
                    windows = [(0, min(40, int(lifetimes[best_agent])))]
                else:
                    # Pick up to 2 eating events; window = [eat-15, eat+5]
                    windows = []
                    for et in eat_ticks[:2]:
                        w_start = max(0, int(et) - 15)
                        w_end = min(speech_np.shape[0], int(et) + 6)
                        windows.append((w_start, w_end))

                for (w_start, w_end) in windows:
                    print(f"    ticks {w_start}-{w_end}:")
                    for t in range(w_start, w_end):
                        if not alive_log[t, best_agent]:
                            continue
                        tok = int(speech_np[t, best_agent])
                        pos = pos_log[t, best_agent]
                        spoke = chr(64 + tok) if tok > 0 else '.'
                        flag = ''
                        if ate_log[t, best_agent]:
                            flag = ' EAT'
                        elif feast_log[t, best_agent]:
                            flag = ' FEAST'
                        e = energy_log[t, best_agent]
                        print(f"      t={t:3d} pos=({int(pos[0]):2d},{int(pos[1]):2d}) "
                              f"spoke={spoke} energy={float(e):5.1f}{flag}")

            # Micro-snapshot: at the tick of the FIRST feast event (if any), print
            # a 7x7 window around the feaster, plus what neighboring agents spoke.
            feast_any = feast_log.any(axis=1)
            if feast_any.any():
                first_feast_tick = int(np.argmax(feast_any))
                # Who feasted?
                feasters = np.where(feast_log[first_feast_tick])[0]
                if len(feasters) > 0:
                    f0 = int(feasters[0])
                    fp = pos_log[first_feast_tick, f0]
                    print(f"  === First feast at t={first_feast_tick} by agent {f0} at ({int(fp[0])},{int(fp[1])}) ===")
                    # 7x7 window around feast position
                    r0 = max(0, int(fp[0]) - 3)
                    r1 = min(grid_size, int(fp[0]) + 4)
                    c0 = max(0, int(fp[1]) - 3)
                    c1 = min(grid_size, int(fp[1]) + 4)
                    # Use the grid from that tick... but we only saved final grid.
                    # Skip the grid; just report positions of all agents within radius 6
                    # and what they said in the 5 ticks leading up to feast.
                    print(f"    Nearby agents (chebyshev <= 6) and their speech at t={first_feast_tick}:")
                    for i in range(MAX_AGENTS):
                        if not alive_log[first_feast_tick, i]:
                            continue
                        ipos = pos_log[first_feast_tick, i]
                        cheb = max(abs(int(ipos[0]) - int(fp[0])), abs(int(ipos[1]) - int(fp[1])))
                        if cheb > 6:
                            continue
                        tok = int(speech_np[first_feast_tick, i])
                        spoke = chr(64 + tok) if tok > 0 else '.'
                        # Also grab this agent's speech in ticks -5..-1
                        prev_speech = [
                            chr(64 + int(speech_np[first_feast_tick - k, i]))
                            if first_feast_tick - k >= 0 and speech_np[first_feast_tick - k, i] > 0
                            else '.'
                            for k in range(5, 0, -1)
                        ]
                        prev_str = "".join(prev_speech)
                        tag = '(FEASTER)' if i == f0 else ''
                        print(f"      a{i} pos=({int(ipos[0]):2d},{int(ipos[1]):2d}) "
                              f"cheb={cheb} prev5_speech=[{prev_str}] now={spoke} {tag}")

            last_print_time = now

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
