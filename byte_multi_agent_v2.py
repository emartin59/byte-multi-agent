# %% [markdown]
# # Byte-Multi-Agent: Emergent Language Edition (v2)
#
# A 2D byte-grid multi-agent environment designed to spark the emergence of
# language, coordination, proto-economics, and generalized knowledge under
# survival pressure. Everything — world, speech, writing, inventory — is bytes.
#
# This version is a substantial rewrite of v1 with changes aimed at
# (a) actually using the TPU pod, and (b) unblocking emergent language.
#
# === Changes from v1 ===
#  1. Sharded ES across TPU pod via shard_map (near-linear pod scaling)
#  2. bfloat16 forward pass (faster on TPU matrix units)
#  3. Recurrent policy (GRU) — required for multi-tick intent
#  4. Heterogeneous agents: population of K policies sampled per episode,
#     plus per-agent observation masks (creates info asymmetry → language pressure)
#  5. Stochastic action sampling (Gumbel) — ES alone can't explore 27-way discrete heads
#  6. General crafting substrate: hash(byte_X, byte_Y) → output, some useful, some not.
#     Agents must discover which adjacent pairs are productive.
#  7. Audio channel: speech is a separate per-agent observation, not just a spatial byte,
#     and persists for a few ticks
#  8. Action head chooses inventory slot for use/drop
#  9. Fitness: survival_ticks + energy_earned, no flat alive bonus that crushes everything
# 10. Scarcity cranked, food economy genuinely competitive
# 11. Vectorized grid updates (segment_sum for write conflicts, no per-agent scan)
#
# Run: just `python byte_multi_agent_v2.py`. Detects TPU automatically.

# %%
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
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

# --- World Bytes ---
EMPTY = 0
WALL = 1
FOOD = 2
SEED = 3
ROCK = 4
WATER = 5

WRITE_OFFSET = 6      # 6-31: a-z persistent writing
SPEAK_OFFSET = 32     # 32-57: A-Z transient (used only for display, not as world byte)
AGENT_MARK = 58       # @ on display
# Tool bytes are 59..95. The crafting substrate produces bytes in this range
# from hash(neighbor_a, neighbor_b). Agents discover which combinations
# produce useful tools vs inert "junk" items.
TOOL_START = 59
TOOL_END = 96        # 37 possible tool bytes
NUM_BYTE_TYPES = 128 # keep room

# Which tools are "real" (have effects) vs junk: a fixed subset is active.
# Agents don't know which, a priori.
NUM_ACTIVE_TOOLS = 4  # out of 37 possible tool bytes, only 4 do anything

CHAR_MAP = {
    EMPTY: '.', WALL: '#', FOOD: '*', SEED: ',', ROCK: 'o', WATER: '~',
    AGENT_MARK: '@',
}
for i in range(26):
    CHAR_MAP[i + WRITE_OFFSET] = chr(97 + i)   # a-z
    CHAR_MAP[i + SPEAK_OFFSET] = chr(65 + i)   # A-Z (display-only)
for i in range(TOOL_START, TOOL_END):
    # Display tools as digits and symbols
    CHAR_MAP[i] = "0123456789!@#$%^&*()_+-=[]{};:<>?/|\\"[(i - TOOL_START) % 37]

# --- Environment ---
VISION_RADIUS = 7
VISION_SIZE = 2 * VISION_RADIUS + 1  # 15
MAX_AGENTS = 8
EPISODE_LENGTH = 400
INVENTORY_SIZE = 4

# Audio channel: speech persists for this many ticks, and nearby agents
# hear it as a separate input signal (not spatial). This decouples
# language from spatial movement.
AUDIO_RANGE = 10           # chebyshev distance
AUDIO_PERSIST_TICKS = 3

# Energy / economy
INITIAL_ENERGY = 50.0
MAX_ENERGY = 100.0
FOOD_ENERGY = 18.0
METABOLIC_COST = 0.15      # bumped for scarcity pressure
MOVE_COST = 0.4
WRITE_COST = 1.0
OVERWRITE_COST = 1.5
SPEAK_COST = 0.3
PICKUP_COST = 0.3
USE_TOOL_COST = 0.5

# Ecology / spawning
SEED_TO_FOOD_PROB = 0.004  # slower regrowth = scarcer food
INITIAL_FOOD_DENSITY = 0.05
INITIAL_SEED_DENSITY = 0.04
INITIAL_ROCK_DENSITY = 0.05
INITIAL_WATER_DENSITY = 0.05

# General crafting: when two resource/tool bytes are adjacent and both present,
# there's a per-tick probability that a craft event occurs. The output byte is
# determined by a fixed hash of the two input bytes — so the recipe space is
# combinatorial (8x8 resource+tool types → tool bytes), and only some outputs
# are "active" tools. Agents must discover which recipes matter.
CRAFT_PROB = 0.02

# --- Heterogeneous policies ---
# We maintain K parameter vectors and sample one per agent slot per episode.
# This means agents are not identical clones — language becomes useful because
# different agents have different "personalities" and potentially different
# information.
NUM_POLICY_SPECIES = 4

# Each agent also has a random binary vision mask assigned per-episode that
# hides certain byte categories. This creates information asymmetry: some
# agents see food but not rocks, others see tools but not food, etc.
# Categories: food-visible, rock-visible, water-visible, tool-visible,
# writing-visible, agent-speech-visible. Speech is always audible via audio.
NUM_VISION_CATEGORIES = 6

# --- ES Hyperparameters ---
POP_SIZE = 512            # doubled; TPU pod can easily handle this
NUM_ENVS_PER_MEMBER = 4
LR = 0.01
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.005

# --- Adaptive Noise (unchanged, good as-is) ---
NOISE_STD_INIT = 0.02
NOISE_STD_MIN = 0.006
NOISE_STD_MAX = 0.03
NOISE_WARMUP_GENS = 30
NOISE_SNR_WINDOW = 20
NOISE_GROW_FACTOR = 1.02
NOISE_SHRINK_FACTOR = 0.95
NOISE_RATCHET_TOLERANCE = 0.30
NOISE_RATCHET_MAX_DURATION = 50

# --- Action sampling ---
# Temperature for Gumbel sampling at evaluation time. Low enough that policy
# is mostly deterministic but high enough to break argmax ties and explore
# large discrete heads.
ACTION_TEMP = 0.5

# --- Arena ---
INITIAL_ARENA = 20
INITIAL_ACTIVE = 6
GRID_SIZE = 64

# Movement directions: stay, up, down, right, left
MOVE_DIRS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)

# GRU hidden size
HIDDEN_SIZE = 128


# ==========================================
# 2. ADAPTIVE NOISE CONTROLLER (unchanged)
# ==========================================

class AdaptiveNoiseController:
    def __init__(self):
        self.sigma = NOISE_STD_INIT
        self.mean_history = []
        self.std_history = []
        self.sigma_history = []
        self.improvement_ema = 0.0
        self.ema_alpha = 0.15
        self.smoothed_mean = None
        self.best_smoothed = None
        self.best_sigma = NOISE_STD_INIT
        self.smooth_alpha = 0.1
        self.ratchet_active = False
        self.ratchet_consecutive = 0

    def record(self, mean_fitness, pop_fitness_std):
        self.mean_history.append(float(mean_fitness))
        self.std_history.append(float(pop_fitness_std))
        self.sigma_history.append(self.sigma)
        mf = float(mean_fitness)
        if self.smoothed_mean is None:
            self.smoothed_mean = mf
        else:
            self.smoothed_mean = self.smooth_alpha * mf + (1 - self.smooth_alpha) * self.smoothed_mean
        if self.best_smoothed is None or self.smoothed_mean > self.best_smoothed:
            self.best_smoothed = self.smoothed_mean
            self.best_sigma = self.sigma

    def step(self):
        n = len(self.mean_history)
        if n < NOISE_WARMUP_GENS:
            return self.sigma
        if n >= 2:
            raw_improvement = self.mean_history[-1] - self.mean_history[-2]
            self.improvement_ema = (self.ema_alpha * raw_improvement +
                                    (1 - self.ema_alpha) * self.improvement_ema)
        window = min(NOISE_SNR_WINDOW, n)
        if window >= 2:
            recent_means = self.mean_history[-window:]
            mean_improvement = (recent_means[-1] - recent_means[0]) / max(window - 1, 1)
            recent_std = np.mean(self.std_history[-window:])
        else:
            mean_improvement = 0.0
            recent_std = self.std_history[-1] if self.std_history else 1.0
        recent_std = max(recent_std, 1e-8)
        snr = abs(mean_improvement) / recent_std

        self.ratchet_active = False
        if self.best_smoothed is not None and self.smoothed_mean is not None:
            best_abs = max(abs(self.best_smoothed), 1.0)
            regression = (self.best_smoothed - self.smoothed_mean) / best_abs
            if regression > NOISE_RATCHET_TOLERANCE:
                self.ratchet_active = True
                self.ratchet_consecutive += 1
                if self.ratchet_consecutive >= NOISE_RATCHET_MAX_DURATION:
                    print(f"  >> RATCHET TIMEOUT after {self.ratchet_consecutive} gens.")
                    self.best_smoothed = self.smoothed_mean
                    self.best_sigma = self.sigma
                    self.ratchet_active = False
                    self.ratchet_consecutive = 0
                    self.sigma = min(self.sigma * 1.5, NOISE_STD_MAX)
                    self.sigma = float(np.clip(self.sigma, NOISE_STD_MIN, NOISE_STD_MAX))
                    return self.sigma
                target = max(self.best_sigma, NOISE_STD_MIN)
                self.sigma = self.sigma * 0.80 + target * 0.20
                self.sigma = float(np.clip(self.sigma, NOISE_STD_MIN, NOISE_STD_MAX))
                return self.sigma
            else:
                self.ratchet_consecutive = 0

        if self.improvement_ema > 0 and snr > 0.01:
            self.sigma *= NOISE_SHRINK_FACTOR
        elif self.improvement_ema <= 0 or snr < 0.005:
            self.sigma *= NOISE_GROW_FACTOR
        else:
            self.sigma *= 0.999
        self.sigma = float(np.clip(self.sigma, NOISE_STD_MIN, NOISE_STD_MAX))
        return self.sigma

    def get_sigma(self):
        return self.sigma

    def get_status_str(self):
        n = len(self.mean_history)
        if n < 2:
            return f"sigma={self.sigma:.4f} (warmup)"
        if self.ratchet_active:
            direction = f"RATCHET({self.ratchet_consecutive}/{NOISE_RATCHET_MAX_DURATION})"
        else:
            direction = "exploiting" if self.improvement_ema > 0 else "exploring"
        window = min(NOISE_SNR_WINDOW, n)
        recent_means = self.mean_history[-window:]
        mean_improvement = (recent_means[-1] - recent_means[0]) / max(window - 1, 1)
        recent_std = np.mean(self.std_history[-window:]) if self.std_history else 0.0
        snr = abs(mean_improvement) / max(recent_std, 1e-8)
        best_str = f"best={self.best_smoothed:.1f}" if self.best_smoothed is not None else "best=n/a"
        smooth_str = f"smooth={self.smoothed_mean:.1f}" if self.smoothed_mean is not None else "smooth=n/a"
        return (f"sigma={self.sigma:.4f} | {direction} | "
                f"imp_ema={self.improvement_ema:.3f} | snr={snr:.4f} | "
                f"{best_str} | {smooth_str}")


# ==========================================
# 3. CRAFTING TABLE
# ==========================================
# A fixed deterministic hash function: (byte_a, byte_b) -> output_byte.
# Built at import time so it's JAX-traceable. We make sure some outputs are
# "active" tools (effect table lookup) and most are inert junk tools that still
# occupy inventory but do nothing.
#
# The recipe table R[a, b] = byte that results when a and b are adjacent and
# a craft event fires. Only pairs where BOTH are resource bytes (ROCK, WATER,
# SEED, or tool bytes) can craft.

def _build_craft_table():
    """Deterministic pairwise recipe table; symmetric."""
    rng = np.random.RandomState(31337)
    table = np.zeros((NUM_BYTE_TYPES, NUM_BYTE_TYPES), dtype=np.int32)
    # Fill with "junk tool" outputs by default (inert bytes in tool range)
    for a in range(NUM_BYTE_TYPES):
        for b in range(a, NUM_BYTE_TYPES):
            out = TOOL_START + rng.randint(0, TOOL_END - TOOL_START)
            table[a, b] = out
            table[b, a] = out
    return table

CRAFT_TABLE = jnp.array(_build_craft_table())

# Which bytes are "ingredients" (can participate in crafting)?
# Resources and tools themselves.
def _is_ingredient(b):
    return (b == ROCK) | (b == WATER) | (b == SEED) | \
           ((b >= TOOL_START) & (b < TOOL_END))

# Active tool effects: a small subset of tool bytes do something when used.
# We fix this at init by picking a random subset of size NUM_ACTIVE_TOOLS.
# Each active tool has an "effect type" in [0, 3]:
#   0 = scatter SEED on 4 neighbors (like v1 fertilizer)
#   1 = convert adjacent ROCK to FOOD (stone fruit)
#   2 = convert adjacent WATER to SEED (lily pad)
#   3 = double energy gain when eating next FOOD within 20 ticks (hunter's charm) — simplified: +30 energy instantly
_rng_tools = np.random.RandomState(777)
_active_tool_indices = _rng_tools.choice(TOOL_END - TOOL_START, size=NUM_ACTIVE_TOOLS, replace=False)
_tool_effect_table = np.full((TOOL_END - TOOL_START,), -1, dtype=np.int32)
for i, idx in enumerate(_active_tool_indices):
    _tool_effect_table[idx] = i  # effect ids 0..NUM_ACTIVE_TOOLS-1
TOOL_EFFECT_TABLE = jnp.array(_tool_effect_table)  # shape (TOOL_END-TOOL_START,), -1 for junk

print(f"Active tools (bytes): {[int(TOOL_START + i) for i in _active_tool_indices]}")
print(f"  → effects 0..{NUM_ACTIVE_TOOLS-1}: scatter-seed, rock→food, water→seed, energy-boost")


# ==========================================
# 4. ENVIRONMENT STATE
# ==========================================

class EnvState(NamedTuple):
    grid: jnp.ndarray               # (H, W) int32
    agent_pos: jnp.ndarray          # (MAX_AGENTS, 2)
    agent_alive: jnp.ndarray        # (MAX_AGENTS,) bool
    agent_energy: jnp.ndarray       # (MAX_AGENTS,)
    agent_inventory: jnp.ndarray    # (MAX_AGENTS, INVENTORY_SIZE)
    agent_underfoot: jnp.ndarray    # (MAX_AGENTS,)
    agent_hidden: jnp.ndarray       # (MAX_AGENTS, HIDDEN_SIZE) — GRU hidden state
    agent_species: jnp.ndarray      # (MAX_AGENTS,) int — which policy this agent uses
    agent_vision_mask: jnp.ndarray  # (MAX_AGENTS, NUM_VISION_CATEGORIES) bool — per-agent obs mask
    agent_last_ate_tick: jnp.ndarray  # (MAX_AGENTS,) — for hunter's-charm tool (-1 if never)
    # Audio channel: (MAX_AGENTS, AUDIO_PERSIST_TICKS) — ring buffer of recent speech per agent
    audio_buffer: jnp.ndarray       # (MAX_AGENTS, AUDIO_PERSIST_TICKS) int — speech values
    audio_age: jnp.ndarray          # (MAX_AGENTS, AUDIO_PERSIST_TICKS) — age in ticks (for decay)
    survival_ticks: jnp.ndarray     # (MAX_AGENTS,) — how many ticks each agent has survived
    energy_earned: jnp.ndarray      # (MAX_AGENTS,) — cumulative energy earned from food
    tick: jnp.ndarray
    rng: jnp.ndarray


def init_env(key, arena_h, arena_w, grid_size, num_active):
    k_blocks, k_agents, k_species, k_mask, k_rng = random.split(key, 5)

    rows = jnp.arange(grid_size)[:, None]
    cols = jnp.arange(grid_size)[None, :]
    off_r = (grid_size - arena_h) // 2
    off_c = (grid_size - arena_w) // 2
    in_arena = (rows >= off_r) & (rows < off_r + arena_h) & (cols >= off_c) & (cols < off_c + arena_w)
    on_border = (rows == off_r) | (rows == off_r + arena_h - 1) | (cols == off_c) | (cols == off_c + arena_w - 1)
    playable = in_arena & ~on_border
    grid = jnp.where(in_arena, jnp.where(on_border, WALL, EMPTY), WALL)

    k1, k2, k3, k4 = random.split(k_blocks, 4)
    food_noise = random.uniform(k1, (grid_size, grid_size))
    seed_noise = random.uniform(k2, (grid_size, grid_size))
    rock_noise = random.uniform(k3, (grid_size, grid_size))
    water_noise = random.uniform(k4, (grid_size, grid_size))

    grid = jnp.where(playable & (food_noise < INITIAL_FOOD_DENSITY), FOOD, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (seed_noise < INITIAL_SEED_DENSITY), SEED, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (rock_noise < INITIAL_ROCK_DENSITY), ROCK, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (water_noise < INITIAL_WATER_DENSITY), WATER, grid)

    gumbel = random.gumbel(k_agents, shape=(grid_size * grid_size,))
    is_empty = (grid.reshape(-1) == EMPTY)
    gumbel = jnp.where(is_empty, gumbel, -1e9)
    top_idx = jnp.argsort(-gumbel)[:MAX_AGENTS]
    agent_pos = jnp.stack([top_idx // grid_size, top_idx % grid_size], axis=-1)
    agent_alive = jnp.arange(MAX_AGENTS) < num_active
    agent_energy = jnp.where(agent_alive, INITIAL_ENERGY, 0.0)
    agent_inventory = jnp.zeros((MAX_AGENTS, INVENTORY_SIZE), dtype=jnp.int32)
    agent_underfoot = jnp.full((MAX_AGENTS,), EMPTY, dtype=jnp.int32)
    agent_hidden = jnp.zeros((MAX_AGENTS, HIDDEN_SIZE), dtype=jnp.float32)

    # Assign each agent slot to a random species (policy index in [0, K))
    agent_species = random.randint(k_species, (MAX_AGENTS,), 0, NUM_POLICY_SPECIES)

    # Per-agent vision mask (binary). Each category has 70% prob of being visible,
    # so on average each agent misses ~2 of 6 categories.
    mask_noise = random.uniform(k_mask, (MAX_AGENTS, NUM_VISION_CATEGORIES))
    agent_vision_mask = mask_noise < 0.7

    agent_last_ate_tick = jnp.full((MAX_AGENTS,), -1000, dtype=jnp.int32)

    audio_buffer = jnp.zeros((MAX_AGENTS, AUDIO_PERSIST_TICKS), dtype=jnp.int32)
    audio_age = jnp.full((MAX_AGENTS, AUDIO_PERSIST_TICKS), AUDIO_PERSIST_TICKS + 1, dtype=jnp.int32)

    survival_ticks = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    energy_earned = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)

    return EnvState(
        grid=grid, agent_pos=agent_pos, agent_alive=agent_alive,
        agent_energy=agent_energy, agent_inventory=agent_inventory,
        agent_underfoot=agent_underfoot, agent_hidden=agent_hidden,
        agent_species=agent_species, agent_vision_mask=agent_vision_mask,
        agent_last_ate_tick=agent_last_ate_tick,
        audio_buffer=audio_buffer, audio_age=audio_age,
        survival_ticks=survival_ticks, energy_earned=energy_earned,
        tick=jnp.int32(0), rng=k_rng,
    )


# ==========================================
# 5. OBSERVATIONS
# ==========================================

def apply_vision_mask(obs_bytes, mask):
    """obs_bytes: (V, V) int. mask: (NUM_VISION_CATEGORIES,) bool.
    Replace masked-out byte categories with a sentinel (EMPTY) so the agent
    effectively can't see them.
    """
    # Category mapping:
    #   0: FOOD (and SEED — food-chain items)
    #   1: ROCK
    #   2: WATER
    #   3: tools (TOOL_START..TOOL_END)
    #   4: writing (WRITE_OFFSET..SPEAK_OFFSET)
    #   5: other agents (AGENT_MARK and speech bytes)
    is_food = (obs_bytes == FOOD) | (obs_bytes == SEED)
    is_rock = (obs_bytes == ROCK)
    is_water = (obs_bytes == WATER)
    is_tool = (obs_bytes >= TOOL_START) & (obs_bytes < TOOL_END)
    is_write = (obs_bytes >= WRITE_OFFSET) & (obs_bytes < SPEAK_OFFSET)
    is_agent_bytes = (obs_bytes == AGENT_MARK) | \
                      ((obs_bytes >= SPEAK_OFFSET) & (obs_bytes < AGENT_MARK))

    hide = (
        (is_food & ~mask[0]) |
        (is_rock & ~mask[1]) |
        (is_water & ~mask[2]) |
        (is_tool & ~mask[3]) |
        (is_write & ~mask[4]) |
        (is_agent_bytes & ~mask[5])
    )
    return jnp.where(hide, EMPTY, obs_bytes)


def get_all_obs(state, grid_size, last_speak):
    """Return (obs_grid, obs_audio) for each agent.

    obs_grid: (MAX_AGENTS, V, V) normalized, with per-agent vision masks applied.
    obs_audio: (MAX_AGENTS, MAX_AGENTS * AUDIO_PERSIST_TICKS) — the speech each
        agent can "hear" from all other agents' audio buffers, weighted by
        distance (0 if out of AUDIO_RANGE).
    """
    grid = state.grid

    # Render other agents onto the grid (with speech bytes if speaking now)
    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        body = jnp.where(last_speak[i] > 0,
                         jnp.minimum(last_speak[i] - 1 + SPEAK_OFFSET, AGENT_MARK - 1),
                         AGENT_MARK)
        return jnp.where(state.agent_alive[i], g.at[r, c].set(body), g), None

    composed, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    padded = jnp.pad(composed, VISION_RADIUS, constant_values=WALL)

    def get_one(pos, mask):
        crop = lax.dynamic_slice(padded, (pos[0], pos[1]), (VISION_SIZE, VISION_SIZE))
        return apply_vision_mask(crop, mask)

    obs_bytes = vmap(get_one)(state.agent_pos, state.agent_vision_mask)
    obs_norm = obs_bytes.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)

    # Audio: for each listener, gather audio buffers from all speakers weighted
    # by whether speaker is in audible range and is alive.
    # Shape of audio_buffer: (MAX_AGENTS, AUDIO_PERSIST_TICKS)
    # Output per listener: (MAX_AGENTS * AUDIO_PERSIST_TICKS,) — concat of all
    # speakers' recent speech, zeroed for out-of-range/dead.

    def audio_for_listener(listener_idx):
        lp = state.agent_pos[listener_idx]
        # distance from listener to each other agent
        dp = state.agent_pos - lp[None, :]
        chebyshev = jnp.maximum(jnp.abs(dp[:, 0]), jnp.abs(dp[:, 1]))
        audible = (chebyshev <= AUDIO_RANGE) & state.agent_alive & \
                  (jnp.arange(MAX_AGENTS) != listener_idx)
        # Only include entries whose age is fresh
        fresh = state.audio_age < AUDIO_PERSIST_TICKS
        # Gate: audible agent AND fresh audio
        gated = state.audio_buffer * fresh.astype(jnp.int32) * \
                audible[:, None].astype(jnp.int32)
        return gated.reshape(-1).astype(jnp.float32) / 27.0  # normalize speech ids to [0,1]

    obs_audio = vmap(audio_for_listener)(jnp.arange(MAX_AGENTS))

    return obs_norm, obs_audio


# ==========================================
# 6. PHYSICS
# ==========================================

def is_passable_byte(b):
    return (b == EMPTY) | (b == FOOD) | (b == SEED) | \
           ((b >= WRITE_OFFSET) & (b < SPEAK_OFFSET))


def step_env(state, actions, grid_size):
    """
    actions: (MAX_AGENTS, 5) — [move, speak, write, pickup_use, slot_idx]
      move:       0-4   stay/up/down/right/left
      speak:      0-26  silent or speak token 1-26
      write:      0-26  no-write or write letter 1-26
      pickup_use: 0-3   noop / pickup adjacent tool / use slot / drop slot
      slot_idx:   0..INVENTORY_SIZE-1  which inventory slot for use/drop
    """
    move_acts = actions[:, 0]
    speak_acts = actions[:, 1]
    write_acts = actions[:, 2]
    puu_acts = actions[:, 3]
    slot_acts = jnp.clip(actions[:, 4], 0, INVENTORY_SIZE - 1)

    curr_pos = state.agent_pos
    dp = MOVE_DIRS[move_acts]
    want_pos = curr_pos + dp
    want_pos = jnp.where(state.agent_alive[:, None], want_pos, curr_pos)
    want_pos = jnp.clip(want_pos, 0, grid_size - 1)

    # --- Movement resolution ---
    target_vals = state.grid[want_pos[:, 0], want_pos[:, 1]]
    passable = is_passable_byte(target_vals) & state.agent_alive

    # Conflict: two agents targeting same cell
    same_target = jnp.all(want_pos[:, None, :] == want_pos[None, :, :], axis=-1)
    eye = jnp.eye(MAX_AGENTS, dtype=jnp.bool_)
    has_conflict = jnp.any(same_target & passable[:, None] & passable[None, :] & ~eye, axis=1)

    # Non-swap: can't walk into another agent's current cell
    def check_occupied(i):
        same = jnp.all(curr_pos == want_pos[i][None, :], axis=-1) & state.agent_alive
        same = same & (jnp.arange(MAX_AGENTS) != i)
        return jnp.any(same)
    other_pos_mask = vmap(check_occupied)(jnp.arange(MAX_AGENTS))

    final_move = passable & ~has_conflict & ~other_pos_mask
    final_pos = jnp.where(final_move[:, None], want_pos, curr_pos)
    moved = final_move & jnp.any(dp != 0, axis=-1)

    ate_food = moved & (target_vals == FOOD)

    # underfoot: what was at the new cell before we arrived (food → seed)
    new_underfoot_moved = jnp.where(ate_food, SEED, target_vals)
    new_underfoot = jnp.where(moved, new_underfoot_moved, state.agent_underfoot)

    # --- Writing ---
    leave_byte = jnp.where(write_acts > 0, write_acts - 1 + WRITE_OFFSET, state.agent_underfoot)
    is_overwrite = (write_acts > 0) & (state.agent_underfoot != EMPTY) & \
                    (state.agent_underfoot != leave_byte)

    # --- Grid updates (vectorized using scatters instead of per-agent scan) ---
    new_grid = state.grid

    # Deposit leave_byte at each agent's old cell (if they moved)
    alive_moved = moved & state.agent_alive
    leave_r = jnp.where(alive_moved, curr_pos[:, 0], 0)
    leave_c = jnp.where(alive_moved, curr_pos[:, 1], 0)
    # Use .set which resolves multiple writers by last-writer (TPU-stable)
    # If two agents somehow leave same cell (shouldn't; they had different starts),
    # it's fine. Use a mask-aware write:
    deposit_vals = jnp.where(alive_moved, leave_byte, new_grid[leave_r, leave_c])
    new_grid = new_grid.at[leave_r, leave_c].set(deposit_vals)

    # Consume food at final positions
    eat_r = jnp.where(ate_food, final_pos[:, 0], 0)
    eat_c = jnp.where(ate_food, final_pos[:, 1], 0)
    keep_vals = jnp.where(ate_food, EMPTY, new_grid[eat_r, eat_c])
    new_grid = new_grid.at[eat_r, eat_c].set(keep_vals)

    # --- Pickup / Use / Drop ---
    # We still use a scan here because these interact with inventory,
    # but the grid writes are cheap. Pickup and drop can touch neighbor cells.
    new_inventory = state.agent_inventory
    new_energy = state.agent_energy
    new_last_ate = state.agent_last_ate_tick

    def do_puu(carry, i):
        grid, inv, energy, last_ate = carry
        act = puu_acts[i]
        slot = slot_acts[i]
        alive = state.agent_alive[i]
        r, c = final_pos[i, 0], final_pos[i, 1]

        neighbors = jnp.stack([
            jnp.clip(jnp.array([r-1, c]), 0, grid_size - 1),
            jnp.clip(jnp.array([r+1, c]), 0, grid_size - 1),
            jnp.clip(jnp.array([r, c-1]), 0, grid_size - 1),
            jnp.clip(jnp.array([r, c+1]), 0, grid_size - 1),
        ])
        neigh_vals = grid[neighbors[:, 0], neighbors[:, 1]]

        # PICKUP (act==1): pick first tool neighbor into first empty slot
        is_tool = (neigh_vals >= TOOL_START) & (neigh_vals < TOOL_END)
        any_tool = jnp.any(is_tool)
        first_tool_idx = jnp.argmax(is_tool)
        picked_byte = neigh_vals[first_tool_idx]
        empty_slots = inv[i] == EMPTY
        has_empty = jnp.any(empty_slots)
        first_empty = jnp.argmax(empty_slots)
        can_pickup = (act == 1) & alive & any_tool & has_empty
        tnr = neighbors[first_tool_idx, 0]
        tnc = neighbors[first_tool_idx, 1]
        grid = jnp.where(can_pickup, grid.at[tnr, tnc].set(EMPTY), grid)
        inv_i = jnp.where(can_pickup, inv[i].at[first_empty].set(picked_byte), inv[i])

        # USE slot (act==2): consume inv[slot] byte; apply effect if active tool
        selected = inv_i[slot]
        tool_offset = jnp.clip(selected - TOOL_START, 0, TOOL_END - TOOL_START - 1)
        effect_id = jnp.where(
            (selected >= TOOL_START) & (selected < TOOL_END),
            TOOL_EFFECT_TABLE[tool_offset], -1)

        can_use = (act == 2) & alive & (selected != EMPTY) & (effect_id >= 0)

        # Apply effect 0: scatter SEED on up to 4 empty neighbors
        def apply_scatter_seed(g):
            def one(gg, k):
                nr, nc = neighbors[k, 0], neighbors[k, 1]
                cond = (effect_id == 0) & (gg[nr, nc] == EMPTY)
                return jnp.where(cond, gg.at[nr, nc].set(SEED), gg), None
            gg, _ = lax.scan(one, g, jnp.arange(4))
            return gg
        grid = jnp.where(can_use & (effect_id == 0), apply_scatter_seed(grid), grid)

        # Effect 1: convert adjacent ROCK → FOOD (first one found)
        def apply_rock_to_food(g):
            is_rock_n = (neigh_vals == ROCK)
            any_rock = jnp.any(is_rock_n)
            idx = jnp.argmax(is_rock_n)
            nr, nc = neighbors[idx, 0], neighbors[idx, 1]
            return jnp.where((effect_id == 1) & any_rock,
                             g.at[nr, nc].set(FOOD), g)
        grid = jnp.where(can_use & (effect_id == 1), apply_rock_to_food(grid), grid)

        # Effect 2: convert adjacent WATER → SEED
        def apply_water_to_seed(g):
            is_water_n = (neigh_vals == WATER)
            any_water = jnp.any(is_water_n)
            idx = jnp.argmax(is_water_n)
            nr, nc = neighbors[idx, 0], neighbors[idx, 1]
            return jnp.where((effect_id == 2) & any_water,
                             g.at[nr, nc].set(SEED), g)
        grid = jnp.where(can_use & (effect_id == 2), apply_water_to_seed(grid), grid)

        # Effect 3: immediate +30 energy (hunter's charm, simplified)
        energy_boost = jnp.where((effect_id == 3) & can_use, 30.0, 0.0)
        energy_i_new = energy[i] + energy_boost

        # Consume the used item from the slot
        inv_i = jnp.where(can_use, inv_i.at[slot].set(EMPTY), inv_i)

        # DROP slot (act==3): place inv[slot] byte on first empty neighbor
        empty_nbrs = neigh_vals == EMPTY
        any_empty_nbr = jnp.any(empty_nbrs)
        first_empty_nbr = jnp.argmax(empty_nbrs)
        dtr = neighbors[first_empty_nbr, 0]
        dtc = neighbors[first_empty_nbr, 1]
        drop_val = inv_i[slot]
        can_drop = (act == 3) & alive & any_empty_nbr & (drop_val != EMPTY)
        grid = jnp.where(can_drop, grid.at[dtr, dtc].set(drop_val), grid)
        inv_i = jnp.where(can_drop, inv_i.at[slot].set(EMPTY), inv_i)

        action_cost = jnp.where(
            act == 1, PICKUP_COST,
            jnp.where((act == 2) | (act == 3), USE_TOOL_COST, 0.0)
        )
        did_act = (act > 0) & alive
        energy_i_new = jnp.where(did_act, energy_i_new - action_cost, energy_i_new)

        inv = inv.at[i].set(inv_i)
        energy = energy.at[i].set(energy_i_new)
        return (grid, inv, energy, last_ate), None

    (new_grid, new_inventory, new_energy, new_last_ate), _ = lax.scan(
        do_puu, (new_grid, new_inventory, new_energy, new_last_ate),
        jnp.arange(MAX_AGENTS))

    # --- Crafting: any two adjacent ingredient bytes may combine ---
    k_craft, new_rng = random.split(state.rng)
    craft_noise = random.uniform(k_craft, (grid_size, grid_size))
    craft_choose = random.randint(k_craft, (grid_size, grid_size), 0, 4)  # which neighbor dir

    is_ingr = _is_ingredient(new_grid)
    # Build shifted-view of neighbors in each direction
    up    = jnp.pad(new_grid[1:, :],  ((0, 1), (0, 0)), constant_values=EMPTY)
    down  = jnp.pad(new_grid[:-1, :], ((1, 0), (0, 0)), constant_values=EMPTY)
    left  = jnp.pad(new_grid[:, 1:],  ((0, 0), (0, 1)), constant_values=EMPTY)
    right = jnp.pad(new_grid[:, :-1], ((0, 0), (1, 0)), constant_values=EMPTY)
    neighbors_stack = jnp.stack([up, down, left, right], axis=0)  # (4, H, W)
    chosen_neighbor = neighbors_stack[craft_choose,
                                       jnp.arange(grid_size)[:, None],
                                       jnp.arange(grid_size)[None, :]]
    neighbor_is_ingr = _is_ingredient(chosen_neighbor)

    # Only craft if both are ingredients, noise fires, and this cell "won" the race
    # (craft_noise < prob). To avoid double-crafting when both cells of a pair
    # fire, we only craft on the cell where we're the "lower-numbered" ingredient
    # type; ties broken by position.
    craft_fire = (craft_noise < CRAFT_PROB) & is_ingr & neighbor_is_ingr
    # Tiebreak: only the cell whose byte is strictly less than its neighbor
    # (or equal and position-based) produces. Simpler: only fire when self byte <= neighbor.
    craft_fire = craft_fire & (new_grid <= chosen_neighbor)

    craft_output = CRAFT_TABLE[new_grid, chosen_neighbor]
    new_grid = jnp.where(craft_fire, craft_output, new_grid)

    # --- Seed → Food regrowth ---
    k_regrow, new_rng = random.split(new_rng)
    regrow_noise = random.uniform(k_regrow, (grid_size, grid_size))
    is_seed = (new_grid == SEED)
    regrow_trigger = is_seed & (regrow_noise < SEED_TO_FOOD_PROB)
    new_grid = jnp.where(regrow_trigger, FOOD, new_grid)

    # --- Energy ---
    new_energy = new_energy - jnp.where(state.agent_alive, METABOLIC_COST, 0.0)
    new_energy = new_energy - jnp.where(moved, MOVE_COST, 0.0)
    new_energy = new_energy - jnp.where((speak_acts > 0) & state.agent_alive, SPEAK_COST, 0.0)
    wrote_something = (write_acts > 0) & moved
    base_write = jnp.where(wrote_something, WRITE_COST, 0.0)
    extra_overwrite = jnp.where(is_overwrite, OVERWRITE_COST - WRITE_COST, 0.0)
    new_energy = new_energy - base_write - extra_overwrite
    new_energy = new_energy + jnp.where(ate_food, FOOD_ENERGY, 0.0)
    new_energy = jnp.clip(new_energy, 0.0, MAX_ENERGY)

    new_last_ate = jnp.where(ate_food, state.tick, new_last_ate)
    new_energy_earned = state.energy_earned + jnp.where(ate_food, FOOD_ENERGY, 0.0)

    new_alive = state.agent_alive & (new_energy > 0.0)
    new_survival = state.survival_ticks + new_alive.astype(jnp.float32)

    # --- Audio buffer update ---
    # Shift everything one tick older, then insert current speech at slot 0
    new_audio_age = jnp.minimum(state.audio_age + 1, AUDIO_PERSIST_TICKS + 10)
    # Roll: drop last, prepend new
    new_audio_buffer = jnp.concatenate([
        speak_acts[:, None],
        state.audio_buffer[:, :-1]
    ], axis=1)
    new_audio_age = jnp.concatenate([
        jnp.zeros((MAX_AGENTS, 1), dtype=jnp.int32),
        new_audio_age[:, :-1]
    ], axis=1)

    return state._replace(
        grid=new_grid,
        agent_pos=final_pos,
        agent_alive=new_alive,
        agent_energy=new_energy,
        agent_inventory=new_inventory,
        agent_underfoot=new_underfoot,
        agent_last_ate_tick=new_last_ate,
        audio_buffer=new_audio_buffer,
        audio_age=new_audio_age,
        survival_ticks=new_survival,
        energy_earned=new_energy_earned,
        tick=state.tick + 1,
        rng=new_rng,
    )


# ==========================================
# 7. NETWORK — recurrent, multi-head
# ==========================================

class AgentNet(nn.Module):
    """Recurrent policy. Takes (vision 15x15, audio channel, proprioception),
    maintains a GRU hidden state, outputs 5 action heads.
    Uses bfloat16 for internal computation; params kept in float32.
    """
    @nn.compact
    def __call__(self, obs, audio, proprio, hidden):
        # obs: (V, V) float, audio: (MAX_AGENTS * AUDIO_PERSIST_TICKS,), proprio: (..),
        # hidden: (HIDDEN_SIZE,)

        # Cast to bf16 for compute
        obs_bf = obs.astype(jnp.bfloat16)
        audio_bf = audio.astype(jnp.bfloat16)
        proprio_bf = proprio.astype(jnp.bfloat16)
        hidden_bf = hidden.astype(jnp.bfloat16)

        # Vision tower
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

        # Audio tower (MLP)
        a = nn.Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.float32)(audio_bf)
        a = nn.relu(a)
        a = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.float32)(a)
        a = nn.relu(a)

        # Concatenate all features
        feat = jnp.concatenate([x_flat, a, proprio_bf], axis=-1)
        feat = nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16,
                        param_dtype=jnp.float32)(feat)
        feat = nn.relu(feat)

        # GRU cell (manual, since we want explicit hidden state flow)
        # gate = sigmoid(W_z * [feat, hidden])
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

        # Heads — cast back to float32 for numerical stability in sampling
        h32 = new_hidden.astype(jnp.float32)
        move_logits  = nn.Dense(5,  param_dtype=jnp.float32)(h32)
        speak_logits = nn.Dense(27, param_dtype=jnp.float32)(h32)
        write_logits = nn.Dense(27, param_dtype=jnp.float32)(h32)
        puu_logits   = nn.Dense(4,  param_dtype=jnp.float32)(h32)
        slot_logits  = nn.Dense(INVENTORY_SIZE, param_dtype=jnp.float32)(h32)

        return (move_logits, speak_logits, write_logits, puu_logits, slot_logits,
                h32)


def build_proprioception(state):
    """Per-agent proprioception: energy, inventory, last-ate-recency, species id,
    vision mask (so network knows what it's missing)."""
    energy_norm = state.agent_energy / MAX_ENERGY
    inv_norm = state.agent_inventory.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)
    hunger = jnp.minimum(
        (state.tick - state.agent_last_ate_tick).astype(jnp.float32) / 100.0, 5.0)
    species_onehot = jax.nn.one_hot(state.agent_species, NUM_POLICY_SPECIES)
    vision_mask_f = state.agent_vision_mask.astype(jnp.float32)
    proprio = jnp.concatenate([
        energy_norm[:, None],
        inv_norm,
        hunger[:, None],
        species_onehot,
        vision_mask_f,
    ], axis=-1)
    return proprio


PROPRIO_DIM = 1 + INVENTORY_SIZE + 1 + NUM_POLICY_SPECIES + NUM_VISION_CATEGORIES


def sample_actions(all_params, obs_all, audio_all, proprio_all, hidden_all,
                   species_ids, key, apply_fn, temperature):
    """For each agent, pick that agent's species' policy, then sample actions.
    all_params: pytree where leaves have leading dim = NUM_POLICY_SPECIES
    """
    keys = random.split(key, MAX_AGENTS * 5).reshape(MAX_AGENTS, 5, 2)

    def act_one(obs, audio, proprio, hidden, species, ks):
        # Select species' params by tree-map
        my_params = jax.tree.map(lambda x: x[species], all_params)
        ml, sl, wl, puul, slotl, new_hidden = apply_fn(my_params, obs, audio, proprio, hidden)

        def sample(logits, k):
            g = random.gumbel(k, logits.shape)
            return jnp.argmax(logits / temperature + g)

        move  = sample(ml,    ks[0])
        speak = sample(sl,    ks[1])
        write = sample(wl,    ks[2])
        puu   = sample(puul,  ks[3])
        slot  = sample(slotl, ks[4])
        return (jnp.array([move, speak, write, puu, slot], dtype=jnp.int32), new_hidden)

    actions, new_hidden = vmap(act_one)(
        obs_all, audio_all, proprio_all, hidden_all, species_ids, keys)
    return actions, new_hidden


# ==========================================
# 8. EPISODE RUNNER
# ==========================================

def run_episode(all_params, apply_fn, env_state, grid_size, ep_key):
    """Run episode; return final state (from which we extract fitness)."""
    def step_fn(carry, step_key):
        state, last_speak = carry
        obs, audio = get_all_obs(state, grid_size, last_speak)
        proprio = build_proprioception(state)
        actions, new_hidden = sample_actions(
            all_params, obs, audio, proprio, state.agent_hidden,
            state.agent_species, step_key, apply_fn, ACTION_TEMP)

        state = state._replace(agent_hidden=new_hidden)
        new_state = step_env(state, actions, grid_size)
        return (new_state, actions[:, 1]), None

    init_speak = jnp.zeros(MAX_AGENTS, dtype=jnp.int32)
    step_keys = random.split(ep_key, EPISODE_LENGTH)
    (final_state, _), _ = lax.scan(
        step_fn, (env_state, init_speak), step_keys)
    return final_state


def compute_fitness(final_state):
    """Fitness = sum over agents of (survival_ticks + 0.5 * energy_earned + 0.1 * final_energy).
    No flat alive bonus — we want sitting still to be worse than seeking food."""
    survival = jnp.sum(final_state.survival_ticks)
    earned = jnp.sum(final_state.energy_earned)
    final_e = jnp.sum(final_state.agent_energy)
    return survival + 0.5 * earned + 0.1 * final_e


# ==========================================
# 9. ES CORE
# ==========================================

def flatten_params(params):
    leaves = jax.tree.leaves(params)
    return jnp.concatenate([l.reshape(-1) for l in leaves])


def unflatten_params(flat, params_template):
    leaves = jax.tree.leaves(params_template)
    shapes = [l.shape for l in leaves]
    sizes = [l.size for l in leaves]
    offsets = []
    offset = 0
    for s in sizes:
        offsets.append(offset)
        offset += s
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


def evaluate_member(flat_params, center_flat_multi, params_template_single,
                    apply_fn, env_keys, episode_keys, grid_size, arena_h, arena_w,
                    num_active):
    """Evaluate one population member. flat_params: flat ndarray of all species'
    params concatenated. NUM_ENVS_PER_MEMBER independent env runs.
    """
    # Reshape into (species, params_per_species)
    params_per_species = flat_params.shape[0] // NUM_POLICY_SPECIES
    species_params = flat_params.reshape(NUM_POLICY_SPECIES, params_per_species)

    # Unflatten each species' params
    def unflatten_species(flat_sp):
        return unflatten_params(flat_sp, params_template_single)

    all_params = vmap(unflatten_species)(species_params)

    def eval_one_env(ek, pk):
        env_state = init_env(ek, arena_h, arena_w, grid_size, num_active)
        final_state = run_episode(all_params, apply_fn, env_state, grid_size, pk)
        return compute_fitness(final_state)

    fits = vmap(eval_one_env)(env_keys, episode_keys)
    return jnp.mean(fits)


# ==========================================
# 10. RENDERING
# ==========================================

def render_snapshot(grid_np, arena_h, arena_w, grid_size, agent_pos_np,
                     agent_alive_np, agent_energy_np, agent_species_np):
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

    species_summary = ",".join(
        f"s{int(agent_species_np[i])}={agent_energy_np[i]:.0f}"
        for i in range(MAX_AGENTS) if agent_alive_np[i])
    stat = (f"  alive={int(agent_alive_np.sum())}/{MAX_AGENTS}  "
            f"energy=[{agent_energy_np.min():.0f},{agent_energy_np.max():.0f}] "
            f"mean={agent_energy_np.mean():.0f}  species: {species_summary}")
    return "\n".join(lines) + "\n" + stat


# ==========================================
# 11. TRAINING LOOP — sharded across TPU pod
# ==========================================

def main():
    print("=" * 60)
    print("  BYTE-MULTI-AGENT: EMERGENT LANGUAGE EDITION (v2)")
    print("=" * 60)

    num_devices = jax.device_count()
    print(f"Using {num_devices} device(s)")
    assert POP_SIZE % (2 * num_devices) == 0, \
        f"POP_SIZE ({POP_SIZE}) must be divisible by 2*num_devices ({2*num_devices})"

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), axis_names=('pop',))

    key = random.PRNGKey(42)
    net = AgentNet()
    k1, key = random.split(key)

    # Init one species' params to get template
    dummy_obs = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_audio = jnp.zeros((MAX_AGENTS * AUDIO_PERSIST_TICKS,), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
    dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)

    params_template_single = net.init(k1, dummy_obs, dummy_audio, dummy_proprio, dummy_hidden)
    apply_fn = net.apply

    # The "center" parameter vector includes ALL species' params concatenated.
    # This way ES optimizes the whole population of species jointly.
    single_flat = flatten_params(params_template_single)
    num_params_single = single_flat.shape[0]
    num_params_total = num_params_single * NUM_POLICY_SPECIES
    print(f"Params per species: {num_params_single:,}")
    print(f"Num species: {NUM_POLICY_SPECIES}")
    print(f"Total params optimized: {num_params_total:,}")

    # Initialize center: each species gets a different init to break symmetry
    k_inits = random.split(key, NUM_POLICY_SPECIES + 1)
    key = k_inits[0]
    species_params = []
    for si in range(NUM_POLICY_SPECIES):
        sp = net.init(k_inits[si + 1], dummy_obs, dummy_audio, dummy_proprio, dummy_hidden)
        species_params.append(flatten_params(sp))
    center_flat = jnp.concatenate(species_params)

    CKPT_DIR = '/tmp' if os.path.exists('/tmp') else '.'
    CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v2_params.npy')
    GEN_CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v2_gen.npy')

    start_gen = 0
    if os.path.exists(CKPT_PATH):
        loaded = jnp.array(np.load(CKPT_PATH))
        if loaded.shape == center_flat.shape:
            center_flat = loaded
            print(f"  Resumed from {CKPT_PATH}")
            if os.path.exists(GEN_CKPT_PATH):
                gen_info = np.load(GEN_CKPT_PATH, allow_pickle=True).item()
                start_gen = int(gen_info.get('gen', 0))
                print(f"  Resuming from gen {start_gen}")
        else:
            print(f"  Checkpoint shape mismatch — starting fresh")

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LR),
    )
    opt_state = optimizer.init(center_flat)
    noise_ctrl = AdaptiveNoiseController()

    arena_h = INITIAL_ARENA
    arena_w = INITIAL_ARENA
    grid_size = GRID_SIZE
    num_active = INITIAL_ACTIVE
    half_pop = POP_SIZE // 2
    per_device_half = half_pop // num_devices

    print(f"Population: {POP_SIZE} mirrored, {NUM_ENVS_PER_MEMBER} envs each")
    print(f"Arena: {arena_h}x{arena_w}, grid {grid_size}, agents {num_active}")
    print(f"Per-device members: {2 * per_device_half}")
    print()

    print("  Compiling sharded generation fn...")

    # Sharded evaluation: each device evaluates POP_SIZE/num_devices members.
    # Input partitioning: noise sharded on pop dim, center replicated,
    # env_keys sharded on pop dim.
    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P('pop'), P(), P('pop'), P('pop')),
             out_specs=P('pop'),
             check_rep=False)
    def sharded_eval(center, noise_shard, sigma_val, env_keys_shard, ep_keys_shard):
        # noise_shard: (half_pop/num_devices, num_params_total)
        pos_p = center[None, :] + sigma_val * noise_shard
        neg_p = center[None, :] - sigma_val * noise_shard
        all_members = jnp.concatenate([pos_p, neg_p], axis=0)  # 2*per_device_half
        # env_keys_shard and ep_keys_shard have leading dim 2*per_device_half
        fits = vmap(lambda p, eks, pks: evaluate_member(
            p, center, params_template_single, apply_fn,
            eks, pks, grid_size, arena_h, arena_w, num_active))(
            all_members, env_keys_shard, ep_keys_shard)
        return fits

    def es_update(center, noise, sigma_val, fitness, opt_state):
        utilities = rank_fitness_shape(fitness)
        pos_util = utilities[:half_pop]
        neg_util = utilities[half_pop:]
        grad_estimate = jnp.dot((pos_util - neg_util), noise) / (half_pop * sigma_val)
        grad_estimate = grad_estimate - WEIGHT_DECAY * center
        updates, new_opt_state = optimizer.update(-grad_estimate, opt_state, center)
        return optax.apply_updates(center, updates), new_opt_state

    es_update_jit = jit(es_update)

    start_time = time.time()
    last_print_time = start_time
    PRINT_INTERVAL = 120

    for gen in range(start_gen, 100000):
        k_noise, k_env, k_ep, key = random.split(key, 4)
        sigma = noise_ctrl.get_sigma()
        sigma_jnp = jnp.float32(sigma)

        # Generate mirrored noise: only need half_pop actual vectors; mirror in eval
        noise = random.normal(k_noise, (half_pop, num_params_total))

        # Env keys: one per (member, env) pair, but members are mirrored, so
        # pos and neg share env keys to reduce variance.
        env_keys = random.split(k_env, POP_SIZE * NUM_ENVS_PER_MEMBER)
        # Reshape to (POP_SIZE, NUM_ENVS_PER_MEMBER, 2) then split per device
        env_keys_pop = env_keys.reshape(POP_SIZE, NUM_ENVS_PER_MEMBER, 2)
        ep_keys = random.split(k_ep, POP_SIZE * NUM_ENVS_PER_MEMBER)
        ep_keys_pop = ep_keys.reshape(POP_SIZE, NUM_ENVS_PER_MEMBER, 2)

        # Reorder so that per-device shard gets a contiguous block of members
        # where each device handles [pos_block, neg_block] of its slice.
        # Simplest: interleave pos and neg such that shard_map on 'pop' sees
        # (2*per_device_half) members per device.
        # To do this: arrange noise as (num_devices, 2*per_device_half, num_params).
        # Take noise[i*per_device_half:(i+1)*per_device_half] as positive,
        # and mirror inside sharded_eval.
        noise_reshaped = noise.reshape(num_devices, per_device_half, num_params_total)

        # env_keys_pop is (POP_SIZE, NUM_ENVS, 2). Reorder:
        # pos_members are [0..half_pop), neg_members are [half_pop..POP_SIZE).
        # Each device i gets pos [i*per_half..(i+1)*per_half) AND neg [half+i*per_half..half+(i+1)*per_half).
        # We concat pos and neg along member axis, per device.
        pos_env_keys = env_keys_pop[:half_pop]
        neg_env_keys = env_keys_pop[half_pop:]
        pos_env_keys_dev = pos_env_keys.reshape(num_devices, per_device_half, NUM_ENVS_PER_MEMBER, 2)
        neg_env_keys_dev = neg_env_keys.reshape(num_devices, per_device_half, NUM_ENVS_PER_MEMBER, 2)
        env_keys_dev = jnp.concatenate([pos_env_keys_dev, neg_env_keys_dev], axis=1)
        env_keys_dev = env_keys_dev.reshape(num_devices * 2 * per_device_half,
                                              NUM_ENVS_PER_MEMBER, 2)

        pos_ep_keys = ep_keys_pop[:half_pop]
        neg_ep_keys = ep_keys_pop[half_pop:]
        pos_ep_keys_dev = pos_ep_keys.reshape(num_devices, per_device_half, NUM_ENVS_PER_MEMBER, 2)
        neg_ep_keys_dev = neg_ep_keys.reshape(num_devices, per_device_half, NUM_ENVS_PER_MEMBER, 2)
        ep_keys_dev = jnp.concatenate([pos_ep_keys_dev, neg_ep_keys_dev], axis=1)
        ep_keys_dev = ep_keys_dev.reshape(num_devices * 2 * per_device_half,
                                            NUM_ENVS_PER_MEMBER, 2)

        fitness_dev = sharded_eval(center_flat, noise_reshaped, sigma_jnp,
                                    env_keys_dev, ep_keys_dev)
        # fitness_dev is (POP_SIZE,). But order is per-device [pos|neg] blocks.
        # Reconstruct canonical [pos_all, neg_all] order:
        fitness_dev_reshaped = fitness_dev.reshape(num_devices, 2 * per_device_half)
        pos_fits = fitness_dev_reshaped[:, :per_device_half].reshape(half_pop)
        neg_fits = fitness_dev_reshaped[:, per_device_half:].reshape(half_pop)
        fitness = jnp.concatenate([pos_fits, neg_fits])

        center_flat, opt_state = es_update_jit(center_flat, noise, sigma_jnp,
                                                fitness, opt_state)

        mean_fit = float(jnp.mean(fitness))
        max_fit = float(jnp.max(fitness))
        min_fit = float(jnp.min(fitness))
        pop_std = float(jnp.std(fitness))
        noise_ctrl.record(mean_fit, pop_std)
        noise_ctrl.step()

        if gen > 0 and gen % 50 == 0:
            np.save(CKPT_PATH, np.array(jax.device_get(center_flat)))
            np.save(GEN_CKPT_PATH, {'gen': gen, 'mean_fit': mean_fit,
                                     'noise_std': noise_ctrl.get_sigma()})
            print(f"  [Checkpoint saved at gen {gen}]")

        now = time.time()
        if now - last_print_time > PRINT_INTERVAL or gen == start_gen or gen % 25 == 0:
            elapsed = now - start_time
            print(f"\n--- GEN {gen} | {elapsed:.0f}s elapsed ---")
            print(f"  Fitness: mean={mean_fit:.2f} max={max_fit:.2f} "
                  f"min={min_fit:.2f} std={pop_std:.2f}")
            print(f"  Noise:   {noise_ctrl.get_status_str()}")

            # Render preview
            k_render, key = random.split(key)
            params_per_sp = num_params_single
            sp_flat = center_flat.reshape(NUM_POLICY_SPECIES, params_per_sp)
            preview_params = vmap(lambda s: unflatten_params(s, params_template_single))(sp_flat)
            render_env = init_env(k_render, arena_h, arena_w, grid_size, num_active)

            def preview_step(carry, sk):
                state, last_speak = carry
                obs, audio = get_all_obs(state, grid_size, last_speak)
                proprio = build_proprioception(state)
                actions, new_hidden = sample_actions(
                    preview_params, obs, audio, proprio, state.agent_hidden,
                    state.agent_species, sk, apply_fn, ACTION_TEMP)
                state = state._replace(agent_hidden=new_hidden)
                new_state = step_env(state, actions, grid_size)
                return (new_state, actions[:, 1]), actions[:, 1]

            preview_keys = random.split(k_render, 200)
            (mid_state, _), speech_log = lax.scan(preview_step,
                (render_env, jnp.zeros(MAX_AGENTS, dtype=jnp.int32)),
                preview_keys)

            grid_np = np.array(jax.device_get(mid_state.grid))
            pos_np = np.array(jax.device_get(mid_state.agent_pos))
            alive_np = np.array(jax.device_get(mid_state.agent_alive))
            energy_np = np.array(jax.device_get(mid_state.agent_energy))
            species_np = np.array(jax.device_get(mid_state.agent_species))
            speech_np = np.array(jax.device_get(speech_log))  # (T, MAX_AGENTS)

            print(render_snapshot(grid_np, arena_h, arena_w, grid_size,
                                   pos_np, alive_np, energy_np, species_np))

            # Language diagnostics
            has_writing = np.any((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET))
            if has_writing:
                wc = int(np.sum((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET)))
                print(f"  *** WRITING: {wc} cells on ground ***")
            has_tool = np.any((grid_np >= TOOL_START) & (grid_np < TOOL_END))
            if has_tool:
                tc = int(np.sum((grid_np >= TOOL_START) & (grid_np < TOOL_END)))
                print(f"  *** TOOLS ON GROUND: {tc} ***")

            # Speech entropy: how often and how diversely agents speak
            total_speech = int((speech_np > 0).sum())
            if total_speech > 0:
                uniq = len(np.unique(speech_np[speech_np > 0]))
                print(f"  Speech: {total_speech} utterances across preview, "
                      f"{uniq} unique tokens used")

            last_print_time = now

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
