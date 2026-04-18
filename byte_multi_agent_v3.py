# %% [markdown]
# # Byte-Multi-Agent: Coordination Edition (v3.2)
#
# Built on v3.1, this version directly addresses the failure mode observed in
# v3.1: agents discovered feast briefly (gens 8-22) but reverted to solo
# foraging because regular food was sufficient on its own. Selection chose
# the lower-variance, lower-effort strategy.
#
# === Changes from v3.1 ===
#  1. SOLO FORAGING IS NO LONGER ENOUGH. Reduced regular food energy and
#     density, and slowed regrowth, so that a single agent cannot survive a
#     full episode on solo eating alone. Feast becomes mandatory, not optional.
#  2. s1 IS NO LONGER BROKEN. v3.1's s1 had weak vision AND weak audio AND
#     expensive metabolism. It had no viable path. v3.2 gives s1 full vision
#     and only weak audio, so it can find food but benefits MASSIVELY from
#     hearing s0's broadcasts.
#  3. FEAST ENERGY DOMINATES. With regular food at 10 energy and feast still
#     at 50 per agent, one successful feast = 5 regular meals. Coordination
#     becomes the obvious winning strategy if discoverable.
#  4. Diagnostic of feast events is now per-generation summary (averaged
#     across pop) instead of just one preview, so we can track adoption rate.
#
# Math sanity check (v3.2):
#   - Episode length: 400 ticks
#   - Metabolic + move cost (active agent): ~0.5/tick → ~200 energy needed
#   - Initial energy: 50
#   - So agent needs to gain ~150 energy from eating during the episode
#   - With FOOD_ENERGY=10, that's 15 successful food eats
#   - With INITIAL_FOOD_DENSITY=0.025, only ~9 food tiles in arena initially
#   - Regrowth at 0.002 = ~1.6 regrowths over 400 ticks per seed tile
#   - Net: solo agents simply cannot survive without feast.

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

# --- World Bytes ---
EMPTY = 0
WALL = 1
FOOD = 2
SEED = 3
ROCK = 4
WATER = 5
FEAST = 6        # NEW: cooperative food, huge reward but only when shared

WRITE_OFFSET = 7      # 7-32: a-z persistent writing (shifted by 1 for FEAST)
SPEAK_OFFSET = 33     # 33-58: A-Z transient (display only)
AGENT_MARK = 59       # @ on display
TOOL_START = 60
TOOL_END = 97        # 37 possible tool bytes
NUM_BYTE_TYPES = 128

NUM_ACTIVE_TOOLS = 4

CHAR_MAP = {
    EMPTY: '.', WALL: '#', FOOD: '*', SEED: ',', ROCK: 'o', WATER: '~',
    FEAST: '&',  # the cooperative food
    AGENT_MARK: '@',
}
for i in range(26):
    CHAR_MAP[i + WRITE_OFFSET] = chr(97 + i)   # a-z
    CHAR_MAP[i + SPEAK_OFFSET] = chr(65 + i)   # A-Z
_TOOL_CHARS = "0123456789!@#$%^&*()_+-=[]{};:<>?/|\\"
for i in range(TOOL_START, TOOL_END):
    CHAR_MAP[i] = _TOOL_CHARS[(i - TOOL_START) % len(_TOOL_CHARS)]

# --- Environment ---
VISION_RADIUS = 7
VISION_SIZE = 2 * VISION_RADIUS + 1  # 15
MAX_AGENTS = 8
EPISODE_LENGTH = 400
INVENTORY_SIZE = 4

AUDIO_RANGE = 10
AUDIO_PERSIST_TICKS = 3

# --- Energy / economy ---
INITIAL_ENERGY = 50.0
MAX_ENERGY = 100.0
FOOD_ENERGY = 10.0               # was 18; now solo foraging can't sustain
FEAST_ENERGY_PER_AGENT = 50.0    # 5x regular food — coordination is worth it
FEAST_RADIUS = 3
FEAST_WINDOW_TICKS = 8
FEAST_MIN_AGENTS = 2

METABOLIC_COST_BASE = 0.15
MOVE_COST = 0.4
WRITE_COST = 0.6
OVERWRITE_COST = 1.0
SPEAK_COST = 0.15
PICKUP_COST = 0.3
USE_TOOL_COST = 0.5

# --- Telepathy bonus (annealed) ---
# Reduced from 4.0 → 1.5 to stop drowning the food signal. With v3's setting,
# telepathy credit was running 200-300 per preview while raw fitness was ~80,
# meaning the gradient was dominated by "speak constantly hoping for credit."
TELEPATHY_BONUS_INIT = 1.5
TELEPATHY_ANNEAL_GENS = 1500
TELEPATHY_ANNEAL_FLOOR = 0.0

# --- Ecology ---
SEED_TO_FOOD_PROB = 0.002        # was 0.004; food regrows slowly now
INITIAL_FOOD_DENSITY = 0.025     # was 0.05; halve initial food
INITIAL_SEED_DENSITY = 0.04
INITIAL_ROCK_DENSITY = 0.05
INITIAL_WATER_DENSITY = 0.05
INITIAL_FEAST_DENSITY = 0.018    # ~6-7 feast tiles in a 20x20 arena
FEAST_REGROW_PROB = 0.0012       # rare but non-trivial — feast is precious

CRAFT_PROB = 0.02

# --- Heterogeneous policies ---
NUM_POLICY_SPECIES = 2

# Species traits redesigned for v3.2 to make both species viable:
#   sp0 "broadcaster": cheap metabolism, far audio, cheap speech, full vision.
#        Good at exploring and signaling. Less efficient eater.
#   sp1 "harvester":  full vision, very efficient eater, weak audio (only
#        nearby), normal speech cost. Has to STAY CLOSE to broadcasters to
#        benefit from their info, but converts food more efficiently when
#        it does eat. The pairing now has clear interdependence: sp0 finds
#        food and yells, sp1 hears (when nearby) and eats efficiently.
SPECIES_METABOLIC_MULT = jnp.array([0.8, 1.0])
SPECIES_FOOD_EFF       = jnp.array([0.8, 1.5])
SPECIES_AUDIO_MULT     = jnp.array([1.5, 0.7])
SPECIES_SPEAK_MULT     = jnp.array([0.5, 1.0])
SPECIES_VISION_QUALITY = jnp.array([1.0, 1.0])  # both species have full vision now

NUM_VISION_CATEGORIES = 6

# --- ES Hyperparameters ---
# NOTE: POP_SIZE and NUM_POLICY_SPECIES sized down for Kaggle TPU memory (small HBM pool).
# On TRC v3-8 or larger, you can safely raise POP_SIZE to 512 or 1024 and
# NUM_POLICY_SPECIES to 4. The per-species trait arrays already have 4 entries —
# they will be silently truncated to NUM_POLICY_SPECIES.
POP_SIZE = 256
NUM_ENVS_PER_MEMBER = 4
LR = 0.01
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.005

# --- Adaptive Noise ---
NOISE_STD_INIT = 0.02
NOISE_STD_MIN = 0.006
NOISE_STD_MAX = 0.03
NOISE_WARMUP_GENS = 30
NOISE_SNR_WINDOW = 20
NOISE_GROW_FACTOR = 1.02
NOISE_SHRINK_FACTOR = 0.95
NOISE_RATCHET_TOLERANCE = 0.30
NOISE_RATCHET_MAX_DURATION = 50

# --- Plateau-busting sigma reset ---
PLATEAU_DETECT_WINDOW = 30       # gens to look back for stagnation
PLATEAU_IMP_THRESHOLD = 5.0      # if mean improvement per gen drops below this for window, reset
PLATEAU_RESET_COOLDOWN = 50      # don't reset more than once per N gens

ACTION_TEMP = 0.5

INITIAL_ARENA = 20
INITIAL_ACTIVE = 6
GRID_SIZE = 64

MOVE_DIRS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)
HIDDEN_SIZE = 128


# ==========================================
# 2. ADAPTIVE NOISE CONTROLLER
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
        # NEW: plateau reset bookkeeping
        self.last_plateau_reset_gen = -10000

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

    def maybe_plateau_reset(self, current_gen):
        """If we've stagnated and aren't in cooldown, reset sigma to high
        exploration to break out. Returns True if reset fired."""
        n = len(self.mean_history)
        if n < PLATEAU_DETECT_WINDOW:
            return False
        if current_gen - self.last_plateau_reset_gen < PLATEAU_RESET_COOLDOWN:
            return False
        recent = self.mean_history[-PLATEAU_DETECT_WINDOW:]
        avg_improvement = (recent[-1] - recent[0]) / max(PLATEAU_DETECT_WINDOW - 1, 1)
        if avg_improvement < PLATEAU_IMP_THRESHOLD:
            self.sigma = NOISE_STD_MAX
            self.last_plateau_reset_gen = current_gen
            self.improvement_ema = 0.0
            self.ratchet_consecutive = 0
            print(f"  >> PLATEAU RESET at gen {current_gen}: "
                  f"avg improvement {avg_improvement:.3f} < {PLATEAU_IMP_THRESHOLD}. "
                  f"Sigma forced to {NOISE_STD_MAX}.")
            return True
        return False

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

def _build_craft_table():
    rng = np.random.RandomState(31337)
    table = np.zeros((NUM_BYTE_TYPES, NUM_BYTE_TYPES), dtype=np.int32)
    for a in range(NUM_BYTE_TYPES):
        for b in range(a, NUM_BYTE_TYPES):
            out = TOOL_START + rng.randint(0, TOOL_END - TOOL_START)
            table[a, b] = out
            table[b, a] = out
    return table

CRAFT_TABLE = jnp.array(_build_craft_table())

def _is_ingredient(b):
    return (b == ROCK) | (b == WATER) | (b == SEED) | \
           ((b >= TOOL_START) & (b < TOOL_END))

_rng_tools = np.random.RandomState(777)
_active_tool_indices = _rng_tools.choice(TOOL_END - TOOL_START, size=NUM_ACTIVE_TOOLS, replace=False)
_tool_effect_table = np.full((TOOL_END - TOOL_START,), -1, dtype=np.int32)
for i, idx in enumerate(_active_tool_indices):
    _tool_effect_table[idx] = i
TOOL_EFFECT_TABLE = jnp.array(_tool_effect_table)

print(f"Active tools (bytes): {[int(TOOL_START + i) for i in _active_tool_indices]}")
print(f"  → effects 0..{NUM_ACTIVE_TOOLS-1}: scatter-seed, rock→food, water→seed, energy-boost")


# ==========================================
# 4. ENVIRONMENT STATE
# ==========================================

class EnvState(NamedTuple):
    grid: jnp.ndarray
    agent_pos: jnp.ndarray
    agent_alive: jnp.ndarray
    agent_energy: jnp.ndarray
    agent_inventory: jnp.ndarray
    agent_underfoot: jnp.ndarray
    agent_hidden: jnp.ndarray
    agent_species: jnp.ndarray
    agent_vision_mask: jnp.ndarray
    agent_last_ate_tick: jnp.ndarray
    audio_buffer: jnp.ndarray
    audio_age: jnp.ndarray
    survival_ticks: jnp.ndarray
    energy_earned: jnp.ndarray
    last_speaker_tick: jnp.ndarray
    last_heard_at_tick: jnp.ndarray
    feast_eaters: jnp.ndarray
    telepathy_credit: jnp.ndarray
    # NEW in v3.1: per-agent "recently ate feast" tracking for asynchronous coord
    recent_feast_tick: jnp.ndarray   # (MAX_AGENTS,) — last tick this agent ate feast
    recent_feast_pos: jnp.ndarray    # (MAX_AGENTS, 2) — position when they ate feast
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

    k1, k2, k3, k4, k5 = random.split(k_blocks, 5)
    food_noise = random.uniform(k1, (grid_size, grid_size))
    seed_noise = random.uniform(k2, (grid_size, grid_size))
    rock_noise = random.uniform(k3, (grid_size, grid_size))
    water_noise = random.uniform(k4, (grid_size, grid_size))
    feast_noise = random.uniform(k5, (grid_size, grid_size))

    grid = jnp.where(playable & (food_noise < INITIAL_FOOD_DENSITY), FOOD, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (seed_noise < INITIAL_SEED_DENSITY), SEED, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (rock_noise < INITIAL_ROCK_DENSITY), ROCK, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (water_noise < INITIAL_WATER_DENSITY), WATER, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (feast_noise < INITIAL_FEAST_DENSITY), FEAST, grid)

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

    agent_species = random.randint(k_species, (MAX_AGENTS,), 0, NUM_POLICY_SPECIES)

    mask_noise = random.uniform(k_mask, (MAX_AGENTS, NUM_VISION_CATEGORIES))
    agent_vision_mask = mask_noise < 0.7

    agent_last_ate_tick = jnp.full((MAX_AGENTS,), -1000, dtype=jnp.int32)
    audio_buffer = jnp.zeros((MAX_AGENTS, AUDIO_PERSIST_TICKS), dtype=jnp.int32)
    audio_age = jnp.full((MAX_AGENTS, AUDIO_PERSIST_TICKS), AUDIO_PERSIST_TICKS + 1, dtype=jnp.int32)

    survival_ticks = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    energy_earned = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)

    last_speaker_tick = jnp.full((MAX_AGENTS,), -1000, dtype=jnp.int32)
    last_heard_at_tick = jnp.full((MAX_AGENTS, MAX_AGENTS), -1000, dtype=jnp.int32)
    feast_eaters = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    telepathy_credit = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    recent_feast_tick = jnp.full((MAX_AGENTS,), -1000, dtype=jnp.int32)
    recent_feast_pos = jnp.zeros((MAX_AGENTS, 2), dtype=jnp.int32)

    return EnvState(
        grid=grid, agent_pos=agent_pos, agent_alive=agent_alive,
        agent_energy=agent_energy, agent_inventory=agent_inventory,
        agent_underfoot=agent_underfoot, agent_hidden=agent_hidden,
        agent_species=agent_species, agent_vision_mask=agent_vision_mask,
        agent_last_ate_tick=agent_last_ate_tick,
        audio_buffer=audio_buffer, audio_age=audio_age,
        survival_ticks=survival_ticks, energy_earned=energy_earned,
        last_speaker_tick=last_speaker_tick,
        last_heard_at_tick=last_heard_at_tick,
        feast_eaters=feast_eaters,
        telepathy_credit=telepathy_credit,
        recent_feast_tick=recent_feast_tick,
        recent_feast_pos=recent_feast_pos,
        tick=jnp.int32(0), rng=k_rng,
    )


# ==========================================
# 5. OBSERVATIONS
# ==========================================

def apply_vision_mask(obs_bytes, mask, vision_quality, vq_key):
    """Apply per-category mask AND species-specific vision quality (random dropout
    of cells). vq_key is per-agent per-tick PRNGKey."""
    is_food = (obs_bytes == FOOD) | (obs_bytes == SEED) | (obs_bytes == FEAST)
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

    # Vision-quality dropout: each non-empty cell has prob (1 - vq) of being hidden
    vq_noise = random.uniform(vq_key, obs_bytes.shape)
    is_nonempty = (obs_bytes != EMPTY) & (obs_bytes != WALL)
    vq_hide = is_nonempty & (vq_noise > vision_quality)

    hide = hide | vq_hide
    return jnp.where(hide, EMPTY, obs_bytes)


def get_all_obs(state, grid_size, last_speak, obs_key):
    grid = state.grid

    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        body = jnp.where(last_speak[i] > 0,
                         jnp.minimum(last_speak[i] - 1 + SPEAK_OFFSET, AGENT_MARK - 1),
                         AGENT_MARK)
        return jnp.where(state.agent_alive[i], g.at[r, c].set(body), g), None

    composed, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    padded = jnp.pad(composed, VISION_RADIUS, constant_values=WALL)

    species_vq = SPECIES_VISION_QUALITY[state.agent_species]  # (MAX_AGENTS,)
    obs_keys = random.split(obs_key, MAX_AGENTS)

    def get_one(pos, mask, vq, k):
        crop = lax.dynamic_slice(padded, (pos[0], pos[1]), (VISION_SIZE, VISION_SIZE))
        return apply_vision_mask(crop, mask, vq, k)

    obs_bytes = vmap(get_one)(state.agent_pos, state.agent_vision_mask, species_vq, obs_keys)
    obs_norm = obs_bytes.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)

    # Audio: per-species audio range
    species_audio_range = AUDIO_RANGE * SPECIES_AUDIO_MULT[state.agent_species]  # (MAX_AGENTS,)

    def audio_for_listener(listener_idx):
        lp = state.agent_pos[listener_idx]
        my_range = species_audio_range[listener_idx]
        dp = state.agent_pos - lp[None, :]
        chebyshev = jnp.maximum(jnp.abs(dp[:, 0]), jnp.abs(dp[:, 1])).astype(jnp.float32)
        audible = (chebyshev <= my_range) & state.agent_alive & \
                  (jnp.arange(MAX_AGENTS) != listener_idx)
        fresh = state.audio_age < AUDIO_PERSIST_TICKS
        gated = state.audio_buffer * fresh.astype(jnp.int32) * \
                audible[:, None].astype(jnp.int32)
        return gated.reshape(-1).astype(jnp.float32) / 27.0

    obs_audio = vmap(audio_for_listener)(jnp.arange(MAX_AGENTS))
    return obs_norm, obs_audio


# ==========================================
# 6. PHYSICS
# ==========================================

def is_passable_byte(b):
    return (b == EMPTY) | (b == FOOD) | (b == SEED) | (b == FEAST) | \
           ((b >= WRITE_OFFSET) & (b < SPEAK_OFFSET))


def step_env(state, actions, grid_size, telepathy_strength):
    """telepathy_strength: scalar ∈ [0, TELEPATHY_BONUS_INIT], annealed by training loop."""
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

    # Movement resolution
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

    # Underfoot tracking
    new_underfoot_moved = jnp.where(ate_food, SEED,
                            jnp.where(stepped_on_feast, EMPTY, target_vals))
    new_underfoot = jnp.where(moved, new_underfoot_moved, state.agent_underfoot)

    # Writing
    leave_byte = jnp.where(write_acts > 0, write_acts - 1 + WRITE_OFFSET, state.agent_underfoot)
    is_overwrite = (write_acts > 0) & (state.agent_underfoot != EMPTY) & \
                    (state.agent_underfoot != leave_byte)

    # Grid updates
    new_grid = state.grid
    alive_moved = moved & state.agent_alive
    leave_r = jnp.where(alive_moved, curr_pos[:, 0], 0)
    leave_c = jnp.where(alive_moved, curr_pos[:, 1], 0)
    deposit_vals = jnp.where(alive_moved, leave_byte, new_grid[leave_r, leave_c])
    new_grid = new_grid.at[leave_r, leave_c].set(deposit_vals)

    # Consume regular food
    eat_r = jnp.where(ate_food, final_pos[:, 0], 0)
    eat_c = jnp.where(ate_food, final_pos[:, 1], 0)
    keep_vals = jnp.where(ate_food, EMPTY, new_grid[eat_r, eat_c])
    new_grid = new_grid.at[eat_r, eat_c].set(keep_vals)

    # FEAST mechanic (v3.1): asynchronous coordination via temporal window.
    # When an agent steps on a feast tile, the tile is consumed AND the agent
    # is marked as having "recently feasted" at that position for FEAST_WINDOW_TICKS.
    # If during that window, ANOTHER agent eats a feast tile within FEAST_RADIUS
    # of the recent feaster's position, BOTH agents receive the bonus.
    # This means coordination doesn't have to be synchronous — agent A can eat
    # at tick 10, agent B can eat at tick 15 nearby, both get the bonus.
    # Each agent can only "claim partner" with one other agent per feast event,
    # and a feast pair fires the bonus exactly once (when the second agent eats).

    feast_now = stepped_on_feast & state.agent_alive

    # For each agent feasting NOW, check whether any OTHER agent has a recent
    # feast within window AND within FEAST_RADIUS of this agent's position.
    # If yes → both agents get the bonus this tick.
    fp_pos = final_pos
    # Distance from this agent's current pos to each other agent's recent_feast_pos
    rfp = state.recent_feast_pos                       # (MAX_AGENTS, 2)
    dpij = fp_pos[:, None, :] - rfp[None, :, :]        # (i: feaster_now, j: prior_feaster)
    cheb_ij = jnp.maximum(jnp.abs(dpij[:, :, 0]), jnp.abs(dpij[:, :, 1]))
    # j must have feasted recently enough (within window) AND be alive
    j_recent = (state.tick - state.recent_feast_tick) <= FEAST_WINDOW_TICKS
    j_recent = j_recent & state.agent_alive
    # Pair valid if i is feasting now AND j had recent feast AND in range AND i != j
    pair_valid = (cheb_ij <= FEAST_RADIUS) & feast_now[:, None] & j_recent[None, :]
    pair_valid = pair_valid & ~jnp.eye(MAX_AGENTS, dtype=jnp.bool_)

    # has_feast_partner_now[i] = True if agent i (feasting now) has at least one
    # eligible recent partner j
    has_feast_partner_now = jnp.any(pair_valid, axis=1)
    # gets_partner_credit[j] = True if any feaster_now i paired with j
    gets_partner_credit = jnp.any(pair_valid, axis=0)

    # Bonus this tick:
    #   - feasters who paired with a recent partner: full bonus
    #   - prior feasters who got "claimed" by a new feaster: full bonus too
    # Note: a single agent can be both i and j across different events but
    # we just sum any.
    bonus_for_now = jnp.where(has_feast_partner_now, FEAST_ENERGY_PER_AGENT, 0.0)
    bonus_for_prior = jnp.where(gets_partner_credit, FEAST_ENERGY_PER_AGENT, 0.0)
    feast_bonus_this_tick = bonus_for_now + bonus_for_prior

    # Update recent_feast_tick and recent_feast_pos:
    # - If agent feasted now, mark them as recent feaster at their final_pos.
    # - Otherwise, leave existing (it'll naturally expire via the window check).
    new_recent_feast_tick = jnp.where(feast_now, state.tick, state.recent_feast_tick)
    new_recent_feast_pos = jnp.where(feast_now[:, None], final_pos, state.recent_feast_pos)

    # Consume feast cells: any FEAST cell that an agent stepped on this tick gets
    # erased. Eating feast solo still consumes the tile (single-agent eats waste it
    # IF no recent partner — but the agent is now marked as a recent feaster, so
    # if a partner arrives in time, they BOTH still get rewarded). This is the key
    # asymmetry: solo eating sets up the possibility of cooperative reward.
    feast_eat_r = jnp.where(stepped_on_feast, final_pos[:, 0], 0)
    feast_eat_c = jnp.where(stepped_on_feast, final_pos[:, 1], 0)
    keep_feast = jnp.where(stepped_on_feast, EMPTY, new_grid[feast_eat_r, feast_eat_c])
    new_grid = new_grid.at[feast_eat_r, feast_eat_c].set(keep_feast)

    # Keep the legacy variable name for downstream code compatibility
    has_feast_partner = has_feast_partner_now | gets_partner_credit

    # Pickup / Use / Drop (same as v2)
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

        selected = inv_i[slot]
        tool_offset = jnp.clip(selected - TOOL_START, 0, TOOL_END - TOOL_START - 1)
        effect_id = jnp.where(
            (selected >= TOOL_START) & (selected < TOOL_END),
            TOOL_EFFECT_TABLE[tool_offset], -1)
        can_use = (act == 2) & alive & (selected != EMPTY) & (effect_id >= 0)

        def apply_scatter_seed(g):
            def one(gg, k):
                nr, nc = neighbors[k, 0], neighbors[k, 1]
                cond = (effect_id == 0) & (gg[nr, nc] == EMPTY)
                return jnp.where(cond, gg.at[nr, nc].set(SEED), gg), None
            gg, _ = lax.scan(one, g, jnp.arange(4))
            return gg
        grid = jnp.where(can_use & (effect_id == 0), apply_scatter_seed(grid), grid)

        def apply_rock_to_food(g):
            is_rock_n = (neigh_vals == ROCK)
            any_rock = jnp.any(is_rock_n)
            idx = jnp.argmax(is_rock_n)
            nr, nc = neighbors[idx, 0], neighbors[idx, 1]
            return jnp.where((effect_id == 1) & any_rock, g.at[nr, nc].set(FOOD), g)
        grid = jnp.where(can_use & (effect_id == 1), apply_rock_to_food(grid), grid)

        def apply_water_to_seed(g):
            is_water_n = (neigh_vals == WATER)
            any_water = jnp.any(is_water_n)
            idx = jnp.argmax(is_water_n)
            nr, nc = neighbors[idx, 0], neighbors[idx, 1]
            return jnp.where((effect_id == 2) & any_water, g.at[nr, nc].set(SEED), g)
        grid = jnp.where(can_use & (effect_id == 2), apply_water_to_seed(grid), grid)

        energy_boost = jnp.where((effect_id == 3) & can_use, 30.0, 0.0)
        energy_i_new = energy[i] + energy_boost

        inv_i = jnp.where(can_use, inv_i.at[slot].set(EMPTY), inv_i)

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

    # Crafting (same as v2)
    k_craft, new_rng = random.split(state.rng)
    craft_noise = random.uniform(k_craft, (grid_size, grid_size))
    craft_choose = random.randint(k_craft, (grid_size, grid_size), 0, 4)

    is_ingr = _is_ingredient(new_grid)
    up    = jnp.pad(new_grid[1:, :],  ((0, 1), (0, 0)), constant_values=EMPTY)
    down  = jnp.pad(new_grid[:-1, :], ((1, 0), (0, 0)), constant_values=EMPTY)
    left  = jnp.pad(new_grid[:, 1:],  ((0, 0), (0, 1)), constant_values=EMPTY)
    right = jnp.pad(new_grid[:, :-1], ((0, 0), (1, 0)), constant_values=EMPTY)
    neighbors_stack = jnp.stack([up, down, left, right], axis=0)
    chosen_neighbor = neighbors_stack[craft_choose,
                                       jnp.arange(grid_size)[:, None],
                                       jnp.arange(grid_size)[None, :]]
    neighbor_is_ingr = _is_ingredient(chosen_neighbor)

    craft_fire = (craft_noise < CRAFT_PROB) & is_ingr & neighbor_is_ingr
    craft_fire = craft_fire & (new_grid <= chosen_neighbor)
    craft_output = CRAFT_TABLE[new_grid, chosen_neighbor]
    new_grid = jnp.where(craft_fire, craft_output, new_grid)

    # Seed -> Food
    k_regrow, new_rng = random.split(new_rng)
    regrow_noise = random.uniform(k_regrow, (grid_size, grid_size))
    is_seed = (new_grid == SEED)
    regrow_trigger = is_seed & (regrow_noise < SEED_TO_FOOD_PROB)
    new_grid = jnp.where(regrow_trigger, FOOD, new_grid)

    # FEAST regrowth: empty arena cells occasionally become FEAST
    k_feast, new_rng = random.split(new_rng)
    feast_regrow_noise = random.uniform(k_feast, (grid_size, grid_size))
    # Only regrow on truly empty arena cells (not walls)
    is_empty_cell = (new_grid == EMPTY)
    # Restrict to interior of grid (we don't track arena bounds here; rely on walls)
    feast_regrow_trigger = is_empty_cell & (feast_regrow_noise < FEAST_REGROW_PROB)
    new_grid = jnp.where(feast_regrow_trigger, FEAST, new_grid)

    # ENERGY accounting
    species_metabolic = METABOLIC_COST_BASE * SPECIES_METABOLIC_MULT[state.agent_species]
    species_food_eff = SPECIES_FOOD_EFF[state.agent_species]
    species_speak_cost = SPEAK_COST * SPECIES_SPEAK_MULT[state.agent_species]

    new_energy = new_energy - jnp.where(state.agent_alive, species_metabolic, 0.0)
    new_energy = new_energy - jnp.where(moved, MOVE_COST, 0.0)
    new_energy = new_energy - jnp.where((speak_acts > 0) & state.agent_alive,
                                          species_speak_cost, 0.0)
    wrote_something = (write_acts > 0) & moved
    base_write = jnp.where(wrote_something, WRITE_COST, 0.0)
    extra_overwrite = jnp.where(is_overwrite, OVERWRITE_COST - WRITE_COST, 0.0)
    new_energy = new_energy - base_write - extra_overwrite

    food_gain = jnp.where(ate_food, FOOD_ENERGY * species_food_eff, 0.0)
    new_energy = new_energy + food_gain
    new_energy = new_energy + feast_bonus_this_tick

    # --- TELEPATHY BONUS ---
    # When listener i eats food (ate_food[i] OR has_feast_partner[i]) AND a nearby
    # speaker j spoke within last few ticks, both get telepathy_strength energy.
    listener_ate = ate_food | has_feast_partner

    # For each listener i, find any speaker j who was audible recently
    # last_heard_at_tick[i, j] = last tick listener i heard speaker j
    recent_speech_window = AUDIO_PERSIST_TICKS  # within this many ticks
    heard_recently = (state.tick - state.last_heard_at_tick) <= recent_speech_window
    # Only credit if listener actually ate this tick AND was alive
    will_credit = listener_ate[:, None] & heard_recently  # (listener, speaker)
    # Don't credit self
    will_credit = will_credit & ~eye
    # Both speaker and listener must be alive
    will_credit = will_credit & state.agent_alive[:, None] & state.agent_alive[None, :]
    # Award per pair: each listener can credit multiple speakers; each speaker
    # gets credit for each listener that ate after hearing them.
    listener_credit = jnp.sum(will_credit.astype(jnp.float32), axis=1) * telepathy_strength
    speaker_credit  = jnp.sum(will_credit.astype(jnp.float32), axis=0) * telepathy_strength
    new_energy = new_energy + listener_credit + speaker_credit
    new_telepathy_credit = state.telepathy_credit + listener_credit + speaker_credit

    new_energy = jnp.clip(new_energy, 0.0, MAX_ENERGY)

    new_last_ate = jnp.where(ate_food | has_feast_partner, state.tick, new_last_ate)
    new_energy_earned = state.energy_earned + food_gain + feast_bonus_this_tick
    new_feast_eaters = state.feast_eaters + feast_bonus_this_tick

    new_alive = state.agent_alive & (new_energy > 0.0)
    new_survival = state.survival_ticks + new_alive.astype(jnp.float32)

    # Audio buffer update + heard tracking
    new_audio_age = jnp.minimum(state.audio_age + 1, AUDIO_PERSIST_TICKS + 10)
    new_audio_buffer = jnp.concatenate([
        speak_acts[:, None],
        state.audio_buffer[:, :-1]
    ], axis=1)
    new_audio_age = jnp.concatenate([
        jnp.zeros((MAX_AGENTS, 1), dtype=jnp.int32),
        new_audio_age[:, :-1]
    ], axis=1)

    # Update last_speaker_tick: if agent spoke this tick, mark it
    new_last_speaker = jnp.where((speak_acts > 0) & state.agent_alive,
                                   state.tick, state.last_speaker_tick)

    # Update last_heard_at_tick: for each listener, for each speaker, if speaker
    # spoke this tick AND listener is in audible range, record state.tick.
    species_audio_range = AUDIO_RANGE * SPECIES_AUDIO_MULT[state.agent_species]
    spoke_now = (speak_acts > 0) & state.agent_alive  # (MAX_AGENTS,) speakers
    # Pairwise distances (listener, speaker) using FINAL positions
    dpij_final = final_pos[:, None, :] - final_pos[None, :, :]
    cheb_final = jnp.maximum(jnp.abs(dpij_final[:, :, 0]),
                              jnp.abs(dpij_final[:, :, 1])).astype(jnp.float32)
    audible_now = (cheb_final <= species_audio_range[:, None]) & state.agent_alive[:, None] & spoke_now[None, :]
    new_last_heard = jnp.where(audible_now, state.tick, state.last_heard_at_tick)

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
        last_speaker_tick=new_last_speaker,
        last_heard_at_tick=new_last_heard,
        feast_eaters=new_feast_eaters,
        telepathy_credit=new_telepathy_credit,
        recent_feast_tick=new_recent_feast_tick,
        recent_feast_pos=new_recent_feast_pos,
        tick=state.tick + 1,
        rng=new_rng,
    )


# ==========================================
# 7. NETWORK
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

        a = nn.Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.float32)(audio_bf)
        a = nn.relu(a)
        a = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.float32)(a)
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
        slot_logits  = nn.Dense(INVENTORY_SIZE, param_dtype=jnp.float32)(h32)

        return (move_logits, speak_logits, write_logits, puu_logits, slot_logits, h32)


def build_proprioception(state):
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
    keys = random.split(key, MAX_AGENTS * 5).reshape(MAX_AGENTS, 5, 2)

    def act_one(obs, audio, proprio, hidden, species, ks):
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

def run_episode(all_params, apply_fn, env_state, grid_size, ep_key, telepathy_strength):
    def step_fn(carry, step_keys):
        state, last_speak = carry
        sk_act, sk_obs = step_keys[0], step_keys[1]
        obs, audio = get_all_obs(state, grid_size, last_speak, sk_obs)
        proprio = build_proprioception(state)
        actions, new_hidden = sample_actions(
            all_params, obs, audio, proprio, state.agent_hidden,
            state.agent_species, sk_act, apply_fn, ACTION_TEMP)

        state = state._replace(agent_hidden=new_hidden)
        new_state = step_env(state, actions, grid_size, telepathy_strength)
        return (new_state, actions[:, 1]), None

    init_speak = jnp.zeros(MAX_AGENTS, dtype=jnp.int32)
    # 2 keys per step: one for action sampling, one for vision dropout
    step_keys = random.split(ep_key, EPISODE_LENGTH * 2).reshape(EPISODE_LENGTH, 2, 2)
    (final_state, _), _ = lax.scan(
        step_fn, (env_state, init_speak), step_keys)
    return final_state


def compute_per_species_fitness(final_state):
    """Returns (species_fitness, coord_metrics):
      species_fitness: (NUM_POLICY_SPECIES,) average fitness per species
      coord_metrics: (3,) array of [total_feast_energy, total_telepathy_credit,
                                     total_regular_food_energy] across all agents
    """
    survival = final_state.survival_ticks
    earned   = final_state.energy_earned
    final_e  = final_state.agent_energy
    feast    = final_state.feast_eaters
    telepathy = final_state.telepathy_credit

    per_agent_fit = (survival
                     + 0.5 * earned
                     + 0.1 * final_e
                     + 0.5 * feast
                     + 0.3 * telepathy)

    species = final_state.agent_species
    one_hot = jax.nn.one_hot(species, NUM_POLICY_SPECIES)
    spawned = (final_state.survival_ticks > 0) | (final_state.agent_energy > 0)
    per_agent_fit = per_agent_fit * spawned.astype(jnp.float32)
    species_fit = jnp.sum(one_hot * per_agent_fit[:, None], axis=0)
    species_count = jnp.sum(one_hot * spawned[:, None].astype(jnp.float32), axis=0)
    species_fit_mean = species_fit / (species_count + 1e-3)

    # Coordination metrics across all agents in this episode
    total_feast = jnp.sum(feast)
    total_telepathy = jnp.sum(telepathy)
    total_regular = jnp.sum(earned) - total_feast  # earned includes both
    coord_metrics = jnp.stack([total_feast, total_telepathy, total_regular])
    return species_fit_mean, coord_metrics


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


def evaluate_member(flat_params, params_template_single, apply_fn, env_keys,
                    episode_keys, grid_size, arena_h, arena_w, num_active,
                    telepathy_strength):
    """Evaluate one member.
    Returns (species_fit, coord_metrics):
      species_fit: (NUM_POLICY_SPECIES,) — averaged across NUM_ENVS_PER_MEMBER envs
      coord_metrics: (3,) — averaged [total_feast, total_telepathy, total_regular]
    """
    params_per_species = flat_params.shape[0] // NUM_POLICY_SPECIES
    species_params = flat_params.reshape(NUM_POLICY_SPECIES, params_per_species)

    def unflatten_species(flat_sp):
        return unflatten_params(flat_sp, params_template_single)

    all_params = vmap(unflatten_species)(species_params)

    def eval_one_env(ek, pk):
        env_state = init_env(ek, arena_h, arena_w, grid_size, num_active)
        final_state = run_episode(all_params, apply_fn, env_state, grid_size,
                                    pk, telepathy_strength)
        return compute_per_species_fitness(final_state)

    species_fits, coord_metrics = vmap(eval_one_env)(env_keys, episode_keys)
    # species_fits: (NUM_ENVS, NUM_SPECIES); coord_metrics: (NUM_ENVS, 3)
    return jnp.mean(species_fits, axis=0), jnp.mean(coord_metrics, axis=0)


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
# 11. TRAINING LOOP
# ==========================================

def main():
    print("=" * 60)
    print("  BYTE-MULTI-AGENT: COORDINATION EDITION (v3.2)")
    print("=" * 60)

    num_devices = jax.device_count()
    print(f"Using {num_devices} device(s)")
    assert POP_SIZE % num_devices == 0, \
        f"POP_SIZE ({POP_SIZE}) must be divisible by num_devices ({num_devices})"
    assert (POP_SIZE // 2) % num_devices == 0, \
        f"half_pop must be divisible by num_devices"

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), axis_names=('pop',))

    key = random.PRNGKey(42)
    net = AgentNet()
    k1, key = random.split(key)

    dummy_obs = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_audio = jnp.zeros((MAX_AGENTS * AUDIO_PERSIST_TICKS,), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
    dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)

    params_template_single = net.init(k1, dummy_obs, dummy_audio, dummy_proprio, dummy_hidden)
    apply_fn = net.apply

    single_flat = flatten_params(params_template_single)
    num_params_single = single_flat.shape[0]
    num_params_total = num_params_single * NUM_POLICY_SPECIES
    print(f"Params per species: {num_params_single:,}")
    print(f"Num species: {NUM_POLICY_SPECIES} (with distinct traits)")
    print(f"Total params optimized: {num_params_total:,}")

    k_inits = random.split(key, NUM_POLICY_SPECIES + 1)
    key = k_inits[0]
    species_params = []
    for si in range(NUM_POLICY_SPECIES):
        sp = net.init(k_inits[si + 1], dummy_obs, dummy_audio, dummy_proprio, dummy_hidden)
        species_params.append(flatten_params(sp))
    center_flat = jnp.concatenate(species_params)

    CKPT_DIR = '/tmp' if os.path.exists('/tmp') else '.'
    CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v3_2_params.npy')
    GEN_CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_v3_2_gen.npy')

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
    per_device = POP_SIZE // num_devices

    print(f"Population: {POP_SIZE} mirrored, {NUM_ENVS_PER_MEMBER} envs each")
    print(f"Arena: {arena_h}x{arena_w}, grid {grid_size}, agents {num_active}")
    print(f"Per-device members: {per_device}")
    print(f"Telepathy bonus init: {TELEPATHY_BONUS_INIT}, anneal over {TELEPATHY_ANNEAL_GENS} gens")
    print()

    print("  Compiling sharded generation fn...")

    # Memory-efficient sharded eval: noise is generated INSIDE the sharded
    # function, per device, so we never materialize a (POP_SIZE, num_params)
    # tensor on a single device. Each device generates its own noise slice
    # using a shared base seed + per-device row offset, so we can still
    # reconstruct the full noise tensor host-side for the ES update.
    #
    # The reconstruction uses the same seed scheme: gen N has base seed
    # `noise_base_seed`, and row i of `full_noise` is the i-th row of
    # `random.normal(noise_base_seed, (half_pop, num_params))` for i < half_pop,
    # and the negation of row (i - half_pop) for i >= half_pop.

    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P(), P(), P('pop'), P('pop'), P()),
             out_specs=(P('pop'), P('pop')),
             check_rep=False)
    def sharded_eval(center, sigma_val, noise_base_seed,
                     env_keys_shard, ep_keys_shard, telepathy_strength):
        my_idx = lax.axis_index('pop')
        start = my_idx * per_device

        def gen_one_row(global_row_idx):
            is_neg = global_row_idx >= half_pop
            base_idx = jnp.where(is_neg, global_row_idx - half_pop, global_row_idx)
            row_key = random.fold_in(noise_base_seed, base_idx)
            row = random.normal(row_key, (num_params_total,))
            return jnp.where(is_neg, -row, row)

        my_global_indices = start + jnp.arange(per_device)
        my_noise = vmap(gen_one_row)(my_global_indices)

        all_members = center[None, :] + sigma_val * my_noise
        fits, coords = vmap(lambda p, eks, pks: evaluate_member(
            p, params_template_single, apply_fn, eks, pks,
            grid_size, arena_h, arena_w, num_active, telepathy_strength))(
            all_members, env_keys_shard, ep_keys_shard)
        return fits, coords  # (per_device, NUM_SPECIES), (per_device, 3)

    @jit
    def reconstruct_noise_for_update(noise_base_seed):
        """Host-side noise reconstruction for the ES update. Only the half_pop
        positive rows are needed (negative is implicit).
        Shape: (half_pop, num_params_total)
        """
        def gen_one_row(base_idx):
            row_key = random.fold_in(noise_base_seed, base_idx)
            return random.normal(row_key, (num_params_total,))
        return vmap(gen_one_row)(jnp.arange(half_pop))

    def es_update_per_species(center, noise, sigma_val, species_fitness, opt_state):
        """species_fitness: (POP_SIZE, NUM_SPECIES). Compute per-species gradient
        from each species' fitness column, then concatenate by species block.
        """
        params_per_species = center.shape[0] // NUM_POLICY_SPECIES
        # Reshape center and noise into (NUM_SPECIES, params_per_species)
        center_blocks = center.reshape(NUM_POLICY_SPECIES, params_per_species)
        noise_blocks = noise.reshape(noise.shape[0], NUM_POLICY_SPECIES, params_per_species)

        def per_species_grad(species_idx):
            # Use this species' fitness to compute gradient on this species' block
            fit = species_fitness[:, species_idx]  # (POP_SIZE,)
            utilities = rank_fitness_shape(fit)
            pos_util = utilities[:half_pop]
            neg_util = utilities[half_pop:]
            # Noise for this species' block
            noise_sp = noise_blocks[:half_pop, species_idx, :]  # (half_pop, params_per_species)
            grad_sp = jnp.dot((pos_util - neg_util), noise_sp) / (half_pop * sigma_val)
            grad_sp = grad_sp - WEIGHT_DECAY * center_blocks[species_idx]
            return grad_sp

        grads = vmap(per_species_grad)(jnp.arange(NUM_POLICY_SPECIES))
        grad_estimate = grads.reshape(-1)  # back to flat
        updates, new_opt_state = optimizer.update(-grad_estimate, opt_state, center)
        return optax.apply_updates(center, updates), new_opt_state

    es_update_jit = jit(es_update_per_species)

    start_time = time.time()
    last_print_time = start_time
    PRINT_INTERVAL = 120

    for gen in range(start_gen, 100000):
        k_noise, k_env, k_ep, key = random.split(key, 4)
        sigma = noise_ctrl.get_sigma()
        sigma_jnp = jnp.float32(sigma)

        # Anneal telepathy strength
        telepathy_strength = max(
            TELEPATHY_ANNEAL_FLOOR,
            TELEPATHY_BONUS_INIT * (1.0 - gen / TELEPATHY_ANNEAL_GENS)
        )
        telepathy_jnp = jnp.float32(telepathy_strength)

        # Noise base seed: each device will derive its own slice from this
        # via fold_in. We do NOT materialize the full noise tensor on host.
        noise_base_seed = k_noise

        env_keys = random.split(k_env, half_pop * NUM_ENVS_PER_MEMBER).reshape(
            half_pop, NUM_ENVS_PER_MEMBER, 2)
        env_keys_full = jnp.concatenate([env_keys, env_keys], axis=0)

        ep_keys = random.split(k_ep, half_pop * NUM_ENVS_PER_MEMBER).reshape(
            half_pop, NUM_ENVS_PER_MEMBER, 2)
        ep_keys_full = jnp.concatenate([ep_keys, ep_keys], axis=0)

        species_fitness, coord_metrics = sharded_eval(
            center_flat, sigma_jnp, noise_base_seed,
            env_keys_full, ep_keys_full, telepathy_jnp)
        # species_fitness: (POP_SIZE, NUM_SPECIES)
        # coord_metrics:   (POP_SIZE, 3) — [feast_total, telepathy_total, regular_food_total]

        # Reconstruct only the positive half of the noise for the ES update.
        noise = reconstruct_noise_for_update(noise_base_seed)

        center_flat, opt_state = es_update_jit(center_flat, noise, sigma_jnp,
                                                species_fitness, opt_state)

        # For monitoring: aggregate fitness = mean over species
        agg_fitness = jnp.mean(species_fitness, axis=1)  # (POP_SIZE,)
        mean_fit = float(jnp.mean(agg_fitness))
        max_fit = float(jnp.max(agg_fitness))
        min_fit = float(jnp.min(agg_fitness))
        pop_std = float(jnp.std(agg_fitness))
        species_means = jnp.mean(species_fitness, axis=0)  # (NUM_SPECIES,)

        noise_ctrl.record(mean_fit, pop_std)
        noise_ctrl.step()
        # Plateau detection (after warmup)
        if gen >= NOISE_WARMUP_GENS:
            noise_ctrl.maybe_plateau_reset(gen)

        if gen > 0 and gen % 50 == 0:
            np.save(CKPT_PATH, np.array(jax.device_get(center_flat)))
            np.save(GEN_CKPT_PATH, {'gen': gen, 'mean_fit': mean_fit,
                                     'noise_std': noise_ctrl.get_sigma()})
            print(f"  [Checkpoint saved at gen {gen}]")

        now = time.time()
        if now - last_print_time > PRINT_INTERVAL or gen == start_gen or gen % 25 == 0:
            elapsed = now - start_time
            print(f"\n--- GEN {gen} | {elapsed:.0f}s elapsed ---")
            print(f"  Aggregate fitness: mean={mean_fit:.2f} max={max_fit:.2f} "
                  f"min={min_fit:.2f} std={pop_std:.2f}")
            sp_means_np = np.array(jax.device_get(species_means))
            print(f"  Per-species mean fitness: " +
                  ", ".join(f"s{i}={sp_means_np[i]:.2f}" for i in range(NUM_POLICY_SPECIES)))
            print(f"  Noise:     {noise_ctrl.get_status_str()}")
            print(f"  Telepathy: strength={telepathy_strength:.3f}")

            # POPULATION-WIDE coordination stats (averaged across all 256 members × 4 envs)
            coord_np = np.array(jax.device_get(coord_metrics))  # (POP_SIZE, 3)
            pop_feast_avg = float(coord_np[:, 0].mean())
            pop_tele_avg = float(coord_np[:, 1].mean())
            pop_food_avg = float(coord_np[:, 2].mean())
            # Fraction of population members who got ANY feast bonus
            members_with_feast = float((coord_np[:, 0] > 0).mean())
            # Top 5% feast performers (encouragement signal: are SOME members
            # discovering feast even if average isn't?)
            top_feast = float(np.percentile(coord_np[:, 0], 95))
            print(f"  Pop coord (avg per episode): "
                  f"feast={pop_feast_avg:.1f}, telepathy={pop_tele_avg:.1f}, "
                  f"regular_food={pop_food_avg:.1f}")
            print(f"  Feast adoption: {members_with_feast*100:.1f}% of members "
                  f"got >0 feast | top-5%-member feast={top_feast:.1f}")

            # Render preview
            k_render, key = random.split(key)
            params_per_sp = num_params_single
            sp_flat = center_flat.reshape(NUM_POLICY_SPECIES, params_per_sp)
            preview_params = vmap(lambda s: unflatten_params(s, params_template_single))(sp_flat)
            render_env = init_env(k_render, arena_h, arena_w, grid_size, num_active)

            def preview_step(carry, sks):
                state, last_speak = carry
                sk_act, sk_obs = sks[0], sks[1]
                obs, audio = get_all_obs(state, grid_size, last_speak, sk_obs)
                proprio = build_proprioception(state)
                actions, new_hidden = sample_actions(
                    preview_params, obs, audio, proprio, state.agent_hidden,
                    state.agent_species, sk_act, apply_fn, ACTION_TEMP)
                state = state._replace(agent_hidden=new_hidden)
                new_state = step_env(state, actions, grid_size, telepathy_jnp)
                return (new_state, actions[:, 1]), actions[:, 1]

            preview_keys = random.split(k_render, 200 * 2).reshape(200, 2, 2)
            (mid_state, _), speech_log = lax.scan(preview_step,
                (render_env, jnp.zeros(MAX_AGENTS, dtype=jnp.int32)),
                preview_keys)

            grid_np = np.array(jax.device_get(mid_state.grid))
            pos_np = np.array(jax.device_get(mid_state.agent_pos))
            alive_np = np.array(jax.device_get(mid_state.agent_alive))
            energy_np = np.array(jax.device_get(mid_state.agent_energy))
            species_np = np.array(jax.device_get(mid_state.agent_species))
            speech_np = np.array(jax.device_get(speech_log))
            feast_eaters_np = np.array(jax.device_get(mid_state.feast_eaters))
            telepathy_np = np.array(jax.device_get(mid_state.telepathy_credit))
            energy_earned_np = np.array(jax.device_get(mid_state.energy_earned))

            print(render_snapshot(grid_np, arena_h, arena_w, grid_size,
                                   pos_np, alive_np, energy_np, species_np))

            # Diagnostics
            has_writing = np.any((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET))
            if has_writing:
                wc = int(np.sum((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET)))
                print(f"  WRITING: {wc} cells on ground")
            has_tool = np.any((grid_np >= TOOL_START) & (grid_np < TOOL_END))
            if has_tool:
                tc = int(np.sum((grid_np >= TOOL_START) & (grid_np < TOOL_END)))
                print(f"  TOOLS ON GROUND: {tc}")
            has_feast = np.any(grid_np == FEAST)
            if has_feast:
                fc = int(np.sum(grid_np == FEAST))
                print(f"  FEAST tiles remaining: {fc}")

            total_speech = int((speech_np > 0).sum())
            if total_speech > 0:
                uniq = len(np.unique(speech_np[speech_np > 0]))
                print(f"  Speech: {total_speech} utterances across preview, "
                      f"{uniq} unique tokens used")

            feast_total = float(feast_eaters_np.sum())
            tele_total = float(telepathy_np.sum())
            energy_earned_total = float(energy_earned_np.sum())
            # Regular food = total energy earned minus feast contribution
            regular_food = max(0.0, energy_earned_total - feast_total)
            # Each feast event yields FEAST_ENERGY_PER_AGENT × 2 (both partners)
            estimated_feast_events = feast_total / (2 * FEAST_ENERGY_PER_AGENT)
            print(f"  Energy sources: regular_food={regular_food:.1f}, "
                  f"feast={feast_total:.1f} ({estimated_feast_events:.1f} pair-events), "
                  f"telepathy={tele_total:.1f}")

            last_print_time = now

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
