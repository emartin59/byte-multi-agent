# %% [markdown]
# # Byte-Multi-Agent: Economy Edition
#
# A minimal 2D byte-grid multi-agent environment for studying emergent
# cooperation, communication, and economics under survival pressure.
# All state (world, agents, communication) is encoded as bytes on a single grid,
# with JAX/Flax for fast TPU-accelerated training via Evolutionary Strategies.
#
# Features:
# - Energy economy (agents have energy, lose it per tick, die at 0)
# - Food and seeds (food is consumed, leaves seed, seed regrows to food)
# - Water tiles (static, used for crafting)
# - Rocks (impassable, used for crafting)
# - Inventory (4 slots per agent, not visible on grid)
# - Crafting (rock adjacent to water -> fertilizer tool, probabilistically)
# - Tools (fertilizer: spreads seeds on adjacent cells when used)
# - Action costs: move 0.5, write 1.0, overwrite 1.5, speak 0.5, idle 0.1,
#   pickup 0.5, use_tool 0.5
#
# Byte layout:
#   0   EMPTY
#   1   WALL
#   2   FOOD
#   3   SEED
#   4   ROCK
#   5   WATER
#   6-31  persistent writing a-z (26 bytes)
#   32-57 transient speech A-Z (26 bytes)
#   58  AGENT marker (@)
#   59  FERTILIZER tool
#   60-63 reserved for future tools
#
# Display symbols:
#   . empty, # wall, * food, , seed, o rock, ~ water
#   @ agent (silent), A-Z agent speaking that letter
#   a-z persistent writing
#   F fertilizer tool (displayed as F when on ground)
#
# When an agent stands on a writing cell, the display shows @ or speech letter,
# but the underlying byte is preserved in the agent's "underfoot" slot and
# restored when the agent leaves.

# %%
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
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
SPEAK_OFFSET = 32     # 32-57: A-Z transient speech
AGENT_MARK = 58       # @ on display; used for rendering, not stored on grid
FERTILIZER = 59       # first tool
# 60-63 reserved for future tools

NUM_BYTE_TYPES = 64  # keep room for more tools

# Tool range — anything in [TOOL_START, TOOL_END) is a tool that can be picked up
TOOL_START = 59
TOOL_END = 64

# Character map for rendering
CHAR_MAP = {
    EMPTY: '.',
    WALL: '#',
    FOOD: '*',
    SEED: ',',
    ROCK: 'o',
    WATER: '~',
    AGENT_MARK: '@',
    FERTILIZER: 'F',
}
for i in range(26):
    CHAR_MAP[i + WRITE_OFFSET] = chr(97 + i)   # a-z
    CHAR_MAP[i + SPEAK_OFFSET] = chr(65 + i)   # A-Z

# --- Environment ---
VISION_RADIUS = 7
VISION_SIZE = 2 * VISION_RADIUS + 1  # 15
MAX_AGENTS = 8
EPISODE_LENGTH = 300  # longer than phase 1 — agents need time to eat, craft, trade
PREP_TICKS = 40
INVENTORY_SIZE = 4

# Energy / economy constants
INITIAL_ENERGY = 50.0
MAX_ENERGY = 100.0
FOOD_ENERGY = 20.0       # eating food restores this much energy
METABOLIC_COST = 0.1     # idle/per-tick cost
MOVE_COST = 0.5
WRITE_COST = 1.0
OVERWRITE_COST = 1.5
SPEAK_COST = 0.5
PICKUP_COST = 0.5
USE_TOOL_COST = 0.5

# Ecology / spawning rates
SEED_TO_FOOD_PROB = 0.005       # per-tick prob a seed grows into food
INITIAL_FOOD_DENSITY = 0.06
INITIAL_SEED_DENSITY = 0.04
INITIAL_ROCK_DENSITY = 0.05
INITIAL_WATER_DENSITY = 0.04   # bumped from 0.03 so rocks more often end up adjacent to water

# Crafting
ROCK_WATER_CRAFT_PROB = 0.03    # per-tick prob that rock adjacent to water crafts fertilizer
# (placed on the rock's cell; rock is consumed)

# --- ES Hyperparameters ---
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

# --- Arena ---
INITIAL_ARENA = 20
INITIAL_ACTIVE = 4

# Movement directions: stay, up, down, right, left
MOVE_DIRS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)


# ==========================================
# 2. ADAPTIVE NOISE CONTROLLER
# ==========================================

class AdaptiveNoiseController:
    """Same controller as Phase 1: SNR-based sigma adaptation with ratchet."""

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
                    print(f"  >> RATCHET TIMEOUT after {self.ratchet_consecutive} gens. "
                          f"Resetting baseline from {self.best_smoothed:.1f} to {self.smoothed_mean:.1f}")
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
# 3. ENVIRONMENT STATE
# ==========================================

class EnvState(NamedTuple):
    grid: jnp.ndarray           # (grid_size, grid_size) - surface view, agents not rendered here
    agent_pos: jnp.ndarray      # (MAX_AGENTS, 2)
    agent_alive: jnp.ndarray    # (MAX_AGENTS,) bool
    agent_energy: jnp.ndarray   # (MAX_AGENTS,) float
    agent_inventory: jnp.ndarray  # (MAX_AGENTS, INVENTORY_SIZE) int32 byte slots
    agent_underfoot: jnp.ndarray  # (MAX_AGENTS,) int32 — byte currently hidden under the agent
    tick: jnp.ndarray           # scalar
    rng: jnp.ndarray            # (2,) — per-env rng for stochastic events


def init_env(key, arena_h, arena_w, grid_size, num_active):
    """Initialize one environment with food, seeds, rocks, water, agents."""
    k_walls, k_blocks, k_agents, k_rng = random.split(key, 4)

    # Build arena walls
    rows = jnp.arange(grid_size)[:, None]
    cols = jnp.arange(grid_size)[None, :]
    off_r = (grid_size - arena_h) // 2
    off_c = (grid_size - arena_w) // 2
    in_arena = (rows >= off_r) & (rows < off_r + arena_h) & (cols >= off_c) & (cols < off_c + arena_w)
    on_border = (rows == off_r) | (rows == off_r + arena_h - 1) | (cols == off_c) | (cols == off_c + arena_w - 1)
    playable = in_arena & ~on_border
    grid = jnp.where(in_arena, jnp.where(on_border, WALL, EMPTY), WALL)

    # Scatter resources using a single noise field; each threshold carves out a band
    k1, k2, k3, k4 = random.split(k_blocks, 4)
    food_noise = random.uniform(k1, (grid_size, grid_size))
    seed_noise = random.uniform(k2, (grid_size, grid_size))
    rock_noise = random.uniform(k3, (grid_size, grid_size))
    water_noise = random.uniform(k4, (grid_size, grid_size))

    grid = jnp.where(playable & (food_noise < INITIAL_FOOD_DENSITY), FOOD, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (seed_noise < INITIAL_SEED_DENSITY), SEED, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (rock_noise < INITIAL_ROCK_DENSITY), ROCK, grid)
    grid = jnp.where(playable & (grid == EMPTY) & (water_noise < INITIAL_WATER_DENSITY), WATER, grid)

    # Place agents on empty cells using Gumbel top-k
    gumbel = random.gumbel(k_agents, shape=(grid_size * grid_size,))
    is_empty = (grid.reshape(-1) == EMPTY)
    gumbel = jnp.where(is_empty, gumbel, -1e9)
    top_idx = jnp.argsort(-gumbel)[:MAX_AGENTS]
    agent_pos = jnp.stack([top_idx // grid_size, top_idx % grid_size], axis=-1)
    agent_alive = jnp.arange(MAX_AGENTS) < num_active
    agent_energy = jnp.where(agent_alive, INITIAL_ENERGY, 0.0)
    agent_inventory = jnp.zeros((MAX_AGENTS, INVENTORY_SIZE), dtype=jnp.int32)
    agent_underfoot = jnp.full((MAX_AGENTS,), EMPTY, dtype=jnp.int32)

    return EnvState(
        grid=grid,
        agent_pos=agent_pos,
        agent_alive=agent_alive,
        agent_energy=agent_energy,
        agent_inventory=agent_inventory,
        agent_underfoot=agent_underfoot,
        tick=jnp.int32(0),
        rng=k_rng,
    )


def render_grid_with_agents(state):
    """Produce a grid that includes agents at their positions for display/observation.
    Uses speech bytes if the agent is speaking, else AGENT_MARK.
    The underlying cell value is preserved in agent_underfoot.
    NOTE: This function assumes the caller has already composed speech bytes
    into a per-agent 'body_byte' array. For the observation path we just use AGENT_MARK.
    """
    grid = state.grid
    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        return jnp.where(state.agent_alive[i], g.at[r, c].set(AGENT_MARK), g), None
    grid, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    return grid


def render_grid_with_agents_and_speech(state, speak_acts):
    """Like render_grid_with_agents but uses speech bytes when agents are speaking."""
    grid = state.grid
    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        body = jnp.where(speak_acts[i] > 0, speak_acts[i] - 1 + SPEAK_OFFSET, AGENT_MARK)
        return jnp.where(state.agent_alive[i], g.at[r, c].set(body), g), None
    grid, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    return grid


def get_all_obs(state, grid_size, last_speak):
    """Observations: 15x15 vision crops centered on each agent.
    The visible grid includes all OTHER agents (using their current speech bytes or @)
    but does NOT show the observing agent itself — only the underlying byte of their cell.

    For simplicity, we render every agent's body byte into the grid before cropping.
    Each agent then sees the composed grid. This means an agent sees itself as AGENT_MARK,
    but it's in a fixed position (center of crop) so the network can easily learn to
    ignore or use that cue.

    last_speak: (MAX_AGENTS,) — what each agent spoke last tick (0 = silent).
    """
    composed = render_grid_with_agents_and_speech(state, last_speak)
    padded = jnp.pad(composed, VISION_RADIUS, constant_values=WALL)

    def get_one(pos):
        return lax.dynamic_slice(padded, (pos[0], pos[1]), (VISION_SIZE, VISION_SIZE))

    obs_bytes = vmap(get_one)(state.agent_pos)  # (MAX_AGENTS, V, V) int32
    obs_norm = obs_bytes.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)
    return obs_norm


# ==========================================
# 4. ACTION SEMANTICS
# ==========================================
# Actions = (move, speak, write, pickup_use)
#   move:       0-4   stay/up/down/right/left
#   speak:      0-26  silent or speak letter 1-26
#   write:      0-26  no-write or write letter 1-26
#   pickup_use: 0-3   noop / pickup adjacent tool / use slot0 tool / drop slot0
#
# This replaces the old (move, speak, write, lock) action set. Lock is removed
# for Phase 2 because it conflated with too many other mechanics; it can return later.


def is_passable_byte(b):
    """A cell is passable if it's empty, a writing byte, a seed, or food
    (food is eaten on step-on, seed can be walked over)."""
    return (b == EMPTY) | (b == FOOD) | (b == SEED) | \
           ((b >= WRITE_OFFSET) & (b < SPEAK_OFFSET))


def step_env(state, actions, grid_size):
    """
    One physics tick.
    actions: (MAX_AGENTS, 4) — [move, speak, write, pickup_use]
    Returns new state and per-agent events dict flattened into a tuple for JAX compatibility.
    """
    move_acts = actions[:, 0]
    speak_acts = actions[:, 1]
    write_acts = actions[:, 2]
    puu_acts = actions[:, 3]   # pickup / use / drop

    curr_pos = state.agent_pos
    dp = MOVE_DIRS[move_acts]
    want_pos = curr_pos + dp
    want_pos = jnp.where(state.agent_alive[:, None], want_pos, curr_pos)
    want_pos = jnp.clip(want_pos, 0, grid_size - 1)

    # --- Movement ---
    target_vals = state.grid[want_pos[:, 0], want_pos[:, 1]]
    passable = is_passable_byte(target_vals) & state.agent_alive

    # Agent-agent collision: no two agents can enter the same cell
    same_target = jnp.all(want_pos[:, None, :] == want_pos[None, :, :], axis=-1)
    eye = jnp.eye(MAX_AGENTS, dtype=jnp.bool_)
    has_conflict = jnp.any(same_target & passable[:, None] & passable[None, :] & ~eye, axis=1)

    # Also can't walk into a cell currently occupied by another agent (non-swapping)
    other_pos_mask = jnp.zeros((MAX_AGENTS,), dtype=jnp.bool_)
    def check_occupied(i):
        # Is any OTHER alive agent currently at want_pos[i]?
        same = jnp.all(curr_pos == want_pos[i][None, :], axis=-1) & state.agent_alive
        same = same & (jnp.arange(MAX_AGENTS) != i)
        return jnp.any(same)
    other_pos_mask = vmap(check_occupied)(jnp.arange(MAX_AGENTS))

    final_move = passable & ~has_conflict & ~other_pos_mask
    final_pos = jnp.where(final_move[:, None], want_pos, curr_pos)
    moved = final_move & jnp.any(dp != 0, axis=-1)

    # --- Eating food / walking on seed ---
    # If the target cell was FOOD, consume it (agent gains energy, cell becomes SEED
    # so that eating produces a seed — realistic and self-balancing)
    ate_food = moved & (target_vals == FOOD)
    # If target was SEED, just walk over it (no energy gain, seed remains beneath)

    # --- Update underfoot ---
    # When an agent moves to a new cell, the new underfoot is the value that was there
    # BEFORE the agent arrived. If the agent ate food, underfoot becomes SEED.
    new_underfoot_moved = jnp.where(ate_food, SEED, target_vals)
    # Writing cells and seeds are preserved under the agent; food becomes seed after eating
    new_underfoot = jnp.where(moved, new_underfoot_moved, state.agent_underfoot)

    # --- Writing ---
    # When an agent leaves a cell, it may leave a writing byte behind.
    # write_acts: 0 = no-write, 1..26 = write letter a..z
    # The cell the agent is LEAVING (curr_pos) should take the agent's OLD underfoot
    # (what was there before they arrived), OR the new write byte if they chose to write.
    # Overwriting a non-empty cell costs more energy.

    leave_byte = jnp.where(write_acts > 0, write_acts - 1 + WRITE_OFFSET, state.agent_underfoot)

    # Detect whether writing is "overwriting" (the leaving cell had something non-empty
    # beneath the agent). This determines energy cost.
    is_overwrite = (write_acts > 0) & (state.agent_underfoot != EMPTY) & \
                    (state.agent_underfoot != leave_byte)

    # --- Grid updates ---
    new_grid = state.grid

    # 1. Clear the cells the agents are leaving, depositing leave_byte
    def clear_old(g, i):
        r, c = curr_pos[i, 0], curr_pos[i, 1]
        return jnp.where(moved[i], g.at[r, c].set(leave_byte[i]), g), None
    new_grid, _ = lax.scan(clear_old, new_grid, jnp.arange(MAX_AGENTS))

    # 2. Consume food at the target cells (set to EMPTY; underfoot is SEED which
    # will restore itself when the agent leaves, so food -> seed realistically)
    def consume_food(g, i):
        r, c = final_pos[i, 0], final_pos[i, 1]
        return jnp.where(ate_food[i], g.at[r, c].set(EMPTY), g), None
    new_grid, _ = lax.scan(consume_food, new_grid, jnp.arange(MAX_AGENTS))

    # --- Pickup / Use / Drop ---
    # puu_acts: 0 noop, 1 pickup, 2 use slot 0, 3 drop slot 0
    #
    # Pickup: look at 4 neighbors for a tool byte; pick up the first one found, place
    # in first empty inventory slot.
    # Use: if slot 0 is FERTILIZER and there's a SEED in the inventory (slot 1, 2, or 3),
    # multiply that seed — FERTILIZER is consumed and the seed slot becomes "4 seeds worth"
    # which we model by just... we don't have a count system. Simpler: using FERTILIZER
    # while holding a SEED in any slot converts that seed into 4 FOOD drops on adjacent
    # cells if possible. Falling back to a simple alternative: FERTILIZER use while
    # standing on an empty cell places 4 SEEDs on the 4 neighboring empty cells.
    # That's easier to implement and gives a clear economic signal.
    # Drop: places slot 0's byte on an adjacent empty cell (or drops the action if none).

    new_inventory = state.agent_inventory
    new_energy = state.agent_energy

    def do_pickup_use_drop(carry, i):
        grid, inv, energy = carry
        act = puu_acts[i]
        alive = state.agent_alive[i]
        r, c = final_pos[i, 0], final_pos[i, 1]

        # ---- PICKUP (act==1) ----
        # Look at 4 neighbors; find first that contains a tool byte
        neighbors = jnp.array([[r-1, c], [r+1, c], [r, c-1], [r, c+1]])
        neighbors = jnp.clip(neighbors, 0, grid_size - 1)
        neigh_vals = grid[neighbors[:, 0], neighbors[:, 1]]
        is_tool = (neigh_vals >= TOOL_START) & (neigh_vals < TOOL_END)
        # Index of first neighbor with a tool (or -1 if none)
        any_tool = jnp.any(is_tool)
        first_tool_idx = jnp.argmax(is_tool)  # returns 0 if no match; guard with any_tool
        picked_byte = neigh_vals[first_tool_idx]

        # Find first empty inventory slot
        empty_slots = inv[i] == EMPTY
        has_empty = jnp.any(empty_slots)
        first_empty = jnp.argmax(empty_slots)

        can_pickup = (act == 1) & alive & any_tool & has_empty

        # Apply pickup: clear neighbor cell, set inventory slot
        target_n_r = neighbors[first_tool_idx, 0]
        target_n_c = neighbors[first_tool_idx, 1]
        grid = jnp.where(can_pickup, grid.at[target_n_r, target_n_c].set(EMPTY), grid)
        inv_i_new = jnp.where(can_pickup, inv[i].at[first_empty].set(picked_byte), inv[i])

        # ---- USE SLOT 0 (act==2) ----
        # If slot 0 is FERTILIZER: place SEED on up to 4 adjacent empty cells, consume fertilizer
        slot0 = inv_i_new[0]
        is_fertilizer = slot0 == FERTILIZER
        can_use = (act == 2) & alive & is_fertilizer

        def place_seed(g, nbr_idx):
            nr, nc = neighbors[nbr_idx, 0], neighbors[nbr_idx, 1]
            target = g[nr, nc]
            should_place = can_use & (target == EMPTY)
            return jnp.where(should_place, g.at[nr, nc].set(SEED), g), None

        grid, _ = lax.scan(place_seed, grid, jnp.arange(4))
        # Consume the fertilizer from slot 0
        inv_i_new = jnp.where(can_use, inv_i_new.at[0].set(EMPTY), inv_i_new)

        # ---- DROP SLOT 0 (act==3) ----
        # Place slot 0 byte on first empty adjacent cell
        empty_nbrs = neigh_vals == EMPTY
        any_empty_nbr = jnp.any(empty_nbrs)
        first_empty_nbr = jnp.argmax(empty_nbrs)
        drop_target_r = neighbors[first_empty_nbr, 0]
        drop_target_c = neighbors[first_empty_nbr, 1]
        slot0_current = inv_i_new[0]
        can_drop = (act == 3) & alive & any_empty_nbr & (slot0_current != EMPTY)
        grid = jnp.where(can_drop, grid.at[drop_target_r, drop_target_c].set(slot0_current), grid)
        inv_i_new = jnp.where(can_drop, inv_i_new.at[0].set(EMPTY), inv_i_new)

        # Energy cost
        action_cost = jnp.where(
            (act == 1), PICKUP_COST,
            jnp.where((act == 2) | (act == 3), USE_TOOL_COST, 0.0)
        )
        # Only charge if the action actually did something (or was attempted)
        did_act = (act > 0) & alive
        new_energy_i = jnp.where(did_act, energy[i] - action_cost, energy[i])

        inv = inv.at[i].set(inv_i_new)
        energy = energy.at[i].set(new_energy_i)
        return (grid, inv, energy), None

    (new_grid, new_inventory, new_energy), _ = lax.scan(
        do_pickup_use_drop, (new_grid, new_inventory, new_energy), jnp.arange(MAX_AGENTS))

    # --- Crafting: rock adjacent to water may become fertilizer ---
    # Use a single random draw per cell; cheap and JIT-friendly
    k_craft, new_rng = random.split(state.rng)
    craft_noise = random.uniform(k_craft, (grid_size, grid_size))

    # Detect rocks adjacent to water
    is_rock = (new_grid == ROCK)
    is_water = (new_grid == WATER)

    # Check 4-neighbor water adjacency via shifts
    water_up    = jnp.pad(is_water[1:, :],  ((0, 1), (0, 0)))
    water_down  = jnp.pad(is_water[:-1, :], ((1, 0), (0, 0)))
    water_left  = jnp.pad(is_water[:, 1:],  ((0, 0), (0, 1)))
    water_right = jnp.pad(is_water[:, :-1], ((0, 0), (1, 0)))
    rock_near_water = is_rock & (water_up | water_down | water_left | water_right)
    craft_trigger = rock_near_water & (craft_noise < ROCK_WATER_CRAFT_PROB)
    new_grid = jnp.where(craft_trigger, FERTILIZER, new_grid)

    # --- Seed -> Food regrowth ---
    k_regrow, new_rng = random.split(new_rng)
    regrow_noise = random.uniform(k_regrow, (grid_size, grid_size))
    is_seed = (new_grid == SEED)
    regrow_trigger = is_seed & (regrow_noise < SEED_TO_FOOD_PROB)
    new_grid = jnp.where(regrow_trigger, FOOD, new_grid)

    # --- Energy accounting ---
    # Base metabolic cost
    new_energy = new_energy - jnp.where(state.agent_alive, METABOLIC_COST, 0.0)

    # Move cost
    new_energy = new_energy - jnp.where(moved, MOVE_COST, 0.0)

    # Speak cost
    new_energy = new_energy - jnp.where((speak_acts > 0) & state.agent_alive, SPEAK_COST, 0.0)

    # Write cost (base)
    wrote_something = (write_acts > 0) & moved
    base_write = jnp.where(wrote_something, WRITE_COST, 0.0)
    extra_overwrite = jnp.where(is_overwrite, OVERWRITE_COST - WRITE_COST, 0.0)
    new_energy = new_energy - base_write - extra_overwrite

    # Eating food restores energy
    new_energy = new_energy + jnp.where(ate_food, FOOD_ENERGY, 0.0)

    # Clamp
    new_energy = jnp.clip(new_energy, 0.0, MAX_ENERGY)

    # Death: agents at 0 energy die
    new_alive = state.agent_alive & (new_energy > 0.0)

    return EnvState(
        grid=new_grid,
        agent_pos=final_pos,
        agent_alive=new_alive,
        agent_energy=new_energy,
        agent_inventory=new_inventory,
        agent_underfoot=new_underfoot,
        tick=state.tick + 1,
        rng=new_rng,
    )


# ==========================================
# 5. FITNESS
# ==========================================

def compute_episode_fitness(final_state, total_energy_earned):
    """Fitness = total energy earned across the episode - starvation penalty.

    total_energy_earned: cumulative food energy gained by each agent
    We reward: staying alive, eating, and reaching end-of-episode with energy.
    """
    alive_bonus = jnp.sum(final_state.agent_alive.astype(jnp.float32)) * 50.0
    total_energy = jnp.sum(final_state.agent_energy)
    earned = jnp.sum(total_energy_earned)
    fitness = alive_bonus + total_energy + earned * 0.5
    return fitness


# ==========================================
# 6. NETWORK
# ==========================================

class AgentNet(nn.Module):
    """CNN policy: vision (15x15) + proprioception (energy + inventory).
    Outputs action logits for move, speak, write, and pickup_use heads.
    """
    @nn.compact
    def __call__(self, obs, proprio):
        # obs: (VISION_SIZE, VISION_SIZE) float
        # proprio: (INVENTORY_SIZE + 1,) float — normalized energy + normalized inventory bytes
        x = obs[None, :, :, None]
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        # Inject proprioception
        p = proprio[None, :]
        x = jnp.concatenate([x, p], axis=-1)

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        move_logits = nn.Dense(5)(x)
        speak_logits = nn.Dense(27)(x)
        write_logits = nn.Dense(27)(x)
        puu_logits = nn.Dense(4)(x)

        return (move_logits.squeeze(0), speak_logits.squeeze(0),
                write_logits.squeeze(0), puu_logits.squeeze(0))


def build_proprioception(state):
    """Build (MAX_AGENTS, INVENTORY_SIZE + 1) proprioception tensor."""
    energy_norm = state.agent_energy / MAX_ENERGY
    inv_norm = state.agent_inventory.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)
    proprio = jnp.concatenate([energy_norm[:, None], inv_norm], axis=-1)
    return proprio


def get_deterministic_actions(params, obs_all, proprio_all, apply_fn):
    def act_one(obs, p):
        ml, sl, wl, puu = apply_fn(params, obs, p)
        return jnp.array([jnp.argmax(ml), jnp.argmax(sl), jnp.argmax(wl), jnp.argmax(puu)],
                         dtype=jnp.int32)
    return vmap(act_one)(obs_all, proprio_all)


# ==========================================
# 7. EPISODE RUNNER
# ==========================================

def run_episode(params, apply_fn, env_state, grid_size):
    """Run a full episode with deterministic actions."""
    def step_fn(carry, _):
        state, energy_earned, last_speak = carry
        obs = get_all_obs(state, grid_size, last_speak)
        proprio = build_proprioception(state)
        actions = get_deterministic_actions(params, obs, proprio, apply_fn)

        # Track energy gained from eating
        dp = MOVE_DIRS[actions[:, 0]]
        want = jnp.clip(state.agent_pos + dp, 0, grid_size - 1)
        target_vals = state.grid[want[:, 0], want[:, 1]]
        will_eat = (target_vals == FOOD) & state.agent_alive
        energy_gain_est = jnp.where(will_eat, FOOD_ENERGY, 0.0)

        new_state = step_env(state, actions, grid_size)
        new_energy_earned = energy_earned + energy_gain_est
        new_last_speak = actions[:, 1]
        return (new_state, new_energy_earned, new_last_speak), None

    init_earned = jnp.zeros(MAX_AGENTS)
    init_speak = jnp.zeros(MAX_AGENTS, dtype=jnp.int32)
    (final_state, total_earned, _), _ = lax.scan(
        step_fn, (env_state, init_earned, init_speak), None, length=EPISODE_LENGTH)
    return final_state, total_earned


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


def generate_noise(key, num_params, half_pop):
    return random.normal(key, shape=(half_pop, num_params))


def rank_fitness_shape(fitness):
    n = fitness.shape[0]
    ranks = jnp.argsort(jnp.argsort(-fitness)).astype(jnp.float32)
    log_util = jnp.maximum(0.0, jnp.log(n / 2.0 + 1.0) - jnp.log(ranks + 1.0))
    utilities = log_util / jnp.sum(log_util) - 1.0 / n
    return utilities


def es_generation(center_flat, noise, noise_std, apply_fn, params_template,
                  env_init_keys, grid_size, arena_h, arena_w, num_active):
    half_pop = noise.shape[0]
    pos_params = center_flat[None, :] + noise_std * noise
    neg_params = center_flat[None, :] - noise_std * noise
    all_params = jnp.concatenate([pos_params, neg_params], axis=0)

    def eval_one_member(flat_p, env_keys):
        params = unflatten_params(flat_p, params_template)
        def eval_one_env(env_key):
            env_state = init_env(env_key, arena_h, arena_w, grid_size, num_active)
            final_state, total_earned = run_episode(params, apply_fn, env_state, grid_size)
            return compute_episode_fitness(final_state, total_earned)
        fitnesses = vmap(eval_one_env)(env_keys)
        return jnp.mean(fitnesses)

    return vmap(eval_one_member)(all_params, env_init_keys)


def es_update(center_flat, noise, noise_std, fitness, optimizer_state, optimizer):
    half_pop = noise.shape[0]
    utilities = rank_fitness_shape(fitness)
    pos_util = utilities[:half_pop]
    neg_util = utilities[half_pop:]
    grad_estimate = jnp.dot((pos_util - neg_util), noise) / (half_pop * noise_std)
    grad_estimate = grad_estimate - WEIGHT_DECAY * center_flat
    neg_grad = -grad_estimate
    updates, new_opt_state = optimizer.update(neg_grad, optimizer_state, center_flat)
    new_center = optax.apply_updates(center_flat, updates)
    return new_center, new_opt_state


# ==========================================
# 9. RENDERING
# ==========================================

def render_snapshot(grid_np, arena_h, arena_w, grid_size, agent_pos_np, agent_alive_np, agent_energy_np):
    """Render a snapshot including agents overlaid at their positions."""
    off_r = (grid_size - arena_h) // 2
    off_c = (grid_size - arena_w) // 2

    # Copy the grid and overlay agents
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
    stat = f"  alive={int(agent_alive_np.sum())}/{MAX_AGENTS}  energy=[{agent_energy_np.min():.0f},{agent_energy_np.max():.0f}] mean={agent_energy_np.mean():.0f}"
    return "\n".join(lines) + "\n" + stat


# ==========================================
# 10. MAIN TRAINING LOOP
# ==========================================

def main():
    print("=" * 60)
    print("  BYTE-MULTI-AGENT: ECONOMY EDITION")
    print("=" * 60)

    key = random.PRNGKey(42)
    net = AgentNet()
    k1, key = random.split(key)
    dummy_obs = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((INVENTORY_SIZE + 1,), dtype=jnp.float32)
    params_template = net.init(k1, dummy_obs, dummy_proprio)
    apply_fn = net.apply

    center_flat = flatten_params(params_template)
    num_params = center_flat.shape[0]
    print(f"Network parameters: {num_params:,}")

    CKPT_DIR = '/kaggle/working' if os.path.exists('/kaggle') else '.'
    CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_params.npy')
    GEN_CKPT_PATH = os.path.join(CKPT_DIR, 'byte_multi_agent_gen.npy')

    start_gen = 0
    if os.path.exists(CKPT_PATH):
        center_flat = jnp.array(np.load(CKPT_PATH))
        print(f"  Resumed from {CKPT_PATH}")
        if os.path.exists(GEN_CKPT_PATH):
            gen_info = np.load(GEN_CKPT_PATH, allow_pickle=True).item()
            start_gen = int(gen_info.get('gen', 0))
            print(f"  Resuming from generation {start_gen}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LR),
    )
    opt_state = optimizer.init(center_flat)
    noise_ctrl = AdaptiveNoiseController()
    half_pop = POP_SIZE // 2

    arena_h = INITIAL_ARENA
    arena_w = INITIAL_ARENA
    grid_size = 64
    num_active = INITIAL_ACTIVE

    print(f"Population: {POP_SIZE} (mirrored), {NUM_ENVS_PER_MEMBER} envs each")
    print(f"Arena: {arena_h}x{arena_w}, grid {grid_size}, agents {num_active}")
    print(f"Adaptive noise: init={NOISE_STD_INIT}, range=[{NOISE_STD_MIN}, {NOISE_STD_MAX}]")
    print()

    print("  Compiling...")
    @jit
    def compiled_gen_fn(center, noise, noise_std_val, env_keys):
        return es_generation(center, noise, noise_std_val, apply_fn, params_template,
                             env_keys, grid_size, arena_h, arena_w, num_active)

    start_time = time.time()
    last_print_time = start_time
    PRINT_INTERVAL = 120

    for gen in range(start_gen, 100000):
        k_noise, k_env, key = random.split(key, 3)
        current_sigma = noise_ctrl.get_sigma()
        current_sigma_jnp = jnp.float32(current_sigma)

        noise = generate_noise(k_noise, num_params, half_pop)
        env_keys = random.split(k_env, POP_SIZE * NUM_ENVS_PER_MEMBER).reshape(
            POP_SIZE, NUM_ENVS_PER_MEMBER, 2)

        fitness = compiled_gen_fn(center_flat, noise, current_sigma_jnp, env_keys)
        center_flat, opt_state = es_update(center_flat, noise, current_sigma_jnp,
                                            fitness, opt_state, optimizer)

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
            print(f"  Fitness: mean={mean_fit:.3f}, max={max_fit:.3f}, min={min_fit:.3f}, std={pop_std:.3f}")
            print(f"  Noise:   {noise_ctrl.get_status_str()}")

            # Render preview
            preview_params = unflatten_params(center_flat, params_template)
            k_render, key = random.split(key)
            render_env = init_env(k_render, arena_h, arena_w, grid_size, num_active)

            def preview_step(carry, _):
                state, last_speak = carry
                obs = get_all_obs(state, grid_size, last_speak)
                proprio = build_proprioception(state)
                actions = get_deterministic_actions(preview_params, obs, proprio, apply_fn)
                new_state = step_env(state, actions, grid_size)
                return (new_state, actions[:, 1]), None

            mid_state, _ = lax.scan(preview_step,
                                     (render_env, jnp.zeros(MAX_AGENTS, dtype=jnp.int32)),
                                     None, length=150)
            final_preview = mid_state[0]

            grid_np = np.array(jax.device_get(final_preview.grid))
            pos_np = np.array(jax.device_get(final_preview.agent_pos))
            alive_np = np.array(jax.device_get(final_preview.agent_alive))
            energy_np = np.array(jax.device_get(final_preview.agent_energy))

            print(render_snapshot(grid_np, arena_h, arena_w, grid_size,
                                   pos_np, alive_np, energy_np))

            has_writing = np.any((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET))
            if has_writing:
                wc = int(np.sum((grid_np >= WRITE_OFFSET) & (grid_np < SPEAK_OFFSET)))
                print(f"  *** WRITING: {wc} cells ***")
            has_tool = np.any((grid_np >= TOOL_START) & (grid_np < TOOL_END))
            if has_tool:
                tc = int(np.sum((grid_np >= TOOL_START) & (grid_np < TOOL_END)))
                print(f"  *** TOOLS ON GROUND: {tc} ***")

            last_print_time = now

    print("\nTraining complete.")
    return center_flat, params_template


if __name__ == "__main__":
    main()