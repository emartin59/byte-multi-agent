# %% [markdown]
# # Byte-Multi-Agent v5.3: Split-Network Lewis Game
#
# Core insight from v5.0-v5.2 runs: ES parameter-space noise at any magnitude
# that allows exploration also corrupts the distilled speaker code. The fix
# is to stop trying to perturb speaker and listener with the same noise.
#
# v5.3 uses TWO separate parameter sets:
#   - speaker_params: FROZEN after distillation. No ES noise applied.
#   - listener_params: EVOLVED by ES. Full perturbation for exploration.
#
# Agents route to speaker_params vs listener_params based on role.
#
# This matches how real emergent-communication papers structure the problem:
# fix one side's encoder, let the other side learn to decode.

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
# CURRICULUM
# ==========================================

STAGES = [
    {
        "name": "stage0_lewis",
        "arena": 8,
        "num_agents": 2,
        "num_speakers": 1,
        "allow_speaker_move": False,
        "num_food_positions": 4,
        "vocab_size": 10,
        "episode_length": 30,
        "graduation_eat_rate": 0.70,    # was 0.80 — v5.5 peaked at 0.787 then oscillated
        "graduation_mi_ratio": 0.72,    # was 0.80
        "max_gens": 800,
        "description": "Pure Lewis game: speaker fixed, listener must comprehend.",
    },
    {
        "name": "stage1_more_positions",
        "arena": 10,
        "num_agents": 2,
        "num_speakers": 1,
        "allow_speaker_move": False,
        "num_food_positions": 9,
        "vocab_size": 12,
        "episode_length": 40,
        "graduation_eat_rate": 0.55,    # was 0.75 — more positions is harder
        "graduation_mi_ratio": 0.60,    # was 0.75
        "max_gens": 1000,
        "description": "9 food positions. Listener must decode richer vocabulary.",
    },
    {
        "name": "stage2_both_move",
        "arena": 12,
        "num_agents": 2,
        "num_speakers": 1,
        "allow_speaker_move": True,
        "num_food_positions": 12,
        "vocab_size": 14,
        "episode_length": 60,
        "graduation_eat_rate": 0.50,
        "graduation_mi_ratio": 0.55,
        "max_gens": 1200,
        "description": "Speaker can move. Tests whether listener tracks a moving speaker.",
    },
    {
        "name": "stage3_multi_agent",
        "arena": 14,
        "num_agents": 3,
        "num_speakers": 1,
        "allow_speaker_move": True,
        "num_food_positions": 16,
        "vocab_size": 16,
        "episode_length": 80,
        "graduation_eat_rate": 0.45,
        "graduation_mi_ratio": 0.50,
        "max_gens": 1500,
        "description": "Three agents, one speaker. Multi-recipient signaling.",
    },
    {
        "name": "stage4_bigger_world",
        "arena": 18,
        "num_agents": 4,
        "num_speakers": 2,
        "allow_speaker_move": True,
        "num_food_positions": 20,
        "vocab_size": 20,
        "episode_length": 120,
        "graduation_eat_rate": 0.40,
        "graduation_mi_ratio": 0.40,
        "max_gens": 2000,
        "description": "Multiple speakers. Approach byte-multi-agent scale.",
    },
]


# ==========================================
# CONSTANTS
# ==========================================

EMPTY = 0
WALL = 1
FOOD = 2
AGENT_MARK = 3
NUM_BYTE_TYPES = 8

HIDDEN_SIZE = 64
VISION_RADIUS = 3
VISION_SIZE = 2 * VISION_RADIUS + 1

MAX_VOCAB_SIZE = max(s["vocab_size"] for s in STAGES)
NUM_MOVE_ACTIONS = 5
NUM_SPEAK_ACTIONS = MAX_VOCAB_SIZE + 1

MAX_AGENTS = max(s["num_agents"] for s in STAGES)
MAX_ARENA = max(s["arena"] for s in STAGES)
MAX_FOOD_POSITIONS = max(s["num_food_positions"] for s in STAGES)

REACH_FOOD_BONUS = 50.0
STEP_CLOSER_REWARD = 0.5
STEP_AWAY_COST = 0.3
TICK_COST = 0.05
SPEAKER_FITNESS_SHARE = 0.5

POP_SIZE = 128
NUM_TRIBES = 4
NUM_ENVS_PER_MEMBER = 16        # up from 8 — cleaner fitness signal
TOTAL_POP = NUM_TRIBES * POP_SIZE

# Listener-only noise: aggressive because we can't corrupt a frozen speaker.
NOISE_STD_INIT = 0.035          # up from 0.025 — wider exploration at start
NOISE_STD_MIN = 0.008           # up from 0.005
NOISE_STD_MAX = 0.06            # up from 0.05
LR = 0.025                      # up from 0.015 — faster updates
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 1.0
ACTION_TEMP = 0.5               # down from 0.7 — sharper action selection

PBT_INTERVAL = 30
PBT_MUTATION_SIGMA = 0.03

TOP_K_PRESERVE = 8

# Snapshot / early-graduation
SNAPSHOT_EVERY_GENS = 50        # save best-so-far listener every N gens
PLATEAU_WINDOW = 80             # gens of history to check for plateau
PLATEAU_STD_THRESHOLD = 0.03    # if eat_rate stdev over window < this, plateaued
# If we plateau with eat_rate within this fraction of graduation target,
# graduate anyway. E.g. 0.85 means plateau at 85%+ of threshold counts.
PLATEAU_GRADUATION_FRACTION = 0.90

TEACHER_TRAIN_STEPS = 3000
TEACHER_BATCH_SIZE = 128
TEACHER_LR = 5e-4
TEACHER_EPISODES = 100


# ==========================================
# ENVIRONMENT STATE
# ==========================================

class EnvState(NamedTuple):
    grid: jnp.ndarray
    agent_pos: jnp.ndarray
    agent_alive: jnp.ndarray
    agent_role: jnp.ndarray
    agent_hidden: jnp.ndarray
    last_speak: jnp.ndarray
    food_pos: jnp.ndarray
    food_slot_id: jnp.ndarray
    reached_food: jnp.ndarray
    reward_accum: jnp.ndarray
    speaker_token_log: jnp.ndarray
    tick: jnp.ndarray
    rng: jnp.ndarray


MOVE_DIRS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)


def get_food_slot_positions(arena, num_positions):
    if num_positions == 4:
        return jnp.array([
            [1, 1], [1, arena - 2],
            [arena - 2, 1], [arena - 2, arena - 2]
        ], dtype=jnp.int32)
    interior = arena - 2
    side = int(np.ceil(np.sqrt(num_positions)))
    positions = []
    for i in range(num_positions):
        row = 1 + (i // side) * max(1, interior // max(1, side))
        col = 1 + (i % side) * max(1, interior // max(1, side))
        row = min(row, arena - 2)
        col = min(col, arena - 2)
        positions.append([row, col])
    return jnp.array(positions, dtype=jnp.int32)


def init_env(key, stage_cfg, food_positions):
    arena = stage_cfg["arena"]
    num_agents = stage_cfg["num_agents"]
    num_speakers = stage_cfg["num_speakers"]
    num_food_positions = stage_cfg["num_food_positions"]
    ep_len = stage_cfg["episode_length"]

    k_food, k_agents, k_rng = random.split(key, 3)

    rows = jnp.arange(MAX_ARENA)[:, None]
    cols = jnp.arange(MAX_ARENA)[None, :]
    in_arena = (rows < arena) & (cols < arena)
    on_border = (rows == 0) | (rows == arena - 1) | (cols == 0) | (cols == arena - 1) | ~in_arena
    grid = jnp.where(on_border, WALL, EMPTY)

    food_slot = random.randint(k_food, (), 0, num_food_positions)
    food_pos = food_positions[food_slot]
    grid = grid.at[food_pos[0], food_pos[1]].set(FOOD)

    flat_noise = random.gumbel(k_agents, shape=(MAX_ARENA * MAX_ARENA,))
    grid_flat = grid.reshape(-1)
    mask = (grid_flat == EMPTY)
    masked_noise = jnp.where(mask, flat_noise, -1e9)
    top_idx = jnp.argsort(-masked_noise)[:MAX_AGENTS]
    agent_pos = jnp.stack([top_idx // MAX_ARENA, top_idx % MAX_ARENA], axis=-1)

    agent_alive = jnp.arange(MAX_AGENTS) < num_agents
    agent_role = (jnp.arange(MAX_AGENTS) < num_speakers).astype(jnp.int32)
    agent_hidden = jnp.zeros((MAX_AGENTS, HIDDEN_SIZE), dtype=jnp.float32)
    last_speak = jnp.zeros((MAX_AGENTS,), dtype=jnp.int32)
    reached_food = jnp.bool_(False)
    reward_accum = jnp.zeros((MAX_AGENTS,), dtype=jnp.float32)
    speaker_token_log = jnp.zeros((ep_len,), dtype=jnp.int32)

    return EnvState(
        grid=grid, agent_pos=agent_pos, agent_alive=agent_alive,
        agent_role=agent_role, agent_hidden=agent_hidden,
        last_speak=last_speak, food_pos=food_pos, food_slot_id=food_slot,
        reached_food=reached_food, reward_accum=reward_accum,
        speaker_token_log=speaker_token_log,
        tick=jnp.int32(0), rng=k_rng,
    )


# ==========================================
# OBSERVATIONS
# ==========================================

def get_obs(state, stage_cfg):
    grid = state.grid

    def place(g, i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        return jnp.where(state.agent_alive[i], g.at[r, c].set(AGENT_MARK), g), None

    grid_with_agents, _ = lax.scan(place, grid, jnp.arange(MAX_AGENTS))
    listener_grid = jnp.where(grid_with_agents == FOOD, EMPTY, grid_with_agents)

    padded_speaker = jnp.pad(grid_with_agents, VISION_RADIUS, constant_values=WALL)
    padded_listener = jnp.pad(listener_grid, VISION_RADIUS, constant_values=WALL)

    def per_agent_grid(i):
        r, c = state.agent_pos[i, 0], state.agent_pos[i, 1]
        speaker_view = lax.dynamic_slice(
            padded_speaker, (r, c), (VISION_SIZE, VISION_SIZE))
        listener_view = lax.dynamic_slice(
            padded_listener, (r, c), (VISION_SIZE, VISION_SIZE))
        return jnp.where(state.agent_role[i] == 1, speaker_view, listener_view)

    grids = vmap(per_agent_grid)(jnp.arange(MAX_AGENTS))
    grids_norm = grids.astype(jnp.float32) / (NUM_BYTE_TYPES - 1.0)

    def one_hot(t):
        return jax.nn.one_hot(t, NUM_SPEAK_ACTIONS)

    spk_0 = jnp.where(state.agent_role[0] == 1, state.last_speak[0], 0)
    spk_1 = jnp.where(state.agent_role[1] == 1, state.last_speak[1], 0)
    speech_concat = jnp.concatenate([one_hot(spk_0), one_hot(spk_1)])
    speech_all = jnp.broadcast_to(speech_concat[None, :],
                                   (MAX_AGENTS, speech_concat.shape[0]))

    role_oh = jax.nn.one_hot(state.agent_role, 2)
    pos_norm = state.agent_pos.astype(jnp.float32) / float(MAX_ARENA)

    # Food slot one-hot: visible ONLY to speakers. Listeners get zeros.
    # This is the key change in v5.4 — give the speaker unambiguous slot-id
    # info so distillation produces a true bijective code.
    slot_oh_full = jax.nn.one_hot(state.food_slot_id, MAX_FOOD_POSITIONS)
    # Mask per-agent: shape (MAX_AGENTS, MAX_FOOD_POSITIONS)
    is_speaker = (state.agent_role == 1).astype(jnp.float32)[:, None]
    slot_oh_per_agent = slot_oh_full[None, :] * is_speaker  # zero for listeners

    proprio = jnp.concatenate([role_oh, pos_norm, slot_oh_per_agent], axis=-1)

    return grids_norm, speech_all, proprio


SPEECH_OBS_DIM = 2 * NUM_SPEAK_ACTIONS
PROPRIO_DIM = 4 + MAX_FOOD_POSITIONS  # role(2) + pos(2) + slot_oh(MAX_FOOD_POSITIONS)


# ==========================================
# PHYSICS
# ==========================================

def step_env(state, actions, stage_cfg, food_positions):
    arena = stage_cfg["arena"]
    allow_speaker_move = stage_cfg["allow_speaker_move"]
    ep_len = stage_cfg["episode_length"]

    move_acts = actions[:, 0]
    speak_acts = actions[:, 1]

    if not allow_speaker_move:
        move_acts = jnp.where(state.agent_role == 1, 0, move_acts)

    curr_pos = state.agent_pos
    dp = MOVE_DIRS[move_acts]
    want_pos = curr_pos + dp
    want_pos = jnp.where(state.agent_alive[:, None], want_pos, curr_pos)
    want_pos = jnp.clip(want_pos, 0, arena - 1)

    target_vals = state.grid[want_pos[:, 0], want_pos[:, 1]]
    is_wall = (target_vals == WALL)

    def check_other_occ(i):
        same = jnp.all(curr_pos == want_pos[i][None, :], axis=-1) & state.agent_alive
        same = same & (jnp.arange(MAX_AGENTS) != i)
        return jnp.any(same)
    other_occ = vmap(check_other_occ)(jnp.arange(MAX_AGENTS))

    same_target = jnp.all(want_pos[:, None, :] == want_pos[None, :, :], axis=-1)
    eye = jnp.eye(MAX_AGENTS, dtype=jnp.bool_)
    has_conflict = jnp.any(same_target & state.agent_alive[:, None] &
                            state.agent_alive[None, :] & ~eye, axis=1)

    final_move = state.agent_alive & ~is_wall & ~other_occ & ~has_conflict
    final_pos = jnp.where(final_move[:, None], want_pos, curr_pos)

    old_dist = jnp.abs(curr_pos - state.food_pos[None, :]).sum(axis=-1).astype(jnp.float32)
    new_dist = jnp.abs(final_pos - state.food_pos[None, :]).sum(axis=-1).astype(jnp.float32)
    closer = (new_dist < old_dist) & (state.agent_role == 0) & state.agent_alive
    farther = (new_dist > old_dist) & (state.agent_role == 0) & state.agent_alive
    step_reward = (jnp.where(closer, STEP_CLOSER_REWARD, 0.0)
                   - jnp.where(farther, STEP_AWAY_COST, 0.0))

    on_food = jnp.all(final_pos == state.food_pos[None, :], axis=-1)
    listener_on_food = on_food & (state.agent_role == 0) & state.agent_alive
    reached_now = jnp.any(listener_on_food)

    time_remaining = jnp.float32(ep_len - state.tick)
    reach_bonus = jnp.where(reached_now,
                             REACH_FOOD_BONUS + time_remaining * 1.0,
                             0.0)

    listener_reach = listener_on_food.astype(jnp.float32) * reach_bonus
    speaker_share_mask = (state.agent_role == 1).astype(jnp.float32)
    num_speakers_alive = jnp.sum(speaker_share_mask * state.agent_alive.astype(jnp.float32))
    speaker_share_per = jnp.where(reached_now,
                                   reach_bonus * SPEAKER_FITNESS_SHARE /
                                   jnp.maximum(num_speakers_alive, 1.0),
                                   0.0)
    speaker_reach = speaker_share_mask * state.agent_alive.astype(jnp.float32) * speaker_share_per

    tick_reward = (step_reward + listener_reach + speaker_reach -
                   jnp.where(state.agent_alive, TICK_COST, 0.0))

    new_reward_accum = state.reward_accum + tick_reward

    first_speaker_token = state.last_speak[0]
    new_token_log = state.speaker_token_log.at[state.tick].set(first_speaker_token)

    new_last_speak = jnp.where(state.agent_alive, speak_acts, 0)
    new_alive = state.agent_alive & ~reached_now

    return state._replace(
        agent_pos=final_pos,
        agent_alive=new_alive,
        last_speak=new_last_speak,
        reached_food=state.reached_food | reached_now,
        reward_accum=new_reward_accum,
        speaker_token_log=new_token_log,
        tick=state.tick + 1,
    )


# ==========================================
# NETWORK
# ==========================================

class AgentNet(nn.Module):
    @nn.compact
    def __call__(self, obs_grid, obs_speech, obs_proprio, hidden):
        grid_bf = obs_grid.astype(jnp.bfloat16)
        speech_bf = obs_speech.astype(jnp.bfloat16)
        proprio_bf = obs_proprio.astype(jnp.bfloat16)
        hidden_bf = hidden.astype(jnp.bfloat16)

        x = grid_bf[None, :, :, None]
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME',
                    dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x_flat = x.reshape(-1)

        s = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.float32)(speech_bf)
        s = nn.relu(s)

        feat = jnp.concatenate([x_flat, s, proprio_bf], axis=-1)
        feat = nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(feat)
        feat = nn.relu(feat)

        zr = nn.Dense(2 * HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(
            jnp.concatenate([feat, hidden_bf], axis=-1))
        z = nn.sigmoid(zr[:HIDDEN_SIZE])
        r = nn.sigmoid(zr[HIDDEN_SIZE:])
        candidate = nn.tanh(
            nn.Dense(HIDDEN_SIZE, dtype=jnp.bfloat16, param_dtype=jnp.float32)(
                jnp.concatenate([feat, r * hidden_bf], axis=-1))
        )
        new_hidden = (1.0 - z) * hidden_bf + z * candidate

        h32 = new_hidden.astype(jnp.float32)
        move_logits = nn.Dense(NUM_MOVE_ACTIONS, param_dtype=jnp.float32)(h32)
        speak_logits = nn.Dense(NUM_SPEAK_ACTIONS, param_dtype=jnp.float32)(h32)
        return move_logits, speak_logits, h32


def sample_actions(speaker_params, listener_params,
                    obs_grid, obs_speech, obs_proprio, hidden,
                    roles, key, apply_fn, temperature):
    keys = random.split(key, MAX_AGENTS * 2).reshape(MAX_AGENTS, 2, 2)

    def act_one(og, os, op, h, role, ks):
        ml_s, sl_s, new_h_s = apply_fn(speaker_params, og, os, op, h)
        ml_l, sl_l, new_h_l = apply_fn(listener_params, og, os, op, h)

        is_speaker = (role == 1)
        ml = jnp.where(is_speaker, ml_s, ml_l)
        sl = jnp.where(is_speaker, sl_s, sl_l)
        new_h = jnp.where(is_speaker, new_h_s, new_h_l)

        def sample(logits, k):
            g = random.gumbel(k, logits.shape)
            return jnp.argmax(logits / temperature + g)

        move = sample(ml, ks[0])
        speak = sample(sl, ks[1])
        return jnp.array([move, speak], dtype=jnp.int32), new_h

    actions, new_hidden = vmap(act_one)(
        obs_grid, obs_speech, obs_proprio, hidden, roles, keys)
    return actions, new_hidden


# ==========================================
# EPISODE
# ==========================================

def run_episode(speaker_params, listener_params, apply_fn, init_state,
                stage_cfg, food_positions, ep_key):
    ep_len = stage_cfg["episode_length"]

    def step_fn(carry, step_key):
        state = carry
        obs_grid, obs_speech, obs_proprio = get_obs(state, stage_cfg)
        actions, new_hidden = sample_actions(
            speaker_params, listener_params,
            obs_grid, obs_speech, obs_proprio, state.agent_hidden,
            state.agent_role, step_key, apply_fn, ACTION_TEMP)
        state = state._replace(agent_hidden=new_hidden)
        new_state = step_env(state, actions, stage_cfg, food_positions)
        return new_state, actions

    step_keys = random.split(ep_key, ep_len)
    final_state, actions_log = lax.scan(step_fn, init_state, step_keys)
    return final_state, actions_log


def compute_fitness(final_state, stage_cfg):
    num_agents = stage_cfg["num_agents"]
    alive_mask = jnp.arange(MAX_AGENTS) < num_agents
    per_agent = final_state.reward_accum * alive_mask.astype(jnp.float32)
    team_reward = jnp.sum(per_agent) / jnp.maximum(
        jnp.sum(alive_mask.astype(jnp.float32)), 1.0)
    reached = final_state.reached_food.astype(jnp.float32)
    return team_reward, reached, final_state.food_slot_id, final_state.speaker_token_log


# ==========================================
# MI
# ==========================================

def np_compute_mi(food_slots_np, mode_tokens_np, num_food_positions, vocab_size):
    max_token = vocab_size + 1
    joint = np.zeros((num_food_positions, max_token), dtype=np.float64)
    for f, t in zip(food_slots_np, mode_tokens_np):
        if f < num_food_positions and t < max_token:
            joint[f, t] += 1
    total = joint.sum()
    if total == 0:
        return 0.0, 0.0
    joint /= total
    p_f = joint.sum(axis=1, keepdims=True)
    p_t = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log2(joint / (p_f * p_t + 1e-12) + 1e-12)
        log_ratio = np.where(joint > 0, log_ratio, 0.0)
        mi = np.sum(joint * log_ratio)
    max_mi = np.log2(num_food_positions)
    return float(mi), float(max_mi)


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


def rank_utility(fitness):
    n = fitness.shape[0]
    ranks = jnp.argsort(jnp.argsort(-fitness)).astype(jnp.float32)
    log_util = jnp.maximum(0.0, jnp.log(n / 2.0 + 1.0) - jnp.log(ranks + 1.0))
    return log_util / jnp.sum(log_util) - 1.0 / n


# ==========================================
# TEACHER (speaker only)
# ==========================================

def teacher_action(obs_grid, obs_speech, obs_proprio, hidden, food_slot, role, key):
    k1, _ = random.split(key)
    random_move = random.randint(k1, (), 0, NUM_MOVE_ACTIONS)
    speaker_token = jnp.minimum(food_slot + 1, NUM_SPEAK_ACTIONS - 1)
    is_speaker = (role == 1)
    move = jnp.where(is_speaker, 0, random_move)
    speak = jnp.where(is_speaker, speaker_token, 0)
    action = jnp.array([move, speak], dtype=jnp.int32)
    new_hidden = jnp.zeros_like(hidden)
    return action, new_hidden


def collect_teacher_data(key, stage_cfg, food_positions, num_episodes):
    ep_len = stage_cfg["episode_length"]

    def run_one_episode(ek):
        k_init, k_run = random.split(ek)
        state = init_env(k_init, stage_cfg, food_positions)

        def step_fn(carry, step_key):
            state = carry
            obs_grid, obs_speech, obs_proprio = get_obs(state, stage_cfg)
            agent_keys = random.split(step_key, MAX_AGENTS)

            def one_agent(og, os, op, h, role, k):
                return teacher_action(og, os, op, h, state.food_slot_id, role, k)

            actions, new_hidden = vmap(one_agent)(
                obs_grid, obs_speech, obs_proprio, state.agent_hidden,
                state.agent_role, agent_keys)

            data = (obs_grid, obs_speech, obs_proprio, state.agent_hidden,
                    actions, state.agent_alive, state.agent_role)
            state = state._replace(agent_hidden=new_hidden)
            new_state = step_env(state, actions, stage_cfg, food_positions)
            return new_state, data

        step_keys = random.split(k_run, ep_len)
        _, data_log = lax.scan(step_fn, state, step_keys)
        return data_log

    ep_keys = random.split(key, num_episodes)
    all_data = vmap(run_one_episode)(ep_keys)
    og, os_, op, h, a, al, ro = all_data

    def flat(x):
        return x.reshape((-1,) + x.shape[3:])

    return flat(og), flat(os_), flat(op), flat(h), flat(a), flat(al), flat(ro)


def distill_speaker(apply_fn, params, data, num_steps, batch_size, lr, key):
    """Train on (obs, action) pairs filtered to alive speakers only."""
    og, os_, op, h, a, al, ro = data
    N = og.shape[0]
    valid_mask = al & (ro == 1)
    valid_count = int(np.array(jax.device_get(valid_mask)).sum())
    print(f"    Data: {N} total samples, {valid_count} alive-speaker")

    if valid_count < batch_size:
        print(f"    Too few speaker samples, skipping")
        return params

    # Compact valid indices for efficient sampling
    valid_flat_indices = jnp.array(
        np.where(np.array(jax.device_get(valid_mask)))[0])
    n_valid = valid_flat_indices.shape[0]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch):
        b_og, b_os, b_op, b_h, b_a = batch

        def per_sample(og, os_, op, h, a):
            ml, sl, _ = apply_fn(params, og, os_, op, h)
            loss_m = optax.softmax_cross_entropy_with_integer_labels(ml, a[0])
            loss_s = optax.softmax_cross_entropy_with_integer_labels(sl, a[1])
            # Weight speech loss more: it's what matters
            return loss_m + 3.0 * loss_s

        losses = vmap(per_sample)(b_og, b_os, b_op, b_h, b_a)
        return jnp.mean(losses)

    grad_fn = jax.value_and_grad(loss_fn)

    @jit
    def train_step(params, opt_state, batch):
        loss, grads = grad_fn(params, batch)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for step in range(num_steps):
        key, sk = random.split(key)
        sample_idx = random.randint(sk, (batch_size,), 0, n_valid)
        idx = valid_flat_indices[sample_idx]
        batch = (og[idx], os_[idx], op[idx], h[idx], a[idx])
        params, opt_state, loss = train_step(params, opt_state, batch)
        if step % 300 == 0:
            print(f"    step {step:5d} loss={float(loss):.3f}")

    return params


# ==========================================
# STAGE
# ==========================================

def run_stage(stage_cfg, net, apply_fn, params_template, num_params, mesh, key,
              listener_init_flat=None, teacher_cache_path=None,
              listener_snapshot_path=None, best_listener_path=None):
    arena = stage_cfg["arena"]
    num_agents = stage_cfg["num_agents"]
    ep_len = stage_cfg["episode_length"]
    vocab_size = stage_cfg["vocab_size"]
    num_food_positions = stage_cfg["num_food_positions"]
    max_gens = stage_cfg["max_gens"]
    grad_eat = stage_cfg["graduation_eat_rate"]
    grad_mi_ratio = stage_cfg["graduation_mi_ratio"]

    food_positions = get_food_slot_positions(arena, num_food_positions)

    print(f"\n=== STAGE: {stage_cfg['name']} ===")
    print(f"  {stage_cfg['description']}")
    print(f"  Arena {arena}x{arena}, {num_agents} agents, "
          f"{num_food_positions} food positions, vocab {vocab_size}")
    print(f"  Graduation: eat_rate>={grad_eat}, MI>={grad_mi_ratio}×log2({num_food_positions})")
    print(f"  Max gens: {max_gens}")

    # --- SPEAKER BOOTSTRAP ---
    speaker_params_flat = None
    if teacher_cache_path is not None and os.path.exists(teacher_cache_path):
        print(f"  Loading cached speaker params from {teacher_cache_path}")
        speaker_params_flat = jnp.array(np.load(teacher_cache_path))
        if speaker_params_flat.shape[0] != num_params:
            print(f"  Cache shape mismatch, regenerating")
            speaker_params_flat = None

    if speaker_params_flat is None:
        print(f"  Collecting teacher data: {TEACHER_EPISODES} episodes...")
        k_collect, key = random.split(key)
        t0 = time.time()
        teacher_data = collect_teacher_data(
            k_collect, stage_cfg, food_positions, TEACHER_EPISODES)
        print(f"  Collection: {time.time() - t0:.1f}s")

        k_init, key = random.split(key)
        dummy_grid = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
        dummy_speech = jnp.zeros((SPEECH_OBS_DIM,), dtype=jnp.float32)
        dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
        dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
        fresh = net.init(k_init, dummy_grid, dummy_speech, dummy_proprio, dummy_hidden)

        print(f"  Distilling speaker for {TEACHER_TRAIN_STEPS} steps...")
        k_distill, key = random.split(key)
        speaker_params = distill_speaker(
            apply_fn, fresh, teacher_data,
            TEACHER_TRAIN_STEPS, TEACHER_BATCH_SIZE, TEACHER_LR, k_distill)
        speaker_params_flat = flatten_params(speaker_params)

        if teacher_cache_path is not None:
            np.save(teacher_cache_path, np.array(jax.device_get(speaker_params_flat)))
            print(f"  Speaker cached to {teacher_cache_path}")

    # --- DIAGNOSTIC ---
    print(f"  Diagnostic: evaluating speaker on {num_food_positions} slots...")
    speaker_params_tree = unflatten_params(speaker_params_flat, params_template)

    run_ep_diag = jit(lambda sp, lp, st, k: run_episode(
        sp, lp, apply_fn, st, stage_cfg, food_positions, k))

    k_diag, key = random.split(key)
    diag_tokens = []
    for slot in range(num_food_positions):
        k_init_d, k_run_d, k_diag = random.split(k_diag, 3)
        state = init_env(k_init_d, stage_cfg, food_positions)
        forced_pos = food_positions[slot]
        new_grid = jnp.where(state.grid == FOOD, EMPTY, state.grid)
        new_grid = new_grid.at[forced_pos[0], forced_pos[1]].set(FOOD)
        state = state._replace(grid=new_grid, food_pos=forced_pos,
                                food_slot_id=jnp.int32(slot))
        _, actions_log = run_ep_diag(speaker_params_tree, speaker_params_tree,
                                       state, k_run_d)
        diag_tokens.append(np.array(jax.device_get(actions_log[:, 0, 1])))

    print(f"  Speaker behavior:")
    mode_toks = []
    for slot in range(num_food_positions):
        tokens = diag_tokens[slot]
        non_silent = tokens[tokens > 0]
        if len(non_silent) > 0:
            counts = np.bincount(non_silent, minlength=NUM_SPEAK_ACTIONS)
            top_token = int(np.argmax(counts[1:])) + 1
            top_pct = 100.0 * int(counts[top_token]) / max(len(non_silent), 1)
        else:
            top_token = 0
            top_pct = 0.0
        mode_toks.append(top_token)
        expected = min(slot + 1, NUM_SPEAK_ACTIONS - 1)
        mark = "✓" if top_token == expected else "✗"
        print(f"    slot {slot}: speaks token {top_token} ({top_pct:.0f}%), "
              f"expected {expected} [{mark}]")

    diag_mi, diag_max_mi = np_compute_mi(
        np.arange(num_food_positions), np.array(mode_toks),
        num_food_positions, vocab_size)
    print(f"  Speaker MI: {diag_mi:.3f}/{diag_max_mi:.3f} "
          f"({diag_mi/max(diag_max_mi, 1e-6):.2f}×)")

    if diag_mi / max(diag_max_mi, 1e-6) < 0.5:
        print(f"  WARNING: speaker MI low — listener has weak signal")

    # --- LISTENER SETUP ---
    # Priority: resume-from-snapshot > transferred-from-prior-stage > fresh random.
    # Resume helps after an interrupted session: we re-do distillation (cheap)
    # but pick the listener up where we left off.
    if (best_listener_path is not None and
        os.path.exists(best_listener_path) and
        listener_init_flat is None):
        try:
            snap = jnp.array(np.load(best_listener_path))
            if snap.shape[0] == num_params:
                print(f"  Listener: RESUMING from snapshot {best_listener_path}")
                listener_init_flat = snap
        except Exception as e:
            print(f"  Could not load snapshot: {e}")

    if listener_init_flat is None:
        print(f"  Listener: fresh random params")
        k_lst, key = random.split(key)
        dummy_grid = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
        dummy_speech = jnp.zeros((SPEECH_OBS_DIM,), dtype=jnp.float32)
        dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
        dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
        fresh_lst = net.init(k_lst, dummy_grid, dummy_speech, dummy_proprio, dummy_hidden)
        listener_init_flat = flatten_params(fresh_lst)
    else:
        print(f"  Listener: starting from provided init")

    # --- ES SETUP ---
    assert POP_SIZE % 2 == 0
    half_pop_per_tribe = POP_SIZE // 2
    total_half = NUM_TRIBES * half_pop_per_tribe
    per_device = TOTAL_POP // jax.device_count()

    tribe_centers_list = []
    for t in range(NUM_TRIBES):
        k_t, key = random.split(key)
        tribe_centers_list.append(
            listener_init_flat + random.normal(k_t, (num_params,)) * PBT_MUTATION_SIGMA)
    tribe_centers = jnp.stack(tribe_centers_list)

    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LR),
    )
    tribe_opt_states = [optimizer.init(tribe_centers[t]) for t in range(NUM_TRIBES)]
    stacked_opt = jax.tree.map(lambda *xs: jnp.stack(xs), *tribe_opt_states)

    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P(), P(), P(), P('pop'), P('pop')),
             out_specs=(P('pop'), P('pop'), P('pop'), P('pop')),
             check_rep=False)
    def sharded_eval(speaker_flat, listener_centers, sigma,
                     noise_base, env_keys_shard, ep_keys_shard):
        my_idx = lax.axis_index('pop')
        start = my_idx * per_device
        spk_local = unflatten_params(speaker_flat, params_template)

        def eval_member(local_idx, ek_m, pk_m):
            global_idx = start + local_idx
            is_neg = global_idx >= total_half
            gh = jnp.where(is_neg, global_idx - total_half, global_idx)
            tribe_idx = gh // half_pop_per_tribe

            row_key = random.fold_in(noise_base, gh)
            noise = random.normal(row_key, (num_params,))
            noise = jnp.where(is_neg, -noise, noise)
            lst_flat = listener_centers[tribe_idx] + sigma * noise
            lst_local = unflatten_params(lst_flat, params_template)

            def eval_env(ek, pk):
                init_state = init_env(ek, stage_cfg, food_positions)
                final_state, _ = run_episode(
                    spk_local, lst_local, apply_fn, init_state,
                    stage_cfg, food_positions, pk)
                team_r, reached, slot, token_log = compute_fitness(final_state, stage_cfg)
                counts = jnp.bincount(token_log, length=NUM_SPEAK_ACTIONS)
                counts_ns = counts.at[0].set(0)
                any_speech = jnp.any(counts_ns > 0)
                mode_tok = jnp.where(any_speech, jnp.argmax(counts_ns), 0)
                return team_r, reached, slot, mode_tok

            teams, reaches, slots, mtoks = vmap(eval_env)(ek_m, pk_m)
            return jnp.mean(teams), jnp.mean(reaches), slots[0], mtoks[0]

        local_idx = jnp.arange(per_device)
        fits, reach_rates, slots, mtoks = vmap(eval_member)(
            local_idx, env_keys_shard, ep_keys_shard)
        return fits, reach_rates, slots, mtoks

    @jit
    def reconstruct_noise(noise_base):
        def gen(idx):
            k = random.fold_in(noise_base, idx)
            return random.normal(k, (num_params,))
        return vmap(gen)(jnp.arange(total_half))

    def es_update_one(center, noise_tribe, sigma, fp, fn, opt_state):
        combined = jnp.concatenate([fp, fn])
        util = rank_utility(combined)
        pos_u = util[:half_pop_per_tribe]
        neg_u = util[half_pop_per_tribe:]
        grad = jnp.dot((pos_u - neg_u), noise_tribe) / (half_pop_per_tribe * sigma)
        grad = grad - WEIGHT_DECAY * center
        updates, new_opt = optimizer.update(-grad, opt_state, center)
        return optax.apply_updates(center, updates), new_opt

    @jit
    def es_update_all(centers, noise_half, sigma, fits, stacked_opt):
        def per_tribe(t_idx, center, opt_state):
            tps = t_idx * half_pop_per_tribe
            tns = total_half + tps
            fp = lax.dynamic_slice(fits, (tps,), (half_pop_per_tribe,))
            fn = lax.dynamic_slice(fits, (tns,), (half_pop_per_tribe,))
            nt = lax.dynamic_slice(noise_half, (tps, 0),
                                     (half_pop_per_tribe, num_params))
            return es_update_one(center, nt, sigma, fp, fn, opt_state)

        tribe_indices = jnp.arange(NUM_TRIBES)
        new_centers, new_opts = vmap(per_tribe)(tribe_indices, centers, stacked_opt)
        return new_centers, new_opts

    # --- ES LOOP ---
    current_sigma = NOISE_STD_INIT
    tribe_fit_history = [[] for _ in range(NUM_TRIBES)]
    t_start = time.time()
    print(f"  Compiling ES loop and running...")

    graduated = False
    graduation_reason = ""
    final_gen = 0
    mean_fit = 0.0
    pop_eat_rate = 0.0
    mi = 0.0
    max_mi = 1.0
    mi_ratio = 0.0
    tribe_means = [0.0] * NUM_TRIBES

    # Best-ever tracking: keep the single best listener we've seen so far,
    # judged by (eat_rate + 0.2 * mi_ratio) as a combined score.
    best_ever_score = -1.0
    best_ever_listener = tribe_centers[0]
    best_ever_eat = 0.0
    best_ever_mi_ratio = 0.0
    best_ever_gen = 0
    eat_rate_history = []

    for gen in range(max_gens):
        final_gen = gen
        k_env, k_ep, k_noise, key = random.split(key, 4)
        sigma_j = jnp.float32(current_sigma)

        env_keys = random.split(k_env, TOTAL_POP * NUM_ENVS_PER_MEMBER).reshape(
            TOTAL_POP, NUM_ENVS_PER_MEMBER, 2)
        ep_keys = random.split(k_ep, TOTAL_POP * NUM_ENVS_PER_MEMBER).reshape(
            TOTAL_POP, NUM_ENVS_PER_MEMBER, 2)

        fits, reached_rates, slots, mtoks = sharded_eval(
            speaker_params_flat, tribe_centers, sigma_j, k_noise, env_keys, ep_keys)
        noise_half = reconstruct_noise(k_noise)
        tribe_centers, stacked_opt = es_update_all(
            tribe_centers, noise_half, sigma_j, fits, stacked_opt)

        mean_fit = float(jnp.mean(fits))
        max_fit = float(jnp.max(fits))
        tribe_means = [
            float(jnp.mean(fits[t * half_pop_per_tribe:(t + 1) * half_pop_per_tribe]))
            for t in range(NUM_TRIBES)]
        for t in range(NUM_TRIBES):
            tribe_fit_history[t].append(tribe_means[t])

        pop_eat_rate = float(jnp.mean(reached_rates))
        eat_rate_history.append(pop_eat_rate)
        slots_np = np.array(jax.device_get(slots))
        mtoks_np = np.array(jax.device_get(mtoks))
        mi, max_mi = np_compute_mi(slots_np, mtoks_np, num_food_positions, vocab_size)
        mi_ratio = mi / max(max_mi, 1e-6)

        # Best-ever: combined score weights eat rate heavily, MI secondary
        current_score = pop_eat_rate + 0.2 * mi_ratio
        if current_score > best_ever_score:
            best_ever_score = current_score
            # Identify best tribe right now and grab its center
            best_tribe_idx = int(np.argmax(tribe_means))
            best_ever_listener = tribe_centers[best_tribe_idx]
            best_ever_eat = pop_eat_rate
            best_ever_mi_ratio = mi_ratio
            best_ever_gen = gen

        if gen > 10 and gen % 5 == 0:
            recent = np.mean([np.mean(h[-5:]) for h in tribe_fit_history])
            older_vals = [np.mean(h[-10:-5]) for h in tribe_fit_history if len(h) >= 10]
            if older_vals:
                older = np.mean(older_vals)
                if recent > older + 0.2:
                    current_sigma = max(NOISE_STD_MIN, current_sigma * 0.97)
                else:
                    current_sigma = min(NOISE_STD_MAX, current_sigma * 1.03)

        if gen > 0 and gen % PBT_INTERVAL == 0:
            recent_perf = [np.mean(h[-10:]) if len(h) >= 10 else h[-1]
                           for h in tribe_fit_history]
            best_t = int(np.argmax(recent_perf))
            worst_t = int(np.argmin(recent_perf))
            if best_t != worst_t:
                k_mut, key = random.split(key)
                mutation = random.normal(k_mut, (num_params,)) * PBT_MUTATION_SIGMA
                new_center = tribe_centers[best_t] + mutation
                tribe_centers = tribe_centers.at[worst_t].set(new_center)
                fresh_opt = optimizer.init(new_center)
                stacked_opt = jax.tree.map(
                    lambda s, f: s.at[worst_t].set(f), stacked_opt, fresh_opt)
                print(f"  >> PBT gen {gen}: tribe {worst_t} ← tribe {best_t}")

        # Periodic snapshot: save best-ever listener AND current tribe centers
        if gen > 0 and gen % SNAPSHOT_EVERY_GENS == 0:
            if best_listener_path is not None:
                try:
                    np.save(best_listener_path,
                            np.array(jax.device_get(best_ever_listener)))
                except Exception as e:
                    print(f"  !! snapshot save failed: {e}")
            if listener_snapshot_path is not None:
                try:
                    np.save(listener_snapshot_path,
                            np.array(jax.device_get(tribe_centers)))
                except Exception as e:
                    print(f"  !! tribe snapshot save failed: {e}")
            print(f"  [snapshot gen {gen}: best_ever eat={best_ever_eat:.3f} "
                  f"MI={best_ever_mi_ratio:.2f}× @ gen {best_ever_gen}]")

        # Hard graduation
        if pop_eat_rate >= grad_eat and mi_ratio >= grad_mi_ratio:
            print(f"  *** GRADUATED at gen {gen}: "
                  f"eat={pop_eat_rate:.3f}, MI={mi:.3f}/{max_mi:.3f} "
                  f"({mi_ratio:.2f}×) ***")
            graduated = True
            graduation_reason = "hard_threshold"
            break

        # Plateau-based soft graduation: if eat_rate has been stable within a
        # narrow band for PLATEAU_WINDOW gens, AND we're within 90% of target,
        # AND MI ratio is within 90% of target, graduate. This prevents
        # grinding against diminishing returns.
        if len(eat_rate_history) >= PLATEAU_WINDOW:
            window = np.array(eat_rate_history[-PLATEAU_WINDOW:])
            window_std = float(np.std(window))
            window_mean = float(np.mean(window))
            if window_std < PLATEAU_STD_THRESHOLD:
                # Check if best_ever is close to thresholds
                if (best_ever_eat >= grad_eat * PLATEAU_GRADUATION_FRACTION and
                    best_ever_mi_ratio >= grad_mi_ratio * PLATEAU_GRADUATION_FRACTION):
                    print(f"  *** PLATEAU GRADUATION at gen {gen}: "
                          f"eat stable for {PLATEAU_WINDOW} gens "
                          f"(mean={window_mean:.3f}, std={window_std:.3f}); "
                          f"best_ever eat={best_ever_eat:.3f}, "
                          f"MI={best_ever_mi_ratio:.2f}× ***")
                    graduated = True
                    graduation_reason = "plateau"
                    break

        if gen % 10 == 0 or gen == max_gens - 1:
            elapsed = time.time() - t_start
            print(f"  gen {gen:4d} | {elapsed:6.0f}s | "
                  f"fit_mean={mean_fit:7.2f} max={max_fit:7.2f} | "
                  f"t:{','.join(f'{m:5.1f}' for m in tribe_means)} | "
                  f"eat={pop_eat_rate:.3f} | MI={mi:.2f}/{max_mi:.2f} ({mi_ratio:.2f}×) | "
                  f"σ={current_sigma:.4f}")

    # --- Collect top-K listeners ---
    # Always include the best-ever listener first, then fill with current
    # tribe centers by last-gen fitness.
    last_order = np.argsort(-np.array(tribe_means))
    top_list = [best_ever_listener]
    for t_idx in last_order:
        candidate = tribe_centers[int(t_idx)]
        top_list.append(candidate)
        if len(top_list) >= TOP_K_PRESERVE:
            break
    while len(top_list) < TOP_K_PRESERVE:
        k_pad, key = random.split(key)
        top_list.append(tribe_centers[0] + random.normal(k_pad, (num_params,)) * 0.01)
    top_k = jnp.stack(top_list[:TOP_K_PRESERVE])

    # Use best-ever listener as the primary output. This is the key fix vs v5.5:
    # we no longer return "whatever the final gen happened to be," we return
    # the best snapshot we saw during the whole run.
    best_listener_flat = best_ever_listener

    summary = {
        "gens_used": final_gen + 1,
        "final_eat_rate": pop_eat_rate,
        "final_mi": mi,
        "final_mi_ratio": mi_ratio,
        "graduated": graduated,
        "graduation_reason": graduation_reason if graduated else "none",
        "best_ever_eat": best_ever_eat,
        "best_ever_mi_ratio": best_ever_mi_ratio,
        "best_ever_gen": best_ever_gen,
    }

    return speaker_params_flat, best_listener_flat, top_k, graduated, summary


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("  BYTE-MULTI-AGENT v5.6: TUNED ES + SNAPSHOTS + PLATEAU GRAD")
    print("  (persistent ckpts, best-ever tracking, lower thresholds)")
    print("=" * 60)

    num_devices = jax.device_count()
    assert TOTAL_POP % num_devices == 0

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), axis_names=('pop',))

    key = random.PRNGKey(2024)
    net = AgentNet()
    apply_fn = net.apply

    k_init, key = random.split(key)
    dummy_grid = jnp.zeros((VISION_SIZE, VISION_SIZE), dtype=jnp.float32)
    dummy_speech = jnp.zeros((SPEECH_OBS_DIM,), dtype=jnp.float32)
    dummy_proprio = jnp.zeros((PROPRIO_DIM,), dtype=jnp.float32)
    dummy_hidden = jnp.zeros((HIDDEN_SIZE,), dtype=jnp.float32)
    params_template = net.init(k_init, dummy_grid, dummy_speech, dummy_proprio, dummy_hidden)
    num_params = flatten_params(params_template).shape[0]
    print(f"Network parameters: {num_params:,} (per role)")
    print(f"Mesh: {num_devices} devices, total pop {TOTAL_POP}")

    # Prefer /kaggle/working (persistent across Kaggle session restarts and
    # downloadable from the Kaggle UI), fall back to /tmp (fast but ephemeral)
    # then current dir.
    if os.path.exists('/kaggle/working'):
        CKPT_DIR = '/kaggle/working'
    elif os.path.exists('/tmp'):
        CKPT_DIR = '/tmp'
    else:
        CKPT_DIR = '.'
    print(f"Checkpoint directory: {CKPT_DIR}")

    current_listener_init = None

    for stage_idx, stage_cfg in enumerate(STAGES):
        speaker_cache = os.path.join(
            CKPT_DIR, f'byte_multi_v5_6_{stage_cfg["name"]}_speaker.npy')
        listener_ckpt = os.path.join(
            CKPT_DIR, f'byte_multi_v5_6_{stage_cfg["name"]}_listener.npy')
        topk_ckpt = os.path.join(
            CKPT_DIR, f'byte_multi_v5_6_{stage_cfg["name"]}_topk.npy')
        # Snapshot paths — written to periodically during run_stage so progress
        # is preserved even if the session dies mid-stage.
        best_snapshot = os.path.join(
            CKPT_DIR, f'byte_multi_v5_6_{stage_cfg["name"]}_best_snapshot.npy')
        tribe_snapshot = os.path.join(
            CKPT_DIR, f'byte_multi_v5_6_{stage_cfg["name"]}_tribe_snapshot.npy')

        if os.path.exists(listener_ckpt) and os.path.exists(topk_ckpt):
            print(f"\n=== STAGE {stage_idx}: {stage_cfg['name']} — CACHED ===")
            cached = jnp.array(np.load(listener_ckpt))
            if cached.shape[0] == num_params:
                current_listener_init = cached
                print(f"  Using cached listener for next stage")
                continue

        speaker_flat, best_listener_flat, top_k, graduated, summary = run_stage(
            stage_cfg, net, apply_fn, params_template, num_params, mesh, key,
            listener_init_flat=current_listener_init,
            teacher_cache_path=speaker_cache,
            listener_snapshot_path=tribe_snapshot,
            best_listener_path=best_snapshot)

        print(f"\n=== STAGE {stage_idx} COMPLETE: {stage_cfg['name']} ===")
        print(f"  Graduated: {graduated} ({summary.get('graduation_reason', 'none')})")
        print(f"  Gens used: {summary['gens_used']}")
        print(f"  Final eat rate: {summary['final_eat_rate']:.3f}")
        print(f"  Final MI: {summary['final_mi']:.3f} ({summary['final_mi_ratio']:.2f}×)")
        print(f"  Best-ever eat rate: {summary['best_ever_eat']:.3f} "
              f"(MI={summary['best_ever_mi_ratio']:.2f}×) @ gen {summary['best_ever_gen']}")

        np.save(listener_ckpt, np.array(jax.device_get(best_listener_flat)))
        np.save(topk_ckpt, np.array(jax.device_get(top_k)))
        print(f"  Best listener saved to {listener_ckpt}")
        print(f"  Top-K saved to {topk_ckpt}")

        if not graduated:
            print(f"\n*** CEILING at stage {stage_idx}. Stopping curriculum. ***")
            print(f"*** Best listener available at: {listener_ckpt} ***")
            break

        current_listener_init = best_listener_flat
        key, _ = random.split(key)

    print("\nCurriculum complete.")


if __name__ == "__main__":
    main()
