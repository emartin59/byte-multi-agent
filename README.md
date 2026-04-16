# Byte-Multi-Agent

**Can language, math, and intelligence emerge from nothing but survival pressure on a byte grid?**

This is a research project exploring whether multi-agent systems can develop language, coordination, proto-economics, and eventually generalized reasoning *tabula rasa* — with no pretraining, no human language priors, no supervised signal. Just agents, bytes, scarcity, and evolutionary pressure.

Think of it as asking: what would AlphaGo Zero for LLMs look like? AGZ threw away human game knowledge and rediscovered Go by self-play. We want to throw away human language and knowledge and have agents rediscover communication, tool use, and abstract concepts by interacting with each other in a world where survival depends on figuring things out together.

It's JAX + Flax, designed for TPU pods (works on GPU/CPU too), and uses evolutionary strategies rather than gradient-based RL so credit assignment across long horizons and multi-agent interactions is cleanly handled.

## Why this project exists

Most work on emergent communication uses referential games — Lewis signaling, Rosetta-stone setups, cooperative navigation. The languages that emerge in those settings are real but narrow: they're solutions to a specific optimization problem, not open-ended systems that could support reasoning about novel concepts.

The bet here is different. If you build a world with:

- Genuine scarcity (agents die without food)
- Information asymmetry (no two agents see exactly the same things)
- Combinatorial discovery (crafting recipes you don't know a priori)
- A persistent writing substrate (bytes on the ground that stay there)
- A transient speech substrate (audio that decays)
- Heterogeneous "species" that must coordinate across different roles

...then some form of communication should be instrumentally useful. And if communication is useful, selection will favor it. The long-term hope is that language in this setting won't just be a one-task code — it will carry *information about the world* that agents can recombine, store, and pass along. That's the substrate on which reasoning can grow.

We're very far from that. But the project aims at it seriously rather than hand-waving at it, and the code is designed to make each next experiment cheap.

## What's in here right now

A single self-contained JAX program (`byte_multi_agent_v2.py`) that implements:

**World.** A 64×64 byte grid with an arena, containing food, seeds (regrow into food), rocks, water, writing letters (a–z), tools, and agents. Everything is a byte. The display is ASCII.

**Agents.** Up to 8 agents per environment, each with energy (die at 0), a 4-slot inventory, a 15×15 vision crop centered on them, an audio channel hearing nearby agents' recent speech, and a GRU hidden state that persists across ticks. Actions: move, speak (27 tokens), write persistent bytes on the ground (27 tokens), pickup/use/drop tools, choose inventory slot.

**Economy.** Metabolic energy cost per tick, higher costs for moving/writing/speaking. Food restores energy. Eating food produces a seed. Seeds stochastically regrow into food. This creates a sustainable-but-fragile resource loop.

**General crafting substrate.** Adjacent resource or tool bytes may spontaneously combine. The output is a deterministic hash of the two input byte types. Of 37 possible output bytes, only 4 are "active" tools with effects (scatter seed, rock→food, water→seed, energy boost); the rest are inert junk. Agents have to *discover* which pairings produce useful items. This is meant to be more AGZ-spirited than hardcoded recipes.

**Heterogeneous agents.** Four distinct "species" (four separate policy networks) are jointly optimized. Each agent slot is randomly assigned a species per episode. On top of that, every agent gets a random binary mask over six observation categories (food visibility, rock visibility, water, tools, writing, other agents) — so different agents literally see different things. The mask is part of each agent's proprioception, so it knows what it's missing. This is the main pressure for communication: you can see the food, I can see the tool, we need each other.

**Audio channel.** Speech is not just rendered on the grid. It goes into a per-agent 3-tick audio buffer that other agents within Chebyshev range 10 can "hear" as a dedicated input. Decoupling language from spatial vision lets agents develop protocols without it getting tangled in pathfinding.

**Recurrent policy.** Small CNN on the vision crop + MLP on audio + proprioception → GRU → five action heads. Bfloat16 internally, float32 params, ~100K parameters per species.

**Training.** Mirrored Evolutionary Strategies with rank shaping and weight decay. An adaptive noise controller monitors signal-to-noise in the fitness trajectory and adjusts perturbation scale automatically, with a ratchet that resists regression. Population of 512, four environments per member. Sharded across all available devices via `shard_map` — near-linear pod scaling.

**Fitness.** `sum(survival_ticks) + 0.5 · sum(energy_earned) + 0.1 · sum(final_energy)`. No flat alive-bonus; idling has to be strictly worse than seeking food.

## Running it

```bash
pip install jax flax optax numpy
# for TPU:
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

python byte_multi_agent_v2.py
```

It autodetects available devices. On a v3-8 or v4-8 TPU you should see ~4–8× throughput over a single device. Checkpoints save every 50 generations to `/tmp`.

Every 25 generations (or every 2 minutes, whichever comes first) it prints a training summary, an ASCII render of a preview episode, a count of writing cells and ground tools, and speech diagnostics (how many utterances, how many unique tokens).

What you'll see early on: agents wandering mostly randomly, energy trending down, lots of death. That's normal. The signal takes tens to hundreds of generations to bootstrap because ES + discrete actions + 400-tick episodes is a hard credit-assignment problem.

What you're looking for as signs of life:
- Mean fitness climbing above ~random-policy baseline
- Agents persistently clustering near food
- Speech token counts growing (not just random noise — the adaptive noise controller will eventually squeeze them toward useful tokens)
- Ground tools accumulating (means agents are crafting)
- Persistent writing (means agents have learned to leave marks)

## Configuration knobs

The top of `byte_multi_agent_v2.py` has all hyperparameters. The ones most worth sweeping first:

- `SPEAK_COST`, `AUDIO_RANGE`, `AUDIO_PERSIST_TICKS` — if agents stay silent, loosen these
- `NUM_POLICY_SPECIES` — more species means more coordination pressure but also more parameters to optimize
- `CRAFT_PROB` — increases tool discovery rate
- `METABOLIC_COST` and `FOOD_ENERGY` — set scarcity
- `POP_SIZE` and `NUM_ENVS_PER_MEMBER` — ES variance/compute tradeoff
- `NOISE_STD_INIT/MIN/MAX` — exploration bounds (the adaptive controller works within these)

## Roadmap — where we want to go

Roughly in order of what would most move the needle on emergent language and reasoning:

### Near-term (good first issues)

- **Speech analysis tooling.** We need a post-hoc script that samples thousands of episodes from a checkpoint and looks for statistical regularities: does token X correlate with nearby food? Does hearing token Y change the listener's next-action distribution? Mutual-information measurements between speech and world state would turn "agents are speaking" into "agents are communicating meaningfully."
- **Better ASCII renderer / video export.** Currently prints occasional snapshots. We want per-tick GIFs of the best episode per generation, with speech bubbles.
- **Unit tests for the environment physics.** Crafting, seed→food regrowth, eating, inventory, and writing all need tests. The vectorized scatter writes are particularly easy to subtly break.
- **Benchmark the sharded training loop** on different TPU topologies and record throughput numbers.

### Medium-term (research directions)

- **Population-based training / island model.** Right now all four species are jointly optimized by one ES center. We want true divergent evolution: periodic tournament selection between species, migration, and species replacement. This is closer to actual open-ended evolution and should produce more specialization.
- **Replace ES with a hybrid ES + policy gradient.** ES is great for credit assignment across long horizons, but it wastes information within an episode. A PPO or V-MPO inner loop with ES as an outer meta-optimizer might combine the best of both.
- **Curriculum / arena growth.** Start with tiny 10×10 arenas and expand as agents master them. Same for number of concurrent agents.
- **Partial observability of audio too.** Right now the audio channel is clean binary distance gating. Add noise/distortion proportional to distance — forces agents to develop error-correcting or redundant encodings, which is how real language got robust.
- **Persistent world / no episode resets.** Move to a continuous world where individual agents die and are replaced, but the grid state persists. This removes the implicit "game" framing and makes writing truly persistent across generations — a precondition for cultural accumulation.

### Long-term (the actual bets)

- **Proto-math from trade.** If we introduce barter (agents can drop items for other agents to pick up) and multiple resource types with different nutritional values, we should see numerical cognition pressure: agents need to represent "how much" and "worth how much." Whether this produces anything we'd recognize as counting or arithmetic is an open empirical question.
- **Cultural transmission via writing.** Writing is persistent. If one agent dies and another reads its writing, does behavior change? The long bet is that writing becomes a medium for passing discoveries (like useful crafting recipes) across generations without direct teaching.
- **Compositional language.** Does the speech code develop systematic combinations — does "AB" mean something related to "A" and "B" separately? This is the classic test for whether emergent language has any productive/generative structure. Most emergent-communication setups fail this; we want to engineer conditions where it can succeed.
- **Transfer.** Can a policy trained in this world be fine-tuned to understand symbolic inputs in a different (also emergent) world? If yes, we've got evidence that the learned representations are compositional in a useful way.
- **Scaling to thousands of agents.** Most of the architecture should tolerate this; the bottleneck is grid size and per-agent scan loops. Larger populations are where evolutionary game theory actually gets interesting — norms, coalitions, enforcement.

### Things we've explicitly rejected (for now)

- **LLM scaffolding of any kind.** No pretrained models, no natural-language inputs or outputs. The whole point is to avoid smuggling in human language.
- **Hand-designed communication protocols** like "agent emits a vector, other agent consumes it." The channel has to emerge from the environment pressures, not be a bolted-on MLP.
- **Reward shaping for communication specifically.** No bonus for speaking, no reward for being understood. Communication has to pay for itself via its downstream effect on survival.

## Contributing

Pull requests welcome. The project is deliberately single-file and light on abstractions so that it stays hackable — if you want to try a wild idea, you should be able to fork, change 50 lines, and run it. Please keep it that way when contributing; prefer inline experimental code over premature refactoring.

If you're adding a feature, include either a unit test or a before/after learning curve so reviewers can see it works.

If you're doing a training run you think is interesting, open an issue with the checkpoint, the config, and a few preview frames. We're trying to build up a shared library of "what happens when you change X."

Research ideas, critiques, and "have you considered" issues are as valuable as code.

## Citation

If this project is useful for your work, a citation is appreciated:

```
@software{byte_multi_agent,
  title   = {Byte-Multi-Agent: Toward Emergent Language and Knowledge Tabula Rasa},
  year    = {2026},
  url     = {https://github.com/emartin59/byte-multi-agent}
}
```

## Acknowledgements

Developed using TPUs provided by the Google TPU Research Cloud (TRC) program. Thanks to the TRC team for making ambitious open research possible.

## License

MIT. See [LICENSE](LICENSE).
