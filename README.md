# autoresearch-maze

Two neural networks train against each other: a Generator that invents mazes and a Solver that navigates them. An AI agent (Claude Code) autonomously experiments with both architectures overnight, editing the training code, running 10-minute experiments, keeping improvements and discarding regressions. You wake up to a log of 50+ experiments and a Solver that can navigate mazes it has never seen before. Unlike board game versions of Autoresearch where the rules are fixed and difficulty comes from the opponent, here the environment itself is learned. The Generator does not play within fixed rules — it invents the wall layout each episode, making this a fundamentally different kind of adversarial training.

## Current results

*To be filled after first run.*

## How the two reward rules work

The Generator's reward system has two rules that together create an automatic difficulty curriculum and prevent the Generator from cheating.

**Rule 1: Progress-gated reward.** After the Solver attempts a maze, we measure how close it got to the goal relative to the shortest possible path. If the Solver reaches the goal, the Generator gets nothing because the maze was too easy. If the Solver barely moves (progress below 30%), the Generator gets nothing because the maze was too hard or degenerate — the Solver learned nothing from it. If the Solver gets 80% of the way there and gets stuck, the Generator gets 0.8 points. The Generator earns the most reward by creating mazes at the precise edge of the Solver's current ability. Too easy earns nothing. Too hard earns nothing. The sweet spot is where the Solver almost solves it but not quite.

**Rule 2: Diversity multiplier.** The Generator maintains a rolling buffer of its last 50 mazes. When it produces a new maze, we compute its structural similarity (fraction of matching cells) to every maze in the buffer. If the most similar maze in the buffer shares more than 80% of its cell placements, the Generator's reward is penalized proportionally. An identical maze gets zero reward regardless of how well-calibrated its difficulty was. This forces the Generator to keep inventing new challenges rather than farming reward by producing slight variations of the same maze.

The final reward is `progress_reward × diversity_multiplier`. The Generator becomes an adaptive teacher that automatically calibrates difficulty and continuously produces novel challenges.

## Why this requires neural networks

No hand-crafted maze generation algorithm can produce "maximally educational" mazes because what counts as educational depends entirely on what the Solver currently knows, and that changes every training step. Early in training, a simple corridor is a challenging maze. Later, the Solver masters corridors and needs dead ends and branching paths. Even later, it needs mazes with misleading shortcuts that look promising but lead to dead ends. The Generator must learn to model the Solver's current weaknesses and exploit them, which is a task that requires a learned model — the optimal maze at step 1000 is completely different from the optimal maze at step 50000.

## The three-file architecture

This project follows Karpathy's Autoresearch pattern exactly, just applied to adversarial maze training instead of language modeling.

**`prepare.py`** is the immutable harness. It contains the maze environment (11×11 grid, movement rules, step tracking), BFS utilities for pathfinding and maze validation, a fixed benchmark suite of 200 mazes spanning five difficulty tiers, and the sacred evaluation metric (`evaluate_solve_rate`). This file never changes. It defines the rules of the world and the yardstick for progress.

**`train.py`** is the mutable artifact. It contains both neural networks (Generator and Solver), the adversarial training loop, all hyperparameters, reward computation, and the diversity buffer. The autoresearch agent modifies only this file, experimenting with architectures, learning rates, reward shaping, update ratios, and anything else that might improve the Solver's benchmark performance.

**`program.md`** is the human's instructions to the AI agent. It defines the experiment loop (edit, commit, train, evaluate, keep or discard, repeat forever), the logging format, and a prioritized list of experiment ideas. The human programs the agent; the agent programs the training code.

## The two-level optimization loop

The system has two nested optimization loops. The inner loop is the adversarial co-training of Generator and Solver over 10 minutes: the Generator produces mazes, the Solver attempts them, both receive rewards, and both update their weights. The outer loop is Autoresearch itself: the AI agent edits `train.py`, runs the inner loop, checks the resulting `solve_rate` on the fixed benchmark, and decides whether to keep or discard the change. The agent never generates a maze or solves one directly. It only sees the final `solve_rate` number and reasons about what architectural or procedural changes might improve it.

## Project structure

**Source files (tracked in git):**

```
prepare.py      — maze environment, BFS, benchmark suite, evaluation (do not modify)
train.py        — Generator + Solver networks, training loop (agent modifies this)
program.md      — agent instructions
analysis.py     — reads results.tsv, generates progress.png
report.py       — reads results.tsv + logs/, generates report.md
pyproject.toml  — dependencies
.gitignore      — excludes generated files
README.md       — this file
```

**Generated files (not tracked):**

```
results.tsv     — experiment log (tab-separated)
logs/           — full output of each experiment, named by commit hash
run.log         — output of the most recent experiment
progress.png    — chart of solve_rate over experiments
report.md       — comprehensive experiment report
```

## Quick start on Lightning.ai with an H100

1. Create a new Lightning Studio with an H100 GPU.

2. Verify Python version:

```bash
python3 --version
# Must print 3.12.x or higher
```

3. Install uv (if not already available):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

4. Install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code@latest
```

5. Set your Anthropic API key. The key must be the full key from console.anthropic.com (approximately 108 characters, starting with `sk-ant-api03-`). Export it without quotes around the value:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-YOUR_FULL_KEY_HERE
```

Verify the key length — it must print a number between 100 and 120, not 0 or 20:

```bash
echo -n "$ANTHROPIC_API_KEY" | wc -c
```

Add the export line to your `~/.bashrc` so it persists across terminal sessions.

6. Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/autoresearch-maze.git
cd autoresearch-maze
uv sync
```

7. Verify the game engine and benchmark suite:

```bash
uv run prepare.py
```

All self-tests should pass and the benchmark suite (200 mazes across 5 tiers) should build successfully.

8. Optionally run a manual training test:

```bash
uv run train.py
```

This trains for 10 minutes and prints the solve_rate. Useful for verifying GPU access and getting a baseline.

9. Create the logs directory:

```bash
mkdir -p logs
```

10. Launch the autonomous experiment loop:

```bash
claude -p "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first." --dangerously-skip-permissions --max-turns 1000
```

This must be a single line with no backslash continuations. The `-p` flag forces headless mode which uses your API key directly and bypasses OAuth (Lightning.ai terminals cannot open a browser for OAuth login). The `--max-turns 1000` gives the agent enough headroom to run overnight. At roughly 6 experiments per hour (10 minutes each), expect about 50 experiments over 8 hours.

## Troubleshooting

### "Invalid API key" or "Fix external API key" error

If Claude Code refuses to start with an API key error, run these diagnostic steps in order:

Check key length (must print 100-120):
```bash
echo -n "$ANTHROPIC_API_KEY" | wc -c
```

Check for hidden characters or extra quotes:
```bash
echo "$ANTHROPIC_API_KEY" | cat -A
```
The output should start with `sk-ant-api03-` and end with `$` (no trailing whitespace or special characters). If you see `"` or `'` characters, re-export without quotes.

Clear stale OAuth cache:
```bash
rm -rf ~/.claude ~/.config/claude-code
```

Update Claude Code to the latest version:
```bash
npm install -g @anthropic-ai/claude-code@latest
```

Test the key directly via curl:
```bash
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' | head -c 200
```
You should see a JSON response with Claude's reply. If you see an authentication error, the key itself is invalid — generate a new one at console.anthropic.com.

As a last resort, try passing the key inline:
```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY claude -p "Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first." --dangerously-skip-permissions --max-turns 1000
```

### The "y y y y" output

When `--dangerously-skip-permissions` is working correctly, Claude Code auto-prints "y" confirmations as it approves each tool use (file edits, shell commands). This "y y y y" output scrolling by is normal and expected. It means the agent is running autonomously without waiting for human confirmation at each step.

If you find yourself manually typing "y" each time a tool use comes up, the flag is not taking effect. The most common cause is a line break in the launch command. Run the entire command on a single line with no backslash continuations. Alternatively, pre-create the settings file:

```bash
mkdir -p ~/.claude
echo '{"permissions": {"defaultMode": "bypassPermissions"}}' > ~/.claude/settings.json
```

Then launch Claude Code again.

## How evaluation works

The Solver is evaluated against a fixed set of 200 benchmark mazes that never change, regardless of what the Generator is currently producing. This is critical because the Generator's difficulty changes during training — if the metric were "solve rate against the current Generator," it could improve simply because the Generator got worse, which is not real progress.

The 200 benchmarks span five difficulty tiers of 40 mazes each: trivial (10-15% wall density, short optimal paths), easy (20-25%), medium (30-35%), hard (35-40%), and very hard (40-50% density, long winding optimal paths). During evaluation, the Solver uses greedy action selection (argmax, no sampling) with a fixed random seed. A maze counts as solved if the Solver reaches the goal within 100 steps. The `solve_rate` is the fraction of all 200 mazes solved, ranging from 0.0 to 1.0.

The per-tier breakdown is especially informative. Early in training, the Solver typically solves trivial and easy mazes first. Improvement on hard and very hard mazes indicates that the adversarial training is genuinely teaching the Solver to navigate complex structures, not just memorizing simple corridors.

## The frontier

These are the breakthroughs the agent should discover overnight.

**Recurrent memory.** The current Solver sees the visited-cells channel (a binary map of where it has been), but this is a spatial snapshot — it does not encode the order of visits or the Solver's trajectory. An LSTM or GRU that maintains hidden state across steps within an episode would give the Solver true sequential memory, letting it reason about dead ends it has already explored and backtrack intelligently.

**Hierarchical planning.** Instead of choosing one step at a time, the Solver could predict a subgoal (an intermediate cell to reach first), navigate to it, then predict the next subgoal. This decomposes the problem into easier pieces and is especially useful for large mazes with long optimal paths.

**Optimal adversarial balance.** The ratio of Solver updates to Generator updates (currently 5:1) fundamentally shapes training dynamics. Too many Solver updates and the Generator cannot keep up, producing mazes that are too easy. Too few and the Solver cannot adapt, making all mazes seem too hard. Finding the sweet spot, and whether it should change during training, is a key research question.

## Design choices

**11×11 grid.** Large enough for interesting mazes with dead ends, loops, and multiple path options. Small enough that episodes complete quickly (under 100 steps typically), letting the agent run thousands of episodes within the 10-minute time budget.

**10 minutes instead of 5.** Adversarial training has two networks that must co-adapt. Both need enough episodes to respond to each other's changes before the experiment ends. Five minutes was tight; ten gives roughly twice as many training cycles and produces more stable signal about whether a change actually helped.

**Chinese wall rules.** Any wall placement is valid as long as at least one path exists from start to goal, verified by BFS. There are no constraints on wall patterns, symmetry, or structure. This maximizes the Generator's creative freedom while ensuring every maze is solvable.

**Fixed benchmark.** The evaluation uses mazes generated once with fixed seeds, never changing across experiments. This decouples the metric from the Generator's current behavior and ensures that improvements in `solve_rate` represent genuine Solver capability, not fluctuations in maze difficulty.

## License

MIT
