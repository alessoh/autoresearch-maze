# autoresearch-maze

This is an experiment to have the LLM do its own research on adversarial maze generation and solving.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, maze environment, BFS, benchmark suite, evaluation. Do not modify.
   - `train.py` — the file you modify. Generator and Solver networks, adversarial training loop, all hyperparameters.
4. **Verify prepare.py**: Run `uv run prepare.py` to confirm the self-tests pass and the benchmark suite builds correctly.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time). You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: both network architectures, optimizer, hyperparameters, training loop, batch size, model sizes, reward shaping, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, maze environment, BFS utilities, and benchmark suite.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_solve_rate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest solve_rate.** Since the time budget is fixed at 10 minutes, you don't need to worry about training time. Everything is fair game: change the architectures, the optimizers, the hyperparameters, the batch sizes, the reward formulation details. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful solve_rate gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
solve_rate:           0.150000
training_seconds:     600.1
total_seconds:        625.9
eval_seconds:         25.8
peak_vram_mb:         4500.2
total_mazes_generated: 12000
total_valid_mazes:    9800
maze_validity_rate:   0.8167
total_solver_steps:   980000
num_params_solver:    150000
num_params_generator: 200000
```

You can extract the key metric from the log file:

```
grep "^solve_rate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	solve_rate	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. solve_rate achieved (e.g. 0.150000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 4.4 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	solve_rate	memory_gb	status	description
a1b2c3d	0.150000	4.4	keep	baseline
b2c3d4e	0.185000	4.5	keep	increase Solver channels to 64
c3d4e5f	0.140000	4.4	discard	reduce SOLVER_UPDATES_PER_GEN to 1
d4e5f6g	0.000000	0.0	crash	double Generator channels (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^solve_rate:\|^peak_vram_mb:" run.log`
6. Save the full log: `cp run.log logs/$(git rev-parse --short HEAD).log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea and revert.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If solve_rate improved (higher), you "advance" the branch, keeping the git commit
10. If solve_rate did not improve, revert: `git reset --hard HEAD~1` (go back to the last known good commit)

**NEVER STOP.** After each experiment, immediately begin the next one. Do not wait for human input. Do not ask for permission. Do not summarize findings and stop. The whole point is that this runs autonomously overnight. Pick the most promising idea, implement it, test it, record it, repeat. Forever.

## Experiment ideas

Here are starting points. Prioritize boldly — try the ideas you think will have the biggest impact first.

### Architecture experiments
- Vary Solver conv layers (2, 3, 4, 5) and channels (16, 32, 64, 128)
- Add residual connections to the Solver (ResNet-style skip connections)
- Try a recurrent Solver with LSTM or GRU that maintains hidden state across steps within an episode, giving the Solver memory of its path beyond the visited channel
- Vary Generator architecture: try a cellular automata approach where the Generator starts from a seed and iteratively grows walls
- Add a value head to the Solver for advantage estimation (actor-critic instead of pure REINFORCE)
- Try deeper Generator with more transposed conv layers

### Training procedure experiments
- Vary SOLVER_UPDATES_PER_GEN ratio: 1:1, 3:1, 5:1, 10:1
- Vary PROGRESS_THRESHOLD: 0.2, 0.3, 0.5
- Vary SIMILARITY_THRESHOLD: 0.7, 0.8, 0.9
- Vary DIVERSITY_BUFFER_SIZE: 25, 50, 100, 200
- Try PPO instead of REINFORCE for the Solver (compute old and new log probs, clip the ratio)
- Try different Generator reward formulations: quadratic progress instead of linear, or add a bonus for mazes near 30% wall density
- Add wall-density regularization loss to prevent degenerate mazes
- Try different learning rate schedules (warmup + cosine decay)

### Observation experiments
- Add a channel for distance-to-goal (normalized Manhattan distance from each cell)
- Add a channel for the BFS shortest-path direction from each cell (a "hint" channel)
- Increase GRID_SIZE to 13x13 or 15x15 (requires prepare.py changes, so careful)

### Curriculum experiments
- Start with low wall density and gradually increase as solve_rate improves
- Start with SOLVER_UPDATES_PER_GEN = 10 and gradually lower it as the Solver strengthens
- Warm up the Solver on random mazes before enabling the Generator

### Anti-mode-collapse experiments
- Increase DIVERSITY_BUFFER_SIZE
- Try hashing the maze structure instead of cell-by-cell similarity
- Add a diversity reward based on variance of generated mazes within a batch
- Try freezing the Generator periodically to let the Solver catch up

### Reward shaping experiments
- Vary step penalty: -0.005, -0.01, -0.02
- Add a reward shaping term for reducing distance to goal at each step
- Try giving partial reward proportional to progress even when the Solver solves the maze
- Experiment with different discount factors: 0.95, 0.99, 0.999
