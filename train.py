"""
Adversarial maze generation and solving — the mutable artifact.

Two networks train against each other:
  - Generator: produces 11x11 mazes from random noise
  - Solver: navigates mazes to reach the goal

The autoresearch agent modifies this file. Everything is fair game:
architecture, hyperparameters, training loop, batch size, etc.

Usage:
    uv run train.py
"""

import os
import sys
import time
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    GRID_SIZE, START_POS, GOAL_POS, MAX_STEPS, TIME_BUDGET,
    NUM_STATE_CHANNELS, ACTION_DELTAS,
    MazeEnvironment, shortest_path, is_valid_maze,
    evaluate_solve_rate,
)

# ---------------------------------------------------------------------------
# Hyperparameters (clearly labeled — agent experiments with these)
# ---------------------------------------------------------------------------

LATENT_DIM = 64                  # Generator noise dimension
GEN_CHANNELS = 128               # Generator hidden channels
SOLVER_CHANNELS = 32             # Solver conv channels
SOLVER_FC_HIDDEN = 128           # Solver fully-connected hidden units
GEN_LR = 3e-4                   # Generator learning rate
SOLVER_LR = 1e-3                 # Solver learning rate
WALL_DENSITY_TARGET = 0.30       # Soft target for wall density regularization
DENSITY_REG_COEF = 0.1           # Strength of wall density regularization
ENTROPY_COEF = 0.01              # Entropy bonus for Solver
SOLVER_UPDATES_PER_GEN = 5       # Solver batches per Generator batch
MAZES_PER_BATCH = 16             # Mazes generated per training cycle
DIVERSITY_BUFFER_SIZE = 50       # Rolling buffer for diversity check
PROGRESS_THRESHOLD = 0.3         # Minimum progress for Generator reward
SIMILARITY_THRESHOLD = 0.8       # Above this, diversity penalty kicks in
GAMMA = 0.99                     # Discount factor for Solver returns
ADV_CLIP = 5.0                   # Advantage clipping range
LOG_PROB_CLIP = 0.2              # Log-probability clipping (PPO-style)
MAX_GEN_ATTEMPTS = 10            # Max attempts to generate a valid maze
LOG_INTERVAL = 10                # Print every N steps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Generator Network
# ---------------------------------------------------------------------------

class GeneratorNetwork(nn.Module):
    """Takes a latent vector and outputs an 11x11 grid of wall probabilities."""

    def __init__(self, latent_dim=LATENT_DIM, hidden_channels=GEN_CHANNELS):
        super().__init__()
        # Project latent to spatial representation
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 4 * 4),
            nn.ReLU(),
        )
        # Upsample from 4x4 to ~11x11 via transposed convolutions
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
        )
        self.hidden_channels = hidden_channels

        # Initialize final conv bias so sigmoid outputs ~30% wall density
        # sigmoid(-0.85) ≈ 0.30, which gives ~30% walls on average
        final_conv = self.deconv[-1]
        nn.init.constant_(final_conv.bias, -0.85)
        nn.init.xavier_uniform_(final_conv.weight, gain=0.1)

    def forward(self, z):
        """z: (batch, latent_dim) -> wall_probs: (batch, GRID_SIZE, GRID_SIZE)"""
        batch = z.shape[0]
        x = self.fc(z)
        x = x.view(batch, self.hidden_channels, 4, 4)
        x = self.deconv(x)  # (batch, 1, 16, 16) approximately
        # Crop or interpolate to exact GRID_SIZE
        x = F.interpolate(x, size=(GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=False)
        x = x.squeeze(1)  # (batch, GRID_SIZE, GRID_SIZE)
        return torch.sigmoid(x)

    def sample_mazes(self, batch_size, temperature=1.0):
        """Generate a batch of binary maze grids.

        Returns:
          wall_probs: (batch, GRID_SIZE, GRID_SIZE) — continuous probabilities
          wall_grids: list of numpy boolean arrays — thresholded grids
        """
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE) * temperature
        wall_probs = self.forward(z)

        wall_grids = []
        for i in range(batch_size):
            grid = (wall_probs[i].detach().cpu().numpy() > 0.5).astype(bool)
            # Always clear start and goal
            grid[START_POS[0], START_POS[1]] = False
            grid[GOAL_POS[0], GOAL_POS[1]] = False
            wall_grids.append(grid)

        return wall_probs, wall_grids


# ---------------------------------------------------------------------------
# Solver Network
# ---------------------------------------------------------------------------

class SolverNetwork(nn.Module):
    """Takes a state tensor and outputs logits over 4 actions."""

    def __init__(self, in_channels=NUM_STATE_CHANNELS, conv_channels=SOLVER_CHANNELS,
                 fc_hidden=SOLVER_FC_HIDDEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat_size = conv_channels * 2 * GRID_SIZE * GRID_SIZE
        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 4),
        )

    def forward(self, x):
        """x: (batch, channels, H, W) -> logits: (batch, 4)"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.conv(x)
        features = features.view(features.shape[0], -1)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Diversity Buffer
# ---------------------------------------------------------------------------

class DiversityBuffer:
    """Rolling buffer of recent mazes for diversity checking."""

    def __init__(self, max_size=DIVERSITY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=max_size)

    def compute_diversity_multiplier(self, grid: np.ndarray) -> float:
        """Compute diversity multiplier for a new maze.

        Returns 1.0 for novel mazes, decaying to 0.0 for duplicates.
        """
        if len(self.buffer) == 0:
            return 1.0

        flat_new = grid.flatten().astype(float)
        max_sim = 0.0

        for stored in self.buffer:
            flat_stored = stored.flatten().astype(float)
            similarity = np.mean(flat_new == flat_stored)
            if similarity > max_sim:
                max_sim = similarity

        if max_sim < SIMILARITY_THRESHOLD:
            return 1.0
        else:
            return max(0.0, (1.0 - max_sim) / (1.0 - SIMILARITY_THRESHOLD))

    def add(self, grid: np.ndarray):
        """Add a maze to the buffer."""
        self.buffer.append(grid.copy())


# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_solver_episode(solver, env, device=DEVICE):
    """Run the Solver through one maze episode.

    Returns:
      states: list of state tensors
      actions: list of action indices
      log_probs: list of log probabilities
      reached_goal: bool
      progress: float (0.0 to 1.0)
      num_steps: int
    """
    states = []
    actions = []
    log_probs = []

    state = env.get_state_tensor(device)
    done = False

    while not done:
        states.append(state)
        logits = solver(state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        lp = dist.log_prob(action)

        actions.append(action.item())
        log_probs.append(lp)

        state, reached_goal, done = env.step(action.item())
        state = state.to(device)

    progress = env.compute_progress()
    return states, actions, log_probs, env.reached_goal, progress, env.steps_taken


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train():
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Mazes per batch: {MAZES_PER_BATCH}")
    print(f"Solver updates per Generator update: {SOLVER_UPDATES_PER_GEN}")
    print()

    # Initialize networks
    generator = GeneratorNetwork().to(DEVICE)
    solver = SolverNetwork().to(DEVICE)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=GEN_LR)
    solver_optimizer = torch.optim.Adam(solver.parameters(), lr=SOLVER_LR)

    diversity_buffer = DiversityBuffer()

    num_params_solver = sum(p.numel() for p in solver.parameters())
    num_params_gen = sum(p.numel() for p in generator.parameters())
    print(f"Solver parameters: {num_params_solver:,}")
    print(f"Generator parameters: {num_params_gen:,}")
    print()

    # Training stats
    step = 0
    total_mazes_generated = 0
    total_valid_mazes = 0
    total_solver_steps = 0
    solver_update_count = 0

    train_start = time.time()

    while True:
        elapsed = time.time() - train_start
        if elapsed >= TIME_BUDGET:
            break

        step += 1

        # --- Generate mazes ---
        generator.eval()
        wall_probs, wall_grids = generator.sample_mazes(MAZES_PER_BATCH)
        generator.train()

        # Validate and collect episodes
        valid_mazes = []
        valid_probs = []
        for i, grid in enumerate(wall_grids):
            total_mazes_generated += 1
            if is_valid_maze(grid):
                total_valid_mazes += 1
                valid_mazes.append(grid)
                valid_probs.append(wall_probs[i])

        if len(valid_mazes) == 0:
            if step % LOG_INTERVAL == 0:
                print(f"Step {step}: no valid mazes generated, skipping")
            continue

        # --- Run Solver on each valid maze ---
        batch_solver_log_probs = []
        batch_solver_returns = []
        batch_gen_rewards = []
        batch_progresses = []

        for idx, grid in enumerate(valid_mazes):
            env = MazeEnvironment(grid)
            states, actions, log_probs_ep, reached_goal, progress, n_steps = \
                run_solver_episode(solver, env, DEVICE)
            total_solver_steps += n_steps

            # Compute Solver rewards per step
            step_rewards = []
            for t in range(len(actions)):
                if t == len(actions) - 1 and reached_goal:
                    step_rewards.append(1.0)
                elif t == len(actions) - 1 and not reached_goal:
                    step_rewards.append(-0.5)
                else:
                    step_rewards.append(-0.01)  # step penalty

            # Compute discounted returns
            returns = []
            G = 0.0
            for r in reversed(step_rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            returns_t = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
            log_probs_t = torch.stack(log_probs_ep).squeeze()

            batch_solver_log_probs.append(log_probs_t)
            batch_solver_returns.append(returns_t)

            # Generator reward (Rule 1 + Rule 2)
            if reached_goal:
                gen_progress_reward = 0.0
            elif progress < PROGRESS_THRESHOLD:
                gen_progress_reward = 0.0
            else:
                gen_progress_reward = progress

            diversity_mult = diversity_buffer.compute_diversity_multiplier(grid)
            gen_reward = gen_progress_reward * diversity_mult
            batch_gen_rewards.append(gen_reward)
            batch_progresses.append(progress)

            diversity_buffer.add(grid)

        # --- Update Solver ---
        solver_optimizer.zero_grad()
        solver_loss = torch.tensor(0.0, device=DEVICE)

        for log_probs_t, returns_t in zip(batch_solver_log_probs, batch_solver_returns):
            if len(returns_t) == 0:
                continue
            # Normalize advantages
            advantages = returns_t
            if len(advantages) > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std
                advantages = advantages.clamp(-ADV_CLIP, ADV_CLIP)

            # Clipped log-probs
            clipped_lp = log_probs_t.clamp(-LOG_PROB_CLIP, 0.0)

            # Policy gradient loss
            pg_loss = -(clipped_lp * advantages.detach()).mean()

            # Entropy bonus
            entropy = -log_probs_t.mean()

            solver_loss = solver_loss + pg_loss - ENTROPY_COEF * entropy

        solver_loss = solver_loss / max(len(batch_solver_log_probs), 1)
        solver_loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        solver_optimizer.step()
        solver_update_count += 1

        # --- Update Generator (every SOLVER_UPDATES_PER_GEN Solver updates) ---
        if solver_update_count % SOLVER_UPDATES_PER_GEN == 0 and len(batch_gen_rewards) > 0:
            gen_optimizer.zero_grad()

            # Recompute wall probs for gradient flow
            z = torch.randn(len(valid_mazes), LATENT_DIM, device=DEVICE)
            gen_wall_probs = generator(z)

            # Generator loss: maximize reward -> minimize -reward * log_prob
            gen_rewards_t = torch.tensor(batch_gen_rewards, device=DEVICE, dtype=torch.float32)

            # Normalize generator rewards
            if len(gen_rewards_t) > 1 and gen_rewards_t.std() > 1e-8:
                gen_rewards_t = (gen_rewards_t - gen_rewards_t.mean()) / (gen_rewards_t.std() + 1e-8)

            # Use wall density as a soft regularization target
            mean_density = gen_wall_probs.mean()
            density_loss = DENSITY_REG_COEF * (mean_density - WALL_DENSITY_TARGET) ** 2

            # REINFORCE-style loss for Generator
            # Encourage generating walls where the reward was high
            gen_loss = -(gen_rewards_t.mean()) + density_loss

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            gen_optimizer.step()

        # --- Logging ---
        if step % LOG_INTERVAL == 0:
            validity_rate = total_valid_mazes / max(total_mazes_generated, 1)
            avg_progress = np.mean(batch_progresses) if batch_progresses else 0.0
            avg_gen_reward = np.mean(batch_gen_rewards) if batch_gen_rewards else 0.0
            elapsed = time.time() - train_start
            print(f"Step {step}: solver_loss={solver_loss.item():.4f} "
                  f"gen_reward={avg_gen_reward:.4f} "
                  f"validity={validity_rate:.3f} "
                  f"progress={avg_progress:.3f} "
                  f"elapsed={elapsed:.1f}s")

    training_seconds = time.time() - train_start
    print(f"\nTraining complete. {training_seconds:.1f}s elapsed.")
    print(f"Total mazes generated: {total_mazes_generated}")
    print(f"Total valid mazes: {total_valid_mazes}")
    maze_validity_rate = total_valid_mazes / max(total_mazes_generated, 1)
    print(f"Maze validity rate: {maze_validity_rate:.4f}")
    print(f"Total Solver steps: {total_solver_steps}")
    print()

    # --- Evaluation ---
    print("Evaluating on fixed benchmark...")
    eval_start = time.time()
    results = evaluate_solve_rate(solver, device=DEVICE)
    eval_seconds = time.time() - eval_start

    total_seconds = time.time() - train_start

    # Peak VRAM
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    # --- Print results ---
    print()
    print("--- Per-tier breakdown ---")
    for tier in ["trivial", "easy", "medium", "hard", "vhard"]:
        print(f"  {tier}: {results['per_tier'][tier]:.4f}")

    print()
    print("---")
    print(f"solve_rate:           {results['solve_rate']:.6f}")
    print(f"training_seconds:     {training_seconds:.1f}")
    print(f"total_seconds:        {total_seconds:.1f}")
    print(f"eval_seconds:         {eval_seconds:.1f}")
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
    print(f"total_mazes_generated: {total_mazes_generated}")
    print(f"total_valid_mazes:    {total_valid_mazes}")
    print(f"maze_validity_rate:   {maze_validity_rate:.4f}")
    print(f"total_solver_steps:   {total_solver_steps}")
    print(f"num_params_solver:    {num_params_solver}")
    print(f"num_params_generator: {num_params_gen}")


if __name__ == "__main__":
    train()
