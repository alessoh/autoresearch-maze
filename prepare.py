"""
Immutable harness for autoresearch-maze.

Contains: maze environment, BFS utilities, fixed benchmark suite,
and the sacred evaluation metric (solve_rate).

DO NOT MODIFY THIS FILE. The autoresearch agent only modifies train.py.

Usage:
    python prepare.py              # run self-tests
    python prepare.py --benchmark  # time a full 200-maze eval with random policy
"""

import argparse
import collections
import random
import time
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

GRID_SIZE = 11
START_POS = (1, 1)
GOAL_POS = (9, 9)
MAX_STEPS = 100
TIME_BUDGET = 600          # 10 minutes
NUM_BENCHMARK_MAZES = 200

# Actions: 0=up, 1=down, 2=left, 3=right
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# State tensor channels
# 0: walls, 1: current position, 2: goal position, 3: visited cells
NUM_STATE_CHANNELS = 4

# ---------------------------------------------------------------------------
# BFS Utilities
# ---------------------------------------------------------------------------

def shortest_path(grid: np.ndarray, start: tuple, goal: tuple) -> int:
    """Return length of shortest path from start to goal, or -1 if none exists.

    grid is a 2D boolean array where True means wall.
    Path length counts the number of steps (edges), not cells.
    """
    if grid[start[0], start[1]] or grid[goal[0], goal[1]]:
        return -1
    if start == goal:
        return 0

    rows, cols = grid.shape
    visited = set()
    visited.add(start)
    queue = collections.deque()
    queue.append((start, 0))

    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and not grid[nr, nc]:
                if (nr, nc) == goal:
                    return dist + 1
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1


def is_valid_maze(grid: np.ndarray) -> bool:
    """Return True if a path exists from START_POS to GOAL_POS."""
    return shortest_path(grid, START_POS, GOAL_POS) != -1


def generate_random_maze(density: float, rng: Optional[random.Random] = None) -> np.ndarray:
    """Create a random maze with approximately the given wall density, guaranteed valid.

    Regenerates until a valid maze is produced.
    """
    if rng is None:
        rng = random.Random()

    while True:
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) == START_POS or (r, c) == GOAL_POS:
                    continue
                if rng.random() < density:
                    grid[r, c] = True
        if is_valid_maze(grid):
            return grid


# ---------------------------------------------------------------------------
# Maze Environment
# ---------------------------------------------------------------------------

class MazeEnvironment:
    """Handles a single maze episode for the Solver."""

    def __init__(self, grid: np.ndarray):
        """Initialize with a wall grid. Caller must ensure grid is valid."""
        self.grid = grid.copy()
        self.pos = START_POS
        self.steps_taken = 0
        self.reached_goal = False
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.visited[self.pos[0], self.pos[1]] = True

        # Track closest approach (Manhattan distance to goal)
        self.closest_approach = abs(self.pos[0] - GOAL_POS[0]) + abs(self.pos[1] - GOAL_POS[1])
        self.sp_length = shortest_path(grid, START_POS, GOAL_POS)

    def get_state_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return state as a (NUM_STATE_CHANNELS, GRID_SIZE, GRID_SIZE) float tensor.

        Channels:
          0 - walls (1.0 where wall, 0.0 where empty)
          1 - current position (one-hot)
          2 - goal position (one-hot)
          3 - visited cells
        """
        state = np.zeros((NUM_STATE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[0] = self.grid.astype(np.float32)
        state[1, self.pos[0], self.pos[1]] = 1.0
        state[2, GOAL_POS[0], GOAL_POS[1]] = 1.0
        state[3] = self.visited.astype(np.float32)
        return torch.from_numpy(state).to(device)

    def step(self, action: int) -> tuple:
        """Take one step. Returns (new_state_tensor, reached_goal, done).

        If the move hits a wall or grid edge, the Solver stays in place.
        """
        if self.reached_goal or self.steps_taken >= MAX_STEPS:
            return self.get_state_tensor(), self.reached_goal, True

        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc

        # Check bounds and walls
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not self.grid[nr, nc]:
            self.pos = (nr, nc)

        self.steps_taken += 1
        self.visited[self.pos[0], self.pos[1]] = True

        # Update closest approach
        dist = abs(self.pos[0] - GOAL_POS[0]) + abs(self.pos[1] - GOAL_POS[1])
        if dist < self.closest_approach:
            self.closest_approach = dist

        # Check goal
        if self.pos == GOAL_POS:
            self.reached_goal = True
            return self.get_state_tensor(), True, True

        done = self.steps_taken >= MAX_STEPS
        return self.get_state_tensor(), False, done

    def compute_progress(self) -> float:
        """Compute Solver progress as fraction of shortest path covered.

        progress = 1.0 - (closest_approach / shortest_path_length)
        Clamped to [0.0, 1.0].
        """
        if self.sp_length <= 0:
            return 0.0
        progress = 1.0 - (self.closest_approach / self.sp_length)
        return max(0.0, min(1.0, progress))


# ---------------------------------------------------------------------------
# Fixed Benchmark Suite
# ---------------------------------------------------------------------------

def _build_benchmark_suite() -> list:
    """Generate 200 fixed benchmark mazes across 5 difficulty tiers.

    Uses fixed random seeds so the benchmark never changes.
    Tiers (40 mazes each):
      Trivial:    10-15% wall density, short optimal paths
      Easy:       20-25% wall density
      Medium:     30-35% wall density
      Hard:       35-40% wall density
      Very Hard:  40-50% wall density, long winding optimal paths
    """
    tiers = [
        ("trivial", 0.10, 0.15, 40),
        ("easy",    0.20, 0.25, 40),
        ("medium",  0.30, 0.35, 40),
        ("hard",    0.35, 0.40, 40),
        ("vhard",   0.40, 0.50, 40),
    ]

    mazes = []
    seed_counter = 42000

    for tier_name, lo, hi, count in tiers:
        tier_mazes = []
        while len(tier_mazes) < count:
            rng = random.Random(seed_counter)
            seed_counter += 1
            density = lo + rng.random() * (hi - lo)
            grid = generate_random_maze(density, rng)
            sp = shortest_path(grid, START_POS, GOAL_POS)
            if sp > 0:
                tier_mazes.append({
                    "grid": grid,
                    "tier": tier_name,
                    "density": density,
                    "shortest_path": sp,
                })
        mazes.extend(tier_mazes)

    return mazes


BENCHMARK_MAZES = _build_benchmark_suite()
TIER_NAMES = ["trivial", "easy", "medium", "hard", "vhard"]


# ---------------------------------------------------------------------------
# Sacred Metric — DO NOT CHANGE
# ---------------------------------------------------------------------------

def evaluate_solve_rate(solver_model, device: str = "cuda") -> dict:
    """Evaluate the Solver on all 200 fixed benchmark mazes.

    Uses greedy action selection (argmax) with a fixed random seed.
    Returns a dict with:
      - solve_rate: fraction of all 200 mazes solved (0.0 to 1.0)
      - per_tier: dict of tier_name -> fraction solved
      - total_solved: int
      - total_mazes: int
    """
    solver_model.eval()
    torch.manual_seed(999)

    tier_solved = {t: 0 for t in TIER_NAMES}
    tier_total = {t: 0 for t in TIER_NAMES}
    total_solved = 0

    with torch.no_grad():
        for maze_info in BENCHMARK_MAZES:
            env = MazeEnvironment(maze_info["grid"])
            state = env.get_state_tensor(device)
            tier = maze_info["tier"]
            tier_total[tier] += 1

            done = False
            while not done:
                logits = solver_model(state.unsqueeze(0))
                action = logits.argmax(dim=-1).item()
                state, reached_goal, done = env.step(action)
                state = state.to(device)

            if env.reached_goal:
                total_solved += 1
                tier_solved[tier] += 1

    solver_model.train()

    per_tier = {}
    for t in TIER_NAMES:
        per_tier[t] = tier_solved[t] / tier_total[t] if tier_total[t] > 0 else 0.0

    return {
        "solve_rate": total_solved / len(BENCHMARK_MAZES),
        "per_tier": per_tier,
        "total_solved": total_solved,
        "total_mazes": len(BENCHMARK_MAZES),
    }


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

def _run_self_tests():
    """Verify correctness of all components."""
    print("Running self-tests...")
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    # Test 1: BFS finds correct shortest path on a known maze
    grid1 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    sp1 = shortest_path(grid1, START_POS, GOAL_POS)
    expected_sp = abs(GOAL_POS[0] - START_POS[0]) + abs(GOAL_POS[1] - START_POS[1])
    check("BFS shortest path on empty grid", sp1 == expected_sp)

    # Test 2: BFS returns -1 on maze with no path
    grid2 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    # Block all cells around the goal
    for dr, dc in ACTION_DELTAS:
        nr, nc = GOAL_POS[0] + dr, GOAL_POS[1] + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            grid2[nr, nc] = True
    sp2 = shortest_path(grid2, START_POS, GOAL_POS)
    check("BFS returns -1 for blocked maze", sp2 == -1)

    # Test 3: Environment blocks wall movement
    grid3 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid3[0, 1] = True  # wall above start
    env3 = MazeEnvironment(grid3)
    old_pos = env3.pos
    env3.step(0)  # try to move up into wall
    check("Environment blocks wall movement", env3.pos == old_pos)

    # Test 4: Environment detects goal reached
    grid4 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    env4 = MazeEnvironment(grid4)
    # Walk to goal via direct path
    for _ in range(GOAL_POS[0] - START_POS[0]):
        env4.step(1)  # down
    for _ in range(GOAL_POS[1] - START_POS[1]):
        env4.step(3)  # right
    check("Environment detects goal reached", env4.reached_goal)

    # Test 5: Closest approach tracking
    grid5 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    env5 = MazeEnvironment(grid5)
    env5.step(1)  # move down (closer to goal)
    env5.step(0)  # move back up
    # After moving down once, closest approach should be initial - 1
    initial_dist = abs(START_POS[0] - GOAL_POS[0]) + abs(START_POS[1] - GOAL_POS[1])
    check("Closest approach tracked correctly", env5.closest_approach == initial_dist - 1)

    # Test 6: Progress = 0.0 when Solver does not move
    grid6 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    # Surround start with walls (but maze is technically valid via other paths)
    # Instead, just create env and don't step
    grid6b = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    env6 = MazeEnvironment(grid6b)
    # No steps taken, progress should be 0.0 (closest = initial distance = sp_length for empty grid)
    prog6 = env6.compute_progress()
    check("Progress = 0.0 when Solver stays at start", prog6 == 0.0)

    # Test 7: Progress = 1.0 when Solver reaches goal
    check("Progress = 1.0 when goal reached", env4.compute_progress() == 1.0)

    # Test 8: Intermediate progress
    grid8 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    env8 = MazeEnvironment(grid8)
    # Move halfway toward goal
    for _ in range(4):
        env8.step(1)  # down
    for _ in range(4):
        env8.step(3)  # right
    prog8 = env8.compute_progress()
    check("Progress between 0 and 1 for partial path", 0.3 < prog8 < 1.0)

    # Test 9: Benchmark suite has exactly 200 valid mazes
    check("Benchmark has 200 mazes", len(BENCHMARK_MAZES) == NUM_BENCHMARK_MAZES)
    all_valid = all(is_valid_maze(m["grid"]) for m in BENCHMARK_MAZES)
    check("All benchmark mazes are valid", all_valid)

    # Test 10: Benchmark spans 5 tiers
    tier_counts = {}
    for m in BENCHMARK_MAZES:
        tier_counts[m["tier"]] = tier_counts.get(m["tier"], 0) + 1
    check("Benchmark has 5 tiers of 40 mazes each",
          len(tier_counts) == 5 and all(v == 40 for v in tier_counts.values()))

    # Test 11: State tensor shape
    grid_t = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    env_t = MazeEnvironment(grid_t)
    st = env_t.get_state_tensor()
    check("State tensor shape correct",
          st.shape == (NUM_STATE_CHANNELS, GRID_SIZE, GRID_SIZE))

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed}")
    if failed > 0:
        raise SystemExit(1)


def _run_benchmark():
    """Time a full evaluation with a random policy."""
    print("Benchmarking evaluation with random policy...")

    class RandomSolver(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            return torch.randn(batch, 4)

    model = RandomSolver()
    start = time.time()
    results = evaluate_solve_rate(model, device="cpu")
    elapsed = time.time() - start

    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Solve rate: {results['solve_rate']:.4f}")
    for tier in TIER_NAMES:
        print(f"    {tier}: {results['per_tier'][tier]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze environment self-test and benchmark")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time a full 200-maze eval with random policy")
    args = parser.parse_args()

    _run_self_tests()
    print()

    if args.benchmark:
        _run_benchmark()
