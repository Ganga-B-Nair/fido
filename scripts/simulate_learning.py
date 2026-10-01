"""Headless learning-curve simulation — Role 3's test bench and the "proof it learns" slide.

Runs the real RuleEngine + ContextualBandit against the SimulatedUser with no sleeping,
then compares against a random-nudge baseline.

    python scripts/simulate_learning.py                  # 300 nudges, UCB vs random
    python scripts/simulate_learning.py --plot curve.png # also save a chart (needs matplotlib)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fido.agent.bandit import ContextualBandit, NudgeTracker  # noqa: E402
from fido.config import demo_config  # noqa: E402
from fido.contracts import Nudge  # noqa: E402
from fido.rules.engine import RuleEngine  # noqa: E402
from fido.sensing.fusion import Fusion  # noqa: E402
from fido.sensing.sim import SimulatedUser  # noqa: E402


def run(policy: str, n_nudges: int, seed: int, tmp: Path):
    cfg = demo_config()
    cfg.agent.algorithm = policy if policy != "random" else "ucb"
    cfg.agent.state_path = tmp / f"state_{policy}.json"
    cfg.agent.log_path = tmp / f"log_{policy}.csv"
    user = SimulatedUser(seed=seed)
    fusion, rules = Fusion(cfg.sensing, cfg.rules.smoothing_window), RuleEngine(cfg.rules)
    bandit, tracker = ContextualBandit(cfg.agent, seed=seed), NudgeTracker(cfg.agent)
    rng = random.Random(seed)
    rewards, t, dt = [], 0.0, cfg.tick_s
    while len(rewards) < n_nudges:
        state = fusion.update(user.step(t, dt))
        out = tracker.check(state, t)
        if out:
            bandit.update(out.context, out.nudge, out.reward)
            rewards.append(1.0 if out.reengaged else 0.0)
        trig = rules.update(state, t, tracker.busy)
        if trig:
            nudge = rng.choice(list(Nudge)) if policy == "random" else bandit.select(trig.context)
            tracker.start(nudge, trig, t)
            user.receive_nudge(nudge, t)
        t += dt
    return rewards, bandit


def rolling(xs, w=30):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nudges", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    tmp = Path(__file__).resolve().parent.parent / "logs" / "sim_runs"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*"):
        f.unlink()

    results = {}
    for policy in ("random", "epsilon", "ucb"):
        rewards, bandit = run(policy, args.nudges, args.seed, tmp)
        results[policy] = rewards
        first, last = rewards[:50], rewards[-50:]
        print(f"{policy:8s} success rate  first 50: {sum(first)/len(first):.0%}   "
              f"last 50: {sum(last)/len(last):.0%}   overall: {sum(rewards)/len(rewards):.0%}")
        if policy != "random":
            print("         learned:\n         " + bandit.summary().replace("\n", "\n         "))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for policy, rewards in results.items():
            ax.plot(rolling(rewards), label=policy, linewidth=2 if policy == "ucb" else 1.3)
        ax.set_xlabel("Nudge number")
        ax.set_ylabel("Re-engagement rate (rolling 30)")
        ax.set_ylim(0, 1)
        ax.set_title("FIDO learns which nudge works")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
