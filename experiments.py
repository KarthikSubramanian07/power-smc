"""MATH500 driver: accuracy and wall-clock latency for baseline, MH, and Power-SMC.

This is the script that produces the headline artifact, the accuracy-vs-latency plot. It
needs a GPU and the model weights, so it is meant for Colab / Kaggle rather than the
correctness tests (those run on CPU via ``validate.py``).

Examples
--------
Baseline decoding on a 100-problem subset::

    python experiments.py run --method baseline --model Qwen/Qwen2.5-1.5B-Instruct \
        --subset 100 --out results/math500_baseline.csv

Power-SMC with 16 particles and alpha=4::

    python experiments.py run --method power-smc --model Qwen/Qwen2.5-1.5B-Instruct \
        --subset 100 --particles 16 --alpha 4 --out results/math500_power_smc.csv

Combine per-method CSVs into the tradeoff plot::

    python experiments.py plot --inputs results/math500_baseline.csv results/math500_power_smc.csv \
        --out results/plots/accuracy_latency.png
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Optional

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# --------------------------------------------------------------------------------------
# Answer extraction and grading
# --------------------------------------------------------------------------------------
def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the last \\boxed{...} in ``text`` (brace-balanced)."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i + 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def normalize_answer(s: str) -> str:
    """Light-weight math answer normalization for string comparison."""
    if s is None:
        return ""
    s = s.strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("$", ""),
                 ("\\%", ""), ("%", ""), ("\\ ", ""), (" ", "")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def answers_match(pred: Optional[str], gold: str) -> bool:
    """Grade a prediction against the gold answer.

    Uses ``math_verify`` when installed (more robust symbolic comparison) and otherwise
    falls back to normalized string and numeric comparison.
    """
    if pred is None:
        return False
    try:
        from math_verify import parse, verify  # type: ignore

        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        pass

    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_math500(subset: Optional[int] = None, seed: int = 0, shuffle: bool = True):
    """Load MATH500 (HuggingFaceH4/MATH-500), optionally a random subset.

    Pass ``shuffle=False`` to take the first ``subset`` problems in dataset order, which
    lines up with the MH reference's first-shard selection.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if subset is not None and subset < len(ds):
        if shuffle:
            ds = ds.shuffle(seed=seed)
        ds = ds.select(range(subset))
    return ds


# --------------------------------------------------------------------------------------
# Generation per method
# --------------------------------------------------------------------------------------
def generate(model, question: str, method: str, alpha: float, particles: int,
             kappa: float, max_tokens: int, temperature: float, ramp: int,
             seed: Optional[int]) -> str:
    from power_smc import power_smc
    from power_smc.baselines import baseline_decode
    from power_smc.ramping import LinearRamp

    prompt_ids = model.encode_chat(question, system=SYSTEM_PROMPT)

    if method == "power-smc":
        schedule = LinearRamp(alpha, t_ramp=ramp) if ramp > 0 else None
        result = power_smc(model, prompt_ids, alpha, n_particles=particles, kappa=kappa,
                           max_tokens=max_tokens, alpha_schedule=schedule, seed=seed)
        return model.decode_text(result.output)

    greedy = method == "greedy"
    seqs = baseline_decode(model, prompt_ids, n_samples=1,
                           temperature=temperature, max_tokens=max_tokens,
                           greedy=greedy, seed=seed)
    return model.decode_text(seqs[0])


def run_benchmark(args) -> None:
    from power_smc.hf_model import HFModel

    model = HFModel(args.model, device=args.device, dtype=args.dtype,
                    load_in_4bit=args.load_in_4bit)
    data = load_math500(subset=args.subset, seed=args.seed, shuffle=args.shuffle)
    if not data:
        print("No problems to run (empty subset).", file=sys.stderr)
        return

    rows = []
    n_correct = 0
    total_time = 0.0
    for i, ex in enumerate(data):
        t0 = time.perf_counter()
        output = generate(model, ex["problem"], args.method, args.alpha, args.particles,
                          args.kappa, args.max_tokens, args.temperature, args.ramp,
                          seed=(None if args.seed is None else args.seed + i))
        dt = time.perf_counter() - t0
        pred = extract_boxed(output)
        correct = answers_match(pred, ex["answer"])
        n_correct += int(correct)
        total_time += dt
        rows.append({
            "index": i,
            "method": args.method,
            "correct": int(correct),
            "seconds": round(dt, 4),
            "predicted": pred or "",
            "gold": ex["answer"],
        })
        print(f"[{i + 1}/{len(data)}] {args.method} correct={correct} "
              f"{dt:.1f}s pred={pred!r} gold={ex['answer']!r}")

    acc = n_correct / len(data)
    mean_latency = total_time / len(data)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{args.method}: accuracy={acc:.3f} mean_latency={mean_latency:.2f}s/problem "
          f"over {len(data)} problems -> {args.out}")


def run_mh(args) -> None:
    """Run the MH reference over MATH500 shards and write results in our CSV schema.

    Clones Karan & Du's repo if needed, runs their MATH500 power-sampling script for
    ``--shards`` shards of 100 problems each, times each shard, grades the ``mcmc_answer``
    column against the gold answer, and writes an experiments-style CSV so the MH point
    drops straight onto the accuracy-vs-latency plot. ``--alpha`` maps to their temperature
    as 1/alpha unless ``--temperature`` is given explicitly.
    """
    from power_smc.baselines import MHReference

    mh = MHReference(root=args.root)
    mh.clone()
    temperature = args.temperature if args.temperature is not None else 1.0 / args.alpha

    rows = []
    n_correct = 0
    total_time = 0.0
    for batch_idx in range(args.shards):
        t0 = time.perf_counter()
        mh.run_math(model=args.model, mcmc_steps=args.mcmc_steps, temperature=temperature,
                    batch_idx=batch_idx, seed=args.seed, save_str=args.save_str,
                    device=args.device)
        dt = time.perf_counter() - t0
        csv_path = mh.output_csv(args.model, args.mcmc_steps, temperature, batch_idx,
                                 args.seed, args.save_str)
        shard = MHReference.load_results(csv_path)
        per_problem = dt / max(len(shard), 1)
        for r in shard:
            correct = answers_match(r.get("mcmc_answer"), r.get("correct_answer", ""))
            n_correct += int(correct)
            rows.append({
                "index": len(rows),
                "method": "mh",
                "correct": int(correct),
                "seconds": round(per_problem, 4),
                "predicted": r.get("mcmc_answer", "") or "",
                "gold": r.get("correct_answer", "") or "",
            })
        total_time += dt
        print(f"[shard {batch_idx}] {len(shard)} problems in {dt:.1f}s")

    if not rows:
        print("No MH results to write.", file=sys.stderr)
        return

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc = n_correct / len(rows)
    print(f"\nmh: accuracy={acc:.3f} mean_latency={total_time / len(rows):.2f}s/problem "
          f"over {len(rows)} problems -> {args.out}")


def make_plot(args) -> None:
    """Build the accuracy-vs-latency plot from per-method CSVs.

    Latency is reported as an overhead multiple relative to the baseline method (the CSV
    whose method column is 'baseline' or 'greedy'), matching the paper's framing.
    """
    from power_smc.plotting import plot_accuracy_latency

    summaries = []
    for path in args.inputs:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"Empty CSV: {path}", file=sys.stderr)
            return
        method = rows[0]["method"]
        acc = sum(int(r["correct"]) for r in rows) / len(rows)
        latency = sum(float(r["seconds"]) for r in rows) / len(rows)
        summaries.append({"method": method, "accuracy": acc, "latency_s": latency})

    base = next((s for s in summaries if s["method"] in ("baseline", "greedy")), None)
    base_latency = base["latency_s"] if base else min(s["latency_s"] for s in summaries)
    points = [{"method": s["method"], "accuracy": s["accuracy"],
               "latency": s["latency_s"] / base_latency} for s in summaries]

    plot_accuracy_latency(points, args.out)
    for p in points:
        print(f"  {p['method']:>10}: acc={p['accuracy']:.3f} latency={p['latency']:.2f}x")
    print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATH500 accuracy/latency for Power-SMC")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a method over MATH500 and write a CSV")
    run.add_argument("--method", choices=["baseline", "greedy", "power-smc"], required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--subset", type=int, default=100)
    run.add_argument("--alpha", type=float, default=4.0)
    run.add_argument("--particles", type=int, default=16)
    run.add_argument("--kappa", type=float, default=0.5)
    run.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--ramp", type=int, default=0, help="alpha-ramp length (0 disables)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="float16")
    run.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    run.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                     help="shuffle before subsetting (default)")
    run.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                     help="take the first --subset problems in order (matches MH shard 0)")
    run.add_argument("--out", required=True)
    run.set_defaults(func=run_benchmark)

    mh = sub.add_parser("mh", help="run Karan & Du's MH reference and write our-schema CSV")
    mh.add_argument("--model", default="qwen",
                    choices=["qwen", "qwen_math", "phi", "tulu"],
                    help="MH short model name (maps to an HF id; see MH_MODEL_IDS)")
    mh.add_argument("--alpha", type=float, default=4.0,
                    help="target exponent; MH temperature is 1/alpha unless --temperature set")
    mh.add_argument("--temperature", type=float, default=None,
                    help="override the MH temperature directly")
    mh.add_argument("--mcmc-steps", dest="mcmc_steps", type=int, default=10)
    mh.add_argument("--shards", type=int, default=1, help="100-problem shards to run")
    mh.add_argument("--seed", type=int, default=0)
    mh.add_argument("--save-str", dest="save_str", default="results")
    mh.add_argument("--device", default="cuda")
    mh.add_argument("--root", default="third_party/reasoning-with-sampling")
    mh.add_argument("--out", required=True)
    mh.set_defaults(func=run_mh)

    plot = sub.add_parser("plot", help="build the accuracy-vs-latency plot from CSVs")
    plot.add_argument("--inputs", nargs="+", required=True)
    plot.add_argument("--out", default="results/plots/accuracy_latency.png")
    plot.set_defaults(func=make_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
