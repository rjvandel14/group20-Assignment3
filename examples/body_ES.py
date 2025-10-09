# body_ES.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
import numpy as np
from pathlib import Path
import csv
import time
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json

DiGraph = "DiGraph"

@dataclass
class ESConfig:
    gens: int = 12
    mu: int = 12
    lam: int = 60
    sigma_init: float = 0.15
    prescreen_retries: int = 3
    seed: int = 42
    verbose: bool = True                  # << add this
    print_every: int = 1                  # << print each gen (or set to 5, etc.)
    save_dir: str = "./__data__/es_anytime"

@dataclass
class Callbacks:
    decode_from_vec: Callable[[np.ndarray], DiGraph]
    prescreen: Callable[[DiGraph], Tuple[bool, float, float]]
    train_controller: Callable[[DiGraph], Tuple[np.ndarray, float]]

def evolve_mu_plus_lambda(
    genotype_size: int,
    callbacks: Callbacks,
    cfg: ESConfig = ESConfig(),
    initial_parents=None
) -> Tuple[DiGraph, np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)

    # Prepare save directory for anytime snapshots
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_history_csv = save_dir / "best_history.csv"
    # create CSV header if not exists
    if not best_history_csv.exists():
        with open(best_history_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "gen", "fitness", "note", "graph_path", "weights_path"])


    dim = 3 * genotype_size
    tau_prime = 1 / np.sqrt(2 * dim)
    tau = 1 / np.sqrt(2 * np.sqrt(dim))

    def make_feasible(x): return np.clip(x, 0, 1)

    def try_decode_and_prescreen(x):
        for trial in range(1, cfg.prescreen_retries + 1):
            g = callbacks.decode_from_vec(x)
            ok, dx, v = callbacks.prescreen(g)
            if ok:
                if cfg.verbose:
                    print(f"[ES] prescreen PASS (trial {trial}/{cfg.prescreen_retries}) | dx={dx:.3f}, v={v:.3f}")
                return g
            if cfg.verbose:
                print(f"[ES] prescreen FAIL (trial {trial}/{cfg.prescreen_retries}) | dx={dx:.3f}, v={v:.3f}")
            x = make_feasible(x + 0.05 * rng.standard_normal(dim))
        return None

    def evaluate_body(x):
        g = try_decode_and_prescreen(x)
        if g is None:
            if cfg.verbose:
                print("[ES] candidate KILLED (prescreen)")
            return 1e9, None, None
        t0 = time.time()
        w, f = callbacks.train_controller(g)
        if cfg.verbose:
            print(f"[ES] CMA done in {time.time()-t0:.2f}s | fitness={f:.4f}")
        return f, g, w

    if cfg.verbose:
        print(f"[ES] init μ={cfg.mu}, λ={cfg.lam}, gens={cfg.gens}, dim={dim}")

    parents = []

    def save_best(best):
        _, _, best_f, best_g, best_w = best
        print("save best")
        print(best_f)
        print(best_g)

        if best_g is not None and best_w is not None:
            ts = int(time.time())
            graph_fn = save_dir / f"best_graph_gen{gen}_fit{best_f:.6f}_{ts}.json"
            weights_fn = save_dir / f"best_weights_gen{gen}_fit{best_f:.6f}_{ts}.csv"

            # save graph (uses provided save_graph_as_json)
            try:
                save_graph_as_json(best_g, graph_fn)
            except Exception as e:
                if cfg.verbose:
                    print(f"[ES SAVE] failed saving graph: {e}")

            # save weights (numpy array to csv)
            try:
                np.savetxt(weights_fn, best_w, delimiter=",")
            except Exception as e:
                if cfg.verbose:
                    print(f"[ES SAVE] failed saving weights: {e}")

            # append to best_history.csv
            with open(best_history_csv, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                                    gen, f"{best_f:.6f}", "new_best", str(graph_fn), str(weights_fn)])

            if cfg.verbose:
                print(f"[ES SAVE] New best saved: gen={gen} fitness={best_f:.6f} -> {graph_fn}, {weights_fn}")


    def eval_and_push(x, note=""):
        sigma = np.full(dim, cfg.sigma_init)
        f, g, w = evaluate_body(x)
        parents.append([x, sigma, f, g, w])
        if cfg.verbose:
            tag = f" ({note})" if note else ""
            print(f"[ES] init parent {len(parents)}/{cfg.mu}{tag} | f={f:.4f}")

    # (A) use provided seeds first (if any)
    if initial_parents:
        for vec in initial_parents[:cfg.mu]:
            vec = np.asarray(vec, dtype=float)
            vec = np.clip(vec, 0, 1)
            eval_and_push(vec, note="seed")

    # (B) fill the rest with randoms
    while len(parents) < cfg.mu:
        x = rng.random(dim)
        eval_and_push(x, note="rand")

    best = min(parents, key=lambda ind: ind[2])

    # Main ES loop
    for gen in range(1, cfg.gens + 1):
        offspring = []
        for _ in range(cfg.lam):
            p = parents[rng.integers(cfg.mu)]
            z0, z = rng.standard_normal(), rng.standard_normal(dim)
            sigma_c = p[1] * np.exp(tau_prime * z0 + tau * z)
            x_c = make_feasible(p[0] + sigma_c * rng.standard_normal(dim))
            f, g, w = evaluate_body(x_c)
            offspring.append([x_c, sigma_c, f, g, w])

        pool = parents + offspring
        pool.sort(key=lambda ind: ind[2])
        parents = pool[:cfg.mu]

        if parents[0][2] < best[2]:
            best = parents[0]
            print("save best")
            save_best(best)
        if cfg.verbose and (gen % cfg.print_every == 0):
            print(f"[ES μ+λ] gen {gen:02d} | best_fit={best[2]:.4f}")

    _, _, best_f, best_g, best_w = best
    if best_g is None or best_w is None:
        raise RuntimeError("No valid body survived.")
    if cfg.verbose:
        print(f"[ES] DONE | best_fit={best_f:.4f}")
    return best_g, best_w, float(best_f)
