# body_ES.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
import numpy as np
import time

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

@dataclass
class Callbacks:
    decode_from_vec: Callable[[np.ndarray], DiGraph]
    prescreen: Callable[[DiGraph], Tuple[bool, float, float]]
    train_controller: Callable[[DiGraph], Tuple[np.ndarray, float]]

def evolve_mu_plus_lambda(
    genotype_size: int,
    callbacks: Callbacks,
    cfg: ESConfig = ESConfig(),
) -> Tuple[DiGraph, np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)
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

    # Initialize μ parents
    if cfg.verbose:
        print(f"[ES] init μ={cfg.mu}, λ={cfg.lam}, gens={cfg.gens}, dim={dim}")
    parents = []
    for i in range(cfg.mu):
        x = rng.random(dim)
        sigma = np.full(dim, cfg.sigma_init)
        f, g, w = evaluate_body(x)
        parents.append([x, sigma, f, g, w])
        if cfg.verbose:
            print(f"[ES] init parent {i+1}/{cfg.mu} | f={f:.4f}")
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
        if cfg.verbose and (gen % cfg.print_every == 0):
            print(f"[ES μ+λ] gen {gen:02d} | best_fit={best[2]:.4f}")

    _, _, best_f, best_g, best_w = best
    if best_g is None or best_w is None:
        raise RuntimeError("No valid body survived.")
    if cfg.verbose:
        print(f"[ES] DONE | best_fit={best_f:.4f}")
    return best_g, best_w, float(best_f)
