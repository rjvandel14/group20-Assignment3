# body_opt.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Any, List, Tuple, Dict
import time, json
import numpy as np
import mujoco as mj
import nevergrad as ng
import time

from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena

SEED = 42
RNG = np.random.default_rng(SEED)
BODY_DIM = 64
NUM_MODULES = 20

DATA = Path("__data__/body_de").resolve()
DATA.mkdir(parents=True, exist_ok=True)

# ---------- genome helpers ----------
def init_body_genome(dim: int = BODY_DIM) -> List[np.ndarray]:
    return [RNG.random(dim, np.float32), RNG.random(dim, np.float32), RNG.random(dim, np.float32)]

def pack_body(geno: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(geno).astype(np.float32)

def unpack_body(vec: np.ndarray, dim: int = BODY_DIM) -> List[np.ndarray]:
    v = np.asarray(vec, np.float32).ravel()
    assert v.size == 3*dim, f"expected {3*dim}, got {v.size}"
    return [v[0:dim], v[dim:2*dim], v[2*dim:3*dim]]

def clip_body(geno: List[np.ndarray]) -> List[np.ndarray]:
    return [np.clip(g, 0.0, 1.0).astype(np.float32) for g in geno]

# ---------- build & screen ----------
def build_robot(genotype: List[np.ndarray], num_modules: int = NUM_MODULES) -> Tuple[Any, Any]:
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    t, c, r = nde.forward(genotype)
    graph = HighProbabilityDecoder(num_modules).probability_matrices_to_graph(t, c, r)
    module = construct_mjspec_from_graph(graph)  # CoreModule
    return graph, module

def is_non_learner(robot_module: Any, probe_seconds: float = 2.0, min_dx: float = 0.05) -> bool:
    world = OlympicArena()
    world.spawn(robot_module.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)
    steps = max(1, int(probe_seconds / model.opt.timestep))
    u = RNG.uniform(-np.pi/2, np.pi/2, size=model.nu).astype(np.float64)
    x0 = float(data.qpos[0])
    for _ in range(steps):
        data.ctrl[:model.nu] = u
        mj.mj_step(model, data)

    return abs(float(data.qpos[0]) - x0) < min_dx

# ---------- DE optimize ----------
def optimize_body_de(controller_eval_fn, pop_budget=60, seed=42, screen_nonlearners=True):
    start_wall = time.time()

    print("\n=== [DE] STARTING BODY OPTIMIZATION ===")
    print(f"=== [DE] BUDGET: {pop_budget} EVALUATIONS | SEED: {seed} | SCREEN_NONLEARNERS: {screen_nonlearners} ===", flush=True)

    start_vec = pack_body(init_body_genome())

    param = ng.p.Array(init=start_vec)
    param.set_bounds(0.0, 1.0)
    param.random_state.seed(seed)

    opt = ng.optimizers.DE(parametrization=param, budget=pop_budget)

    # progress counters
    eval_count = 0
    best_so_far = -1e18

    def objective(x):
        nonlocal eval_count, best_so_far
        eval_count += 1

        vec = x.value if hasattr(x, "value") else np.asarray(x, np.float32)
        geno = clip_body(unpack_body(vec))

        # BUILD + SCREEN
        graph, module = build_robot(geno)
        if screen_nonlearners and is_non_learner(module):
            print(f"[DE][EVAL {eval_count}/{pop_budget}] NON-LEARNER → SKIPPING WITH LARGE LOSS", flush=True)
            return 1e6  # minimize

        # CONTROLLER SCORING
        print(f"[DE][EVAL {eval_count}/{pop_budget}] START SCORING BODY…", flush=True)
        fit = float(controller_eval_fn(graph))
        print(f"[DE][EVAL {eval_count}/{pop_budget}] BODY FITNESS = {fit:.4f}", flush=True)

        if fit > best_so_far:
            best_so_far = fit
            print(f"*** [DE] NEW BEST BODY FITNESS = {best_so_far:.4f} ***", flush=True)
        return -fit  # nevergrad minimizes

    rec = opt.minimize(objective)

    # reconstruct best
    best_vec = np.clip(rec.value, 0.0, 1.0).astype(np.float32)
    best_genome = unpack_body(best_vec)
    best_graph, _ = build_robot(best_genome)

    run_dir = DATA / time.strftime("de_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_path = run_dir / "best_body_graph.json"
    save_graph_as_json(best_graph, graph_path)

    elapsed = time.time() - start_wall
    print(f"=== [DE] DONE | EVALS: {eval_count} | BEST_FIT≈ {-rec.loss:.4f} | SAVED: {graph_path} | WALL-TIME: {elapsed:.1f}s ===\n", flush=True)

    out = {
        "best_genome": [g.tolist() for g in best_genome],
        "best_fitness_est": float(-rec.loss),
        "graph_path": str(graph_path),
        "run_dir": str(run_dir),
        "evals": int(eval_count),
        "wall_time_sec": float(elapsed),
    }
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    return out