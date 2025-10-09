"""Assignment 3 template code — with progress logging and stability guards."""

# --- Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import sys
import time
import random

# --- Third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import nevergrad as ng

# --- Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko  # noqa: F401 (kept for quick switch)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller  # noqa: F401 (kept)
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer  # noqa: F401 (kept)
from ariel.utils.runners import simple_runner  # noqa: F401 (kept)
from ariel.utils.tracker import Tracker  # noqa: F401 (kept)
from ariel.utils.video_recorder import VideoRecorder  # noqa: F401 (kept)

# =========================
# Type hints
# =========================
if TYPE_CHECKING:
    from networkx import DiGraph

type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# =========================
# RNG / Globals
# =========================
SEED = 42
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
random.seed(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
SPAWN_POS = [0.0, 0.0, 0.1]
HISTORY: list[list[float]] = []

# =========================
# Logging helper (unbuffered)
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)

# =========================
# Viz helper
# =========================
def show_xpos_history(history: list[list[float]]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.show()

# =========================
# Random baseline controller (kept for experiments)
# =========================
def random_move(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    num_joints = model.nu
    hinge_range = np.pi / 2
    return RNG.uniform(
        low=-hinge_range,
        high=hinge_range,
        size=num_joints,
    ).astype(np.float64)

# =========================
# Tiny neural controller
# =========================
class NeuralController:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray | None = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_params = (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)
        if weights is None:
            self.weights = np.random.randn(self.num_params) * 0.1
        else:
            self.weights = np.array(weights, dtype=float)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        idx = 0
        W1 = self.weights[idx: idx + self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        idx += self.input_size * self.hidden_size
        W2 = self.weights[idx: idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        W3 = self.weights[idx: idx + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)

        layer1 = np.tanh(inputs @ W1)
        layer2 = np.tanh(layer1 @ W2)
        outputs = np.tanh(layer2 @ W3)
        return outputs.reshape(self.output_size)

def controller(model: mj.MjModel, data: mj.MjData, to_track, neural_net: NeuralController) -> None:
    # Inputs: joint pos+vel + 2 time features
    inputs = np.concatenate([
        data.qpos.copy(),
        data.qvel.copy(),
        [np.sin(data.time * 2 * np.pi)],
        [np.cos(data.time * 2 * np.pi)],
    ])
    raw_output = neural_net.forward(inputs)
    target_angles = np.clip(raw_output, -1.0, 1.0) * (np.pi / 2)  # map to [-pi/2, pi/2]

    smoothing = 0.1  # 0: no smoothing, 1: only target angles
    data.ctrl[:] = (1 - smoothing) * data.ctrl[:] + smoothing * target_angles
    # record path
    if to_track:
        HISTORY.append(to_track[0].xpos.copy())

# =========================
# Task objective
# =========================
def fitness(to_track, joint_history: np.ndarray) -> float:
    final_pos = to_track[0].xpos.copy()
    displacement_y = abs(final_pos[1])
    displacement_x = final_pos[0]

    # Encourage some oscillation, punish saturation
    oscillation_reward = float(np.mean(np.std(joint_history, axis=0)))
    saturation_penalty = float(np.mean(np.abs(np.abs(joint_history) - (np.pi / 2))))

    # Tuneable weights
    score = displacement_x + 0.08 * oscillation_reward - 0.08 * saturation_penalty - 0.3 * displacement_y
    return float(score)

# =========================
# Single rollout for given weights & body
# =========================
def evaluate(weights: np.ndarray, robot_graph, steps: int | None = None) -> float:
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mj.MjData(model)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size=input_size, hidden_size=8, output_size=model.nu, weights=weights)

    if steps is None:
        steps = CFG["eval_steps"]

    joint_history = []
    # Stability guard: bail early on NaN/Inf in accelerations
    for _ in range(steps):
        controller(model, data, to_track, neural_net)
        mj.mj_step(model, data)
        if not np.isfinite(data.qacc).all():
            return -1e6  # heavy penalty for unstable sim
        joint_history.append(data.ctrl.copy())

    joint_history = np.array(joint_history)
    return fitness(to_track, joint_history)

# =========================
# Inner optimizer: CMA-ES over controller
# =========================
def experiment(robot_graph: Any, mode: ViewerTypes = "viewer", budget: int | None = None) -> np.ndarray:
    """Optimize controller weights for a fixed body using CMA-ES."""
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)

    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Neural controller sizes
    input_size = len(data.qpos) + len(data.qvel) + 2
    hidden_size = 8
    output_size = model.nu
    dummy_net = NeuralController(input_size, hidden_size, output_size)
    num_params = dummy_net.num_params

    # CMA budget
    if budget is None:
        budget = CFG["inner_budget"]

    # Nevergrad setup
    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(SEED)
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=budget)

    log(f"      [CMA] start — budget={budget}, dim={num_params}")

    # Objective (Nevergrad minimizes)
    def objective(x):
        f = -evaluate(x, robot_graph)
        tells = optimizer.num_tell  # how many points told so far
        # Progress every 10 steps + first
        if tells % 10 == 0 or tells == 1 or tells == budget:
            log(f"      CMA step {tells}/{budget} | fitness = {-f:.3f}")
        return f

    recommendation = optimizer.minimize(objective)
    best = -objective(recommendation.value)
    log(f"      [CMA] best fitness: {best:.6f}")
    return recommendation.value

# =========================
# Outer ES: evolve body; train a brain per body
# =========================
def evolve_body(mu=4, lam=8, gens=10, num_modules=20, genotype_size=64):
    log(f"\n=== BODY EVOLUTION START === gens={gens}, mu={mu}, lam={lam}\n")

    # Parents are (3, G) arrays: [type_p, conn_p, rot_p]
    parents = [np.random.rand(3, genotype_size).astype(np.float32) for _ in range(mu)]
    parent_fitness = [evaluate_body(p, num_modules) for p in parents]
    log(f"[GEN 0/{gens}] seeded {mu} parents — best={max(parent_fitness):.3f}")

    for gen in range(1, gens + 1):
        log(f"\n--- GENERATION {gen}/{gens} ---")
        offspring: list[tuple[np.ndarray, float]] = []

        for i in range(lam):
            log(f"  -> creating offspring {i+1}/{lam}")
            parent = random.choice(parents)
            child = np.clip(
                parent + np.random.normal(0, 0.05, parent.shape),
                0.0, 1.0
            ).astype(np.float32)

            off_fit = evaluate_body(child, num_modules)
            offspring.append((child, off_fit))
            log(f"     offspring {i+1}/{lam} fitness = {off_fit:.3f}")

        # Select top μ among parents + offspring
        pool = list(zip(parents, parent_fitness)) + offspring
        pool.sort(key=lambda it: it[1], reverse=True)
        parents, parent_fitness = zip(*pool[:mu])
        parents, parent_fitness = list(parents), list(parent_fitness)

        log(f"  ✅ best fitness this generation: {parent_fitness[0]:.3f}")

    best_idx = int(np.argmax(parent_fitness))
    log("\n=== BODY EVOLUTION END ===")
    return parents[best_idx]

def evaluate_body(genotype, num_modules: int) -> float:
    """Decode body, then optimize a fresh controller for it K times, average fitness."""
    # Accept either a (3, G) array or a list/tuple of 3 arrays
    if isinstance(genotype, np.ndarray):
        if genotype.ndim != 2 or genotype.shape[0] != 3:
            raise ValueError(f"Genotype must be shape (3, G); got {genotype.shape}")
        type_p, conn_p, rot_p = genotype
    else:
        if not (isinstance(genotype, (list, tuple)) and len(genotype) == 3):
            raise ValueError("Genotype must be a list/tuple of 3 arrays.")
        type_p, conn_p, rot_p = (np.asarray(g, dtype=np.float32) for g in genotype)

    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    p_matrices = nde.forward([type_p, conn_p, rot_p])

    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    K = CFG["repeats"]
    scores = []
    for r in range(K):
        log(f"    [body eval {r+1}/{K}] optimize controller for this body…")
        best_weights = experiment(robot_graph, budget=CFG["inner_budget"])
        fit = evaluate(best_weights, robot_graph, steps=CFG["eval_steps"])
        log(f"    -> fitness {r+1}: {fit:.3f}")
        scores.append(fit)

    avg = float(np.mean(scores)) if scores else -1e9
    log(f"    [body avg fitness] {avg:.3f}")
    return avg

# =========================
# Training config
# =========================
CFG = {
    "inner_budget": 100,   # controller CMA steps
    "eval_steps": 2500,    # physics steps per evaluation
    "gens": 5,             # body ES generations (adjust as needed)
    "mu": 4,               # parents
    "lam": 8,              # offspring per gen
    "repeats": 1,          # evaluate each body K times (increase for robustness)
}

# =========================
# Main
# =========================
def main() -> None:
    """Entry point."""
    num_modules = 20
    genotype_size = 64

    # --- evolve the body ---
    best_genotype = evolve_body(
        mu=CFG["mu"], lam=CFG["lam"], gens=CFG["gens"],
        num_modules=num_modules, genotype_size=genotype_size
    )

    # --- decode the best body ---
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    type_p, conn_p, rot_p = best_genotype
    p_matrices = nde.forward([type_p, conn_p, rot_p])
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph = hpd.probability_matrices_to_graph(*p_matrices)
    save_graph_as_json(robot_graph, DATA / "robot_graph.json")

    core = construct_mjspec_from_graph(robot_graph)

    # Clear old history
    HISTORY.clear()

    # Optimize weights for the winning body
    log("\n[final CMA] optimizing controller for winning body…")
    best_weights = experiment(robot_graph=robot_graph, budget=CFG["inner_budget"])

    # Create world and spawn robot for simulation
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)

    # Bind robot geoms to track
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Neural controller for final viewing
    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, best_weights)
    HISTORY.clear()
    mj.set_mjcb_control(lambda m, d: controller(m, d, to_track, neural_net))

    log("\n🎯 Finished full evolution! Launching viewer…")
    viewer.launch(model=model, data=data)

    # Show path
    show_xpos_history(HISTORY)

if __name__ == "__main__":
    # Run unbuffered (prints appear immediately). In CLI you can also use: python -u script.py
    main()
