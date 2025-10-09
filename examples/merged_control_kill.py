"""Assignment 3 template code with Stepwise Controller + non-learner prescreen."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import random
import nevergrad as ng
from networkx import Graph

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.simulation.environments import OlympicArena
from ariel.utils.tracker import Tracker

from body_ES import ESConfig, Callbacks, evolve_mu_plus_lambda

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
SPAWN_POS = [-0.8, 0.0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# non-learner test values
NONLEARNER_SECONDS = 4.0   # active phase to see if we should kill
MAX_BODY_RETRIES = 10      # how many bodies we retry before giving up
SETTLE_SECONDS = 1.5       # let it fall/settle for a second
KICK_SCALE = 0.7           # strength of joint kicks during test

def make_decode_from_vec(num_modules: int, genotype_size: int):
    def decode_from_vec(vec: np.ndarray):
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        hpd = HighProbabilityDecoder(num_modules)
        a = np.clip(vec[:genotype_size], 0, 1).astype(np.float32)
        b = np.clip(vec[genotype_size:2*genotype_size], 0, 1).astype(np.float32)
        c = np.clip(vec[2*genotype_size:], 0, 1).astype(np.float32)
        p_mats = nde.forward([a, b, c])
        return hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    return decode_from_vec

def cma_train_controller(graph):
    print("[CMA] starting...")
    w = experiment(graph)
    f = evaluate(w, graph)
    print(f"[CMA] best fitness={f:.4f}")
    return w, float(f)

def count_from_graph(graph: Graph, name) -> int:
    count = 0
    if name == "BRICK":
        for node in graph.nodes:
            module_type = graph.nodes[node]["type"]
            if module_type in ModuleType.BRICK.name:
                count += 1
    elif name == "HINGE":
        for node in graph.nodes:
            module_type = graph.nodes[node]["type"]
            if module_type in ModuleType.HINGE.name:
                count += 1
    return count

def fitness_function(history: list[float], graph: Graph) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )

    num_blocks = count_from_graph(graph, "BRICK")
    num_hinges = count_from_graph(graph, "HINGE")

    arch_penalty = 0
    ratio = num_blocks/(num_hinges)
    if ratio > 1:
        arch_penalty = 0.3

    return cartesian_distance + arch_penalty * ratio

# def fitness(history: list[float], joint_history):
#     final_pos = history[-1]
#     displacement_y = abs(final_pos[1])
#     displacement_x = final_pos[0]

#     oscillation_reward = np.mean(np.std(joint_history, axis=0))
#     saturation_penalty = np.mean(np.abs(np.abs(joint_history) - (np.pi / 2)))

#     fitness = displacement_x + 0.08 * oscillation_reward - 0.08 * saturation_penalty - 0.3 * displacement_y
#     return fitness

def show_xpos_history(history: list[float]) -> None:
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

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()

class NeuralController:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_params = (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)
        self.weights = np.array(weights) if weights is not None else np.random.randn(self.num_params) * 0.1

    def forward(self, inputs):
        idx = 0
        W1 = self.weights[idx: idx + self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        idx += self.input_size * self.hidden_size
        W2 = self.weights[idx: idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        W3 = self.weights[idx: idx + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)

        layer1 = np.tanh(np.dot(inputs, W1))
        layer2 = np.tanh(np.dot(layer1, W2))
        outputs = np.tanh(np.dot(layer2, W3))

        return outputs.reshape(self.output_size)

# Stepwise Controller
class StepwiseController:
    def __init__(self, neural_net, tracker, ctrl_every=50, save_every=100, alpha=0.1):
        self.neural_net = neural_net
        self.tracker = tracker
        self.ctrl_every = ctrl_every
        self.save_every = save_every
        self.alpha = alpha
        self.step_count = 0

    def step(self, model, data):
        self.step_count += 1

        if self.step_count % self.save_every == 0:
            self.tracker.update(data)

        if self.step_count % self.ctrl_every == 0:
            inputs = np.concatenate([
                data.qpos.copy(),
                data.qvel.copy(),
                [np.sin(data.time * 2 * np.pi)],
                [np.cos(data.time * 2 * np.pi)]
            ])
            output = self.neural_net.forward(inputs)
            target_angles = output * (np.pi / 2)
            data.ctrl[:] = (1 - self.alpha) * data.ctrl[:] + self.alpha * target_angles

def evaluate(weights, robot_graph):
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    # robot = gecko()
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, weights)

    controller = StepwiseController(neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.8)

    steps = 2500
    #joint_history = []

    # --- EARLY BAIL SETTINGS ---
    dt = model.opt.timestep
    BAIL_SECONDS = 1.5                      # evaluate “promise” in first ~1.5s
    BAIL_STEPS = max(1, int(BAIL_SECONDS / dt))
    CHECK_EVERY = max(1, int(0.25 / dt))    # check each 0.25s
    MIN_DX_BAIL = 0.005                     # < 5 mm displacement → bail
    # ----------------------------

    # for bail metrics
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    core_bind = next((data.bind(g) for g in geoms if "core" in g.name), None)
    start_xy = None
    if core_bind is not None:
        mj.mj_forward(model, data)
        start_xy = np.array(core_bind.xpos[:2], dtype=float)

    for k in range(steps):
        controller.step(model, data)
        mj.mj_step(model, data)

        # ---- Early-bail check (only in the first BAIL_SECONDS) ----
        if start_xy is not None and k <= BAIL_STEPS and (k % CHECK_EVERY == 0):
            cur_xy = np.array(core_bind.xpos[:2], dtype=float)
            dx = float(np.linalg.norm(cur_xy - start_xy))
            if dx < MIN_DX_BAIL and k >= BAIL_STEPS:
                # hopeless controller/body combo → kill fast
                print(f"[EVAL] early bail at t≈{k*dt:.2f}s (dx={dx:.4f} m < {MIN_DX_BAIL} m)")
                return 1e9
        # -----------------------------------------------------------

    return fitness_function(tracker.history["xpos"][0], robot_graph)

def experiment(robot_graph: Any) -> np.ndarray:
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    # robot = gecko()
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    input_size = len(data.qpos) + len(data.qvel) + 2
    dummy_net = NeuralController(input_size, 8, model.nu)
    num_params = dummy_net.num_params

    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(SEED)
    optimizer = ng.optimizers.CMA(parametrization=num_params, budget=200)

    def objective(x):
        # minimize distance+penalty (your fitness_function returns lower=better)
        return evaluate(x, robot_graph)

    recommendation = optimizer.minimize(objective)
    print("Best fitness:", objective(recommendation.value))
    return recommendation.value

# # non-learner test: test if robot can move and otherwise kill-off robot
# def random_move(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
#     num_joints = model.nu
#     hinge_range = np.pi / 2 # hinges take values between -pi/2 and pi/2
#     return RNG.uniform(
#         low=-hinge_range, # -pi/2
#         high=hinge_range, # pi/2
#         size=num_joints
#     ).astype(np.float64)

def is_learning(robot_graph) -> tuple[bool, float, float]:
    """Prescreen: settle, then apply a small deterministic joint pattern to test basic locomotion."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    core_bind = next((data.bind(g) for g in geoms if "core" in g.name), None)
    if core_bind is None:
        return (False, 0.0, 0.0)

    dt = model.opt.timestep
    MIN_DX = 0.02 + 0.01 * (dt / 0.002)
    MIN_V_RMS = 0.01 * (0.002 / dt)

    settle_steps = max(1, int(SETTLE_SECONDS / dt))
    active_steps = max(1, int(NONLEARNER_SECONDS / dt))

    # Phase 1: settle under gravity
    data.ctrl[:] = 0.0
    for _ in range(settle_steps):
        mj.mj_step(model, data)

    # Phase 2: small, deterministic sine pattern across joints
    nu = model.nu
    omega = 2.0 * np.pi * 0.8     # ~0.8 Hz
    phase = np.linspace(0.0, np.pi, num=nu, dtype=float)

    start_xy = np.array(core_bind.xpos[:2], dtype=float)
    prev_xy = start_xy.copy()
    v_hist: list[float] = []
    floor_contact = False

    for k in range(active_steps):
        t = k * dt
        target = KICK_SCALE * (np.pi / 4.0) * np.sin(omega * t + phase)
        data.ctrl[:] = 0.7 * data.ctrl[:] + 0.3 * target
        mj.mj_step(model, data)
        cur_xy = np.array(core_bind.xpos[:2], dtype=float)
        speed = float(np.linalg.norm(cur_xy - prev_xy) / dt)
        v_hist.append(speed)
        floor_contact |= (data.ncon > 0)
        prev_xy = cur_xy

    dx = float(np.linalg.norm(prev_xy - start_xy))
    last_1s = max(1, int(1.0 / dt))
    v_rms = float(np.sqrt(np.mean(np.square(v_hist[-last_1s:]))) if v_hist else 0.0)

    passed = floor_contact and ((dx >= MIN_DX) or (v_rms >= MIN_V_RMS))
    print(f"[non-learner] {'KEPT' if passed else 'KILLED'} | dx={dx:.4f} m, v_rms={v_rms:.4f} m/s "
          f"(need contact & (dx≥{MIN_DX:.3f} or v_rms≥{MIN_V_RMS:.3f}))")
    return (passed, dx, v_rms)

# generating new bodies until one passes the non-learner test
def sample_robot_nonlearner(num_modules: int, genotype_size: int) -> "DiGraph":
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    hpd = HighProbabilityDecoder(num_modules)

    for attempt in range(1, MAX_BODY_RETRIES + 1):
        # random genotype
        type_p_genes = RNG.random(genotype_size).astype(np.float32)
        conn_p_genes = RNG.random(genotype_size).astype(np.float32)
        rot_p_genes  = RNG.random(genotype_size).astype(np.float32)
        genotype = [type_p_genes, conn_p_genes, rot_p_genes]

        # decode robot
        p_matrices = nde.forward(genotype)
        robot_graph = hpd.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        )

        passed, dx, v_rms = is_learning(robot_graph)
        print(f"[attempt {attempt:02d}] {'OK' if passed else 'retry'} (dx={dx:.3f}, v_rms={v_rms:.3f})")

        if passed:
            return robot_graph

    raise RuntimeError(f"Failed to sample a learner body in {MAX_BODY_RETRIES} attempts")
# end non-learner test

def main() -> None:
    genotype_size = 64

     # 1) Build ES callbacks from your local logic
    callbacks = Callbacks(
        decode_from_vec=make_decode_from_vec(NUM_OF_MODULES, genotype_size),
        prescreen=is_learning,
        train_controller=cma_train_controller,
    )

    # 2) Run ES body search (μ+λ); prescreen is called inside for every candidate
    best_graph, best_weights, best_fit = evolve_mu_plus_lambda(
        genotype_size=genotype_size,
        callbacks=callbacks,
        cfg=ESConfig(gens=12, mu=12, lam=60, sigma_init=0.15, prescreen_retries=3, seed=SEED),
    )
    print(f"[FINAL] best fitness: {best_fit:.4f}")

    # 3) Save + visualize winner (unchanged)
    save_graph_as_json(best_graph, DATA / "robot_graph.json")
    core = construct_mjspec_from_graph(best_graph)
    # core = gecko()

    mj.set_mjcb_control(None)

    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, best_weights)
    tracker.update(data)
    stepwise_ctrl = StepwiseController(neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.1)
    mj.set_mjcb_control(lambda m, d: stepwise_ctrl.step(m, d))

    viewer.launch(model=model, data=data)

    print(tracker.history["xpos"][0])
    show_xpos_history(tracker.history["xpos"][0])

if __name__ == "__main__":
    main()
