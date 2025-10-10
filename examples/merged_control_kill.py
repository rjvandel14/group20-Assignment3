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
SEED = 111
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
SPAWN_POS = [
    [-0.8, 0.0, 0.1],
    [1.0, 0.0, 0.2],
    [3.0, 0.0, 0.2],
]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# non-learner test values
NONLEANER_SECONDS = 4.0   # active phase to see if we should kill
MAX_BODY_RETRIES = 15    # how many bodies we retry before giving up
SETTLE_SECONDS = 1.5   # let it fall/settle for a second
KICK_SCALE = 0.7    # strength of joint kicks during test

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
def symmetry_score(graph) -> float:
    """Return [0,1]; 1 = perfectly mirrored around x=0."""
    try:
        core = construct_mjspec_from_graph(graph)
    except Exception:
        return 0.0

    coords = []
    for body in core.spec.bodies:
        if hasattr(body, "pos") and body.pos is not None:
            coords.append(np.asarray(body.pos, float))
        for geom in getattr(body, "geoms", []):
            if hasattr(geom, "pos") and geom.pos is not None:
                coords.append(np.asarray(geom.pos, float))
    if not coords:
        return 0.0

    coords = np.vstack(coords)
    mirrored = coords.copy()
    mirrored[:, 0] *= -1

    dists = []
    for m in mirrored:
        d = np.sqrt(((coords - m) ** 2).sum(axis=1)).min()
        dists.append(d)
    mean_min = float(np.mean(dists))

    scale = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    if scale <= 1e-9:
        return 0.0

    return float(np.clip(1.0 - mean_min / scale, 0.0, 1.0))

def stability_penalty(graph, z_min=0.15, weight=3.0) -> float:
    """Penalty grows if average body height < z_min."""
    core = construct_mjspec_from_graph(graph)
    z_positions = [np.asarray(b.pos, float)[2]
                   for b in core.spec.bodies if hasattr(b, "pos")]
    if not z_positions:
        return 0.0
    return max(0.0, z_min - float(np.mean(z_positions))) * weight

def single_connection_hinge_penalty(g, w_single: float = 0.10) -> float:
    deg = g.degree
    bad = sum(1 for n, d in g.nodes(data=True)
              if d.get("type") == "HINGE" and deg(n) == 1)
    return w_single * bad

def arch_penalty(graph, base=0.30) -> float:
    num_blocks = count_from_graph(graph, "BRICK")
    num_hinges = count_from_graph(graph, "HINGE")
    if num_hinges == 0:
        return 1e6
    ratio_over = max(0.0, num_blocks/num_hinges - 1.0)
    return base * ratio_over

def cma_train_controller(graph):
    print("[CMA] starting...")

    sym_bonus = 0.5 * symmetry_score(graph)            # subtract later (good thing)
    pen_arch  = arch_penalty(graph, base=0.30)         # add
    pen_hinge = single_connection_hinge_penalty(graph, w_single=0.10)  # add
    pen_stab  = stability_penalty(graph, z_min=0.15, weight=0.6)       # add (tuned lower)

    penalty = (pen_arch + pen_hinge + pen_stab) - sym_bonus

    w, f = experiment(graph,penalty)
    #f = evaluate(w, graph)
    print(f"[CMA] best fitness={f:.4f}")
    return w, float(f)

def fitness_function(history: list[float], graph : Graph, penalty) -> float:
    if not history:
        return 1e6  
    
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )

    return cartesian_distance+penalty


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
    ym0, ymc = 0, SPAWN_POS[0][0]

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

def evaluate(weights, robot_graph, spawn_pos, penalty):
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=spawn_pos)

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, weights)

    controller = StepwiseController(neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.8)

    steps = 800 #2500
    #joint_history = []

    # --- EARLY BAIL SETTINGS ---
    dt = model.opt.timestep
    BAIL_SECONDS = 1.5
    BAIL_STEPS = max(1, int(BAIL_SECONDS / dt))
    CHECK_EVERY = max(1, int(0.25 / dt))
    MIN_DX_BAIL = 0.005
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

    # compute fitness using tracker history and graph-based penalty
    f = fitness_function(tracker.history["xpos"][0], robot_graph, penalty)
    # add provided penalty (if any) — keep backwards-compatible
    return f

def experiment(robot_graph: Any, penalty) -> np.ndarray:
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    # robot = gecko()
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS[0])

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos) + len(data.qvel) + 2
    dummy_net = NeuralController(input_size, 8, model.nu)
    num_params = dummy_net.num_params

    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(SEED)
    optimizer = ng.optimizers.CMA(parametrization=num_params, budget=15)# 200)
    
    spawn_positions = SPAWN_POS

    best_score_so_far = float("inf")
    weights_path = DATA / "best_weights.csv"

    def objective(x):
        nonlocal best_score_so_far
        weights = np.asarray(x)
        scores = []
        for sp in spawn_positions:
            try:
                f = evaluate(weights, robot_graph, sp, penalty)
            except Exception as e:
                print("Evaluation error at spawn", sp, ":", e)
                f = 1e6
            scores.append(f)
        score = float(np.mean(scores))

        # If this candidate is better, save weights and print
        if score < best_score_so_far:
            best_score_so_far = score
            np.savetxt(weights_path, weights, delimiter=",")
            print(f"New best score: {score:.4f} – weights saved to {weights_path}")

        return score

    recommendation = optimizer.minimize(objective)
    print("Best fitness:", objective(recommendation.value))
    return recommendation.value, objective(recommendation.value)

# non-learner test: test if robot can move and otherwise kill-off robot
def random_move(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    num_joints = model.nu
    hinge_range = np.pi / 2 # hinges take values between -pi/2 and pi/2
    return RNG.uniform(
        low=-hinge_range, # -pi/2
        high=hinge_range, # pi/2
        size=num_joints
    ).astype(np.float64)

def is_learning(robot_graph) -> tuple[bool, float, float]:
    
    mj.set_mjcb_control(None)
    world = OlympicArena()

    # robust build
    try:
        robot = construct_mjspec_from_graph(robot_graph)
    except (ValueError, KeyError) as e:
        print(f"[non-learner] invalid graph during construct: {type(e).__name__}: {e}")
        return (False, 0.0, 0.0)

    world.spawn(robot.spec, spawn_position=SPAWN_POS[0])
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # find core and track movement in xy-plane
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    core_bind = next((data.bind(g) for g in geoms if "core" in g.name), None)
    if core_bind is None:
        return (False, 0.0, 0.0)

    # thresholds 
    dt = model.opt.timestep
    MIN_DX = 0.02    # approx 2–3 cm, robot must move at least a few centimeters
    MIN_V_RMS = 0.01   # approx 1 cm/s over last 1 s, robot must show some speed

    # timing
    dt = model.opt.timestep # get simulation time step
    settle_steps = max(1, int(SETTLE_SECONDS / dt)) # how many steps in the settling phase
    active_steps = max(1, int(NONLEANER_SECONDS / dt)) # how many steps in the active test phase

    # phase 1: settle (no control), let it fall and stabilize
    data.ctrl[:] = 0.0
    for _ in range(settle_steps):
        mj.mj_step(model, data)

    # initial xy position
    start_xy = np.array(core_bind.xpos[:2], dtype=float)
    prev_xy = start_xy.copy()
    v_hist: list[float] = []
    floor_contact = False

    # phase 2: active phase - movement test
    # apply random movements in joints to see if robot is capable of moving
    for _ in range(active_steps):
        data.ctrl[:] = KICK_SCALE * random_move(model, data) # random control signal to every joint
        mj.mj_step(model, data) # react to applied torques
        cur_xy = np.array(core_bind.xpos[:2], dtype=float) # xy-position in this step
        speed = np.linalg.norm(cur_xy - prev_xy) / dt # how fast robot moved in this step
        v_hist.append(speed)
        floor_contact |= (data.ncon > 0) # at least one contact with floor
        prev_xy = cur_xy

    dx = float(np.linalg.norm(prev_xy - start_xy))
    last_1s = max(1, int(1.0 / dt))
    speed_end = float(np.sqrt(np.mean(np.square(v_hist[-last_1s:]))) if v_hist else 0.0)

    # passed when robot had contact with floor and min distance and speed is met
    passed = floor_contact and ((dx >= MIN_DX) or (speed_end >= MIN_V_RMS))

    status = "robot passed" if passed else "killed"
    print(f"[non-learner filter] {status} | dx={dx:.4f} m, speed_end={speed_end:.4f} m/s "
          f"(need contact & (distance≥{MIN_DX:.3f} AND speed_end≥{MIN_V_RMS:.3f}))")
    return (passed, dx, speed_end)

# generating new bodies until one passes the non-learner test
def sample_robot_nonlearner(num_modules: int, genotype_size: int) -> tuple["DiGraph", np.ndarray]:
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

        try:
            _ = construct_mjspec_from_graph(robot_graph)  # check buildability
        except (ValueError, KeyError) as e:
            print(f"[attempt {attempt:02d}] invalid graph: {type(e).__name__}: {e} -> resample")
            continue

        passed, dx, speed_end = is_learning(robot_graph)
        print(f"[attempt {attempt:02d}] {'OK' if passed else 'retry'} (dx={dx:.3f}, speed_end={speed_end:.3f})")

        if passed:
            return robot_graph,  np.concatenate([type_p_genes, conn_p_genes, rot_p_genes])

    raise RuntimeError(f"Failed to sample a learner body in {max_retries} attempts")
# end non-learner test

def main() -> None:
    genotype_size = 64

    smoke_graph, smoke_vec = sample_robot_nonlearner(NUM_OF_MODULES, 64)

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
        cfg=ESConfig(gens=12, mu=12, lam=48, sigma_init=0.20, prescreen_retries=3, seed=SEED),
        initial_parents=[smoke_vec],
    )
    print(f"[FINAL] best fitness: {best_fit:.4f}")

    print("\nNode attributes of best_graph:")
    for node, attrs in best_graph.nodes(data=True):
        print(node, attrs)
    # 3) Save + visualize winner (unchanged)
    save_graph_as_json(best_graph, DATA / "robot_graph.json")
    core = construct_mjspec_from_graph(best_graph)
    # core = gecko()

    mj.set_mjcb_control(None)

    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS[0])
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
