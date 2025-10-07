"""Assignment 3 template code."""

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

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

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
SPAWN_POS = [0.0, 0.0, 0.1]
HISTORY = []


# --- Non-learner prescreen settings --- #
NONLEARNER_SECONDS = 2.0     # short, cheap sim
MIN_DX = 0.10                # minimal planar displacement (m)
MIN_V_RMS = 0.04             # minimal RMS planar speed (m/s) over last ~1s
MAX_BODY_RETRIES = 10        # don't get stuck resampling forever
SETTLE_SECONDS = 0.5      # let it fall/settle; ignore this phase
KICK_SCALE = 1.0           # 1.0 = full-range random torques during active phase

# put near the top with the other constants
SMOOTHING    = 0.85   # raise from 0.1 so actions actually change
KICK_SECONDS = 1.2   # brief boost to break static friction
KICK_GAIN    = 2.0  # 25% stronger control for the first 0.4s
EXPL_NOISE   = 0.015  # small exploration noise on controls (optional)




def is_learner(robot_graph) -> tuple[bool, float, float]:
    """Prescreen with settle→active phases. Return (passed, dx, v_rms)."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)  # ensure derived fields sync'd

    # Bind the 'core' geom for planar motion tracking
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    core_bind = next((data.bind(g) for g in geoms if "core" in g.name), None)
    if core_bind is None:
        return (False, 0.0, 0.0)

    dt = model.opt.timestep
    settle_steps = max(1, int(SETTLE_SECONDS / dt))
    active_steps = max(1, int(NONLEARNER_SECONDS / dt))

    # ---- Phase 1: settle (no control) ----
    data.ctrl[:] = 0.0
    for _ in range(settle_steps):
        mj.mj_step(model, data)

    # Take the post-settle position as the baseline
    start_xy = np.array(core_bind.xpos[:2], dtype=float)
    prev_xy = start_xy.copy()
    v_hist: list[float] = []

    # ---- Phase 2: active (random torques) ----
    for _ in range(active_steps):
        data.ctrl[:] = KICK_SCALE * random_move(model, data)
        mj.mj_step(model, data)
        cur_xy = np.array(core_bind.xpos[:2], dtype=float)
        speed = np.linalg.norm(cur_xy - prev_xy) / dt
        v_hist.append(speed)
        prev_xy = cur_xy

    dx = float(np.linalg.norm(prev_xy - start_xy))
    last_1s = max(1, int(1.0 / dt))
    v_rms = float(np.sqrt(np.mean(np.square(v_hist[-last_1s:])))) if v_hist else 0.0
    passed = (dx >= MIN_DX) or (v_rms >= MIN_V_RMS)
    return (passed, dx, v_rms)

def sample_learner_robot_graph(num_modules: int, genotype_size: int) -> "DiGraph":
    """Resample NDE genotypes until the body passes the prescreen (or give up)."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    hpd = HighProbabilityDecoder(num_modules)

    for attempt in range(1, MAX_BODY_RETRIES + 1):
        type_p = RNG.random(genotype_size).astype(np.float32)
        conn_p = RNG.random(genotype_size).astype(np.float32)
        rot_p  = RNG.random(genotype_size).astype(np.float32)
        p_mats = nde.forward([type_p, conn_p, rot_p])
        robot_graph = hpd.probability_matrices_to_graph(*p_mats)

        passed, dx, v_rms = is_learner(robot_graph)
        status = "KEPT " if passed else "KILLED"
        print(
            f"[non-learner filter] attempt {attempt:02d}: {status} | "
            f"dx={dx:.4f} m, v_rms={v_rms:.4f} m/s "
            f"(need dx≥{MIN_DX:.3f} OR v_rms≥{MIN_V_RMS:.3f})"
        )

        if passed:
            return robot_graph

    raise RuntimeError(f"Failed to sample a learner body in {MAX_BODY_RETRIES} attempts")

## my code above

# === BODY EVOLUTION (NEW) =====================================================
# We evolve the NDE *input vectors* [type_p, conn_p, rot_p] with an outer optimizer.
# For each candidate body, we (cheaply) optimize controller weights inside (inner loop),
# evaluate with your 'fitness', and send that score back to the outer optimizer.

def decode_graph_from_body_vec(vec: np.ndarray, num_modules: int, genotype_size: int) -> "DiGraph":
    """vec is length 3*genotype_size: [type | conn | rot] in [0,1]."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    hpd = HighProbabilityDecoder(num_modules)
    type_p = vec[0:genotype_size].astype(np.float32)
    conn_p = vec[genotype_size:2*genotype_size].astype(np.float32)
    rot_p  = vec[2*genotype_size:3*genotype_size].astype(np.float32)
    p_mats = nde.forward([type_p, conn_p, rot_p])
    return hpd.probability_matrices_to_graph(*p_mats)

def optimize_last_layer_for_body(robot_graph: "DiGraph",
                                 base_weights: np.ndarray | None = None,
                                 seed: int = 42,
                                 budget: int = 120) -> tuple[np.ndarray, float]:
    """
    Inner loop: given a body, find good controller weights using Nevergrad CMA (or DE).
    Returns (best_weights, best_fitness).
    """
    rng = np.random.default_rng(seed)
    # Build a dummy model to get num_params from NeuralController
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)
    input_size = len(data.qpos) + len(data.qvel) + 2 + 3
    hidden_size = 8
    output_size = model.nu
    dummy = NeuralController(input_size, hidden_size, output_size, weights=None)
    num_params = dummy.num_params

    # Parameterization
    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(seed)
    if base_weights is not None and base_weights.shape[0] == num_params:
        parametrization.set_bounds(-3.0, 3.0)  # gentle box; optional
        parametrization.value = base_weights.copy()

    # Choose the controller optimizer (CMA or DE). CMA is strong; DE is simpler.
    # opt = ng.optimizers.CMA(parametrization=parametrization, budget=budget)
    opt = ng.optimizers.CMA(parametrization=parametrization, budget=budget)

    def objective(w):
        return -evaluate(w, robot_graph)

    rec = opt.minimize(objective)
    best_fit = -objective(rec.value)
    return rec.value, best_fit

def evolve_body(num_modules: int,
                genotype_size: int,
                outer_budget: int = 60,
                inner_budget: int = 120,
                seed: int = SEED) -> tuple["DiGraph", np.ndarray, float]:
    """
    Outer loop over the NDE vectors (3 * genotype_size).
    For each candidate:
      - decode to a graph
      - quick prescreen (kill non-movers)
      - inner optimize controller
    We *minimize* the negative fitness.
    """
    total_dim = 3 * genotype_size
    parametrization = ng.p.Array(shape=(total_dim,))
    parametrization.random_state.seed(seed)

    # Initialize in [0,1] as NDE inputs are probabilities
    init = RNG.random(total_dim).astype(np.float32)
    parametrization.value = init

    # Outer optimizer: use DE for simplicity (you can swap to CMA easily)
    # outer = ng.optimizers.CMA(parametrization=parametrization, budget=outer_budget)
    outer = ng.optimizers.DE(parametrization=parametrization, budget=outer_budget)

    A2_BASE_WEIGHTS = None  # or a warm start np.ndarray if you have one from A2

    def outer_objective(vec):
        # Keep values in [0,1] for NDE probabilities
        v = np.clip(np.asarray(vec, dtype=np.float32), 0.0, 1.0)
        try:
            graph = decode_graph_from_body_vec(v, num_modules, genotype_size)
        except Exception:
            # Bad decode → heavy penalty
            return 1e6

        # Non-learner prescreen: cheap sim with settle+random kicks
        passed, dx, v_rms = is_learner(graph)
        if not passed:
            # Penalize but keep some grad-less signal (smaller dx/v helps a bit)
            return 1e3 - 10.0 * (dx + v_rms)

        # Inner optimization over controller for this body
        best_w, best_fit = optimize_last_layer_for_body(
            graph,
            base_weights=A2_BASE_WEIGHTS,
            seed=seed,
            budget=inner_budget,
        )
        # We minimize, so return negative fitness
        return -best_fit

    rec = outer.minimize(outer_objective)
    best_vec = np.clip(rec.value, 0.0, 1.0).astype(np.float32)
    best_graph = decode_graph_from_body_vec(best_vec, num_modules, genotype_size)
    best_weights, best_fit = optimize_last_layer_for_body(
        best_graph, base_weights=None, seed=seed, budget=inner_budget
    )

    return best_graph, best_weights, best_fit
# === END BODY EVOLUTION (NEW) ================================================

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



def random_move(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Get the number of joints
    num_joints = model.nu

    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi / 2
    return RNG.uniform(
        low=-hinge_range,  # -pi/2
        high=hinge_range,  # pi/2
        size=num_joints,
    ).astype(np.float64)


# class NeuralController:
#     def __init__(self, input_size, hidden_size, output_size, weights=None):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         self.num_params = (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)

#         if weights is None:
#             self.weights = np.random.randn(self.num_params) * 0.1
#         else:
#             self.weights = np.array(weights)

#     def forward(self, inputs):
#         def tanh(x):
#             return np.tanh(x)

#         idx = 0
#         W1 = self.weights[idx: idx + self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
#         idx += self.input_size * self.hidden_size
#         W2 = self.weights[idx: idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size)
#         idx += self.hidden_size * self.hidden_size
#         W3 = self.weights[idx: idx + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)
        
#         layer1 = tanh(np.dot(inputs, W1))
#         layer2 = tanh(np.dot(layer1, W2))
#         outputs = tanh(np.dot(layer2, W3))  

#         return outputs.reshape(self.output_size)

class NeuralController:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # params: W1 + b1 + W2 + b2 + W3 + b3
        self.nW1 = input_size * hidden_size
        self.nb1 = hidden_size
        self.nW2 = hidden_size * hidden_size
        self.nb2 = hidden_size
        self.nW3 = hidden_size * output_size
        self.nb3 = output_size
        self.num_params = self.nW1 + self.nb1 + self.nW2 + self.nb2 + self.nW3 + self.nb3

        if weights is None:
            self.weights = np.random.randn(self.num_params) * 0.1
        else:
            self.weights = np.array(weights, dtype=float)

    def forward(self, inputs):
        idx = 0
        W1 = self.weights[idx: idx + self.nW1].reshape(self.input_size, self.hidden_size); idx += self.nW1
        b1 = self.weights[idx: idx + self.nb1];                                               idx += self.nb1
        W2 = self.weights[idx: idx + self.nW2].reshape(self.hidden_size, self.hidden_size);   idx += self.nW2
        b2 = self.weights[idx: idx + self.nb2];                                               idx += self.nb2
        W3 = self.weights[idx: idx + self.nW3].reshape(self.hidden_size, self.output_size);   idx += self.nW3
        b3 = self.weights[idx: idx + self.nb3];                                               idx += self.nb3

        h1 = np.tanh(inputs @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = np.tanh(h2 @ W3 + b3)   # [-1,1]
        return out.reshape(self.output_size)




# precompute once per model
def joint_phases(model):
    # evenly spaced phases per actuator
    return np.linspace(0.0, 2*np.pi, num=model.nu, endpoint=False)

PHASES = None  # global/cache

def rmat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo xmat -> roll,pitch,yaw (radians). R is 3x3."""
    # R is world-from-body; this is a standard Tait-Bryan (XYZ) extraction.
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = -np.arcsin(np.clip(R[2,0], -1.0, 1.0))
    yaw   = np.arctan2(R[1,0], R[0,0])
    return float(roll), float(pitch), float(yaw)

def controller(model, data, to_track, neural_net: NeuralController, record=True):
    # --- terrain features from the 'core' geom ---
    # (to_track[0] is bound to a core geom)
    R = np.array(to_track[0].xmat, dtype=float).reshape(3, 3)
    roll, pitch, _ = rmat_to_rpy(R)             # radians
    core_z = float(to_track[0].xpos[2])         # height above ground

    # --- inputs: add terrain cues (roll, pitch, height) ---
    inputs = np.concatenate([
        data.qpos.copy(), data.qvel.copy(),
        [np.sin(data.time * 2 * np.pi), np.cos(data.time * 2 * np.pi)],
        [roll, pitch, core_z],                    # <-- NEW (3 features)
    ])

    # ---- base CPG params (good defaults for bumpy ground) ----
    BASE_FREQ_FLAT = 1.6      # slower than on flat for better footing
    BIAS_AMP       = 0.40     # higher step clearance
    DUTY           = 0.65     # longer push phase
    CTRL_GAIN_BASE = 1.9      # stronger commands on bumps

    # --- adapt to tilt: slower but stronger when tilted ---
    tilt = min(1.0, 0.7*abs(roll) + 0.7*abs(pitch))     # ~[0,1+] from tilt magnitude
    base_freq = BASE_FREQ_FLAT * (1.0 - 0.5*tilt)       # reduce freq up to 50% when tilted
    ctrl_gain = CTRL_GAIN_BASE * (1.0 + 0.6*tilt)       # increase amplitude up to +60%

    omega  = 2 * np.pi * base_freq
    # anti-phase pattern helps over irregularities when limbs come in pairs
    phases = (np.arange(model.nu) % 2) * np.pi
    cpg    = np.sin(omega * data.time + phases)

    # oscillatory bias for clearance
    bias = BIAS_AMP * cpg

    # NN output in [-1,1], include bias once
    raw = np.tanh(neural_net.forward(inputs) + bias)
    assert raw.shape[0] == model.nu, f"NN outputs {raw.shape[0]} but model.nu={model.nu}"

    # duty gating: keep some thrust in the "weak" half
    gate = (cpg > 0).astype(np.float64) * DUTY + (1.0 - DUTY)
    raw *= gate

    # ======= actuation & smoothing =======
    ctrlrange = model.actuator_ctrlrange[:model.nu]
    lo, hi = ctrlrange[:, 0], ctrlrange[:, 1]

    # add small exploration noise (helps free when wedged)
    u_base = ctrl_gain * raw + EXPL_NOISE * RNG.standard_normal(model.nu)

    # stronger/longer kick gets you onto/over obstacles
    kick = KICK_GAIN if data.time < KICK_SECONDS else 1.0
    u = np.clip(u_base * kick, lo, hi)

    # slightly less smoothing on bumps (more reactive than glass-smooth)
    data.ctrl[:] = (1.0 - SMOOTHING) * data.ctrl[:] + SMOOTHING * u

    if record:
        HISTORY.append(to_track[0].xpos.copy())




def evaluate(weights,
             robot_graph,
             steps: int = 1500,
             record: bool = False,
             action_stride: int = 2,
             early_stop: bool = True) -> float:
    """
    One evaluation of a controller on a given body.
    - Builds world+model, runs a short rollout, and returns a progress-oriented fitness.
    - 'record=False' avoids HISTORY spam during optimization.
    - 'action_stride>1' applies the controller every k steps (cheap speedup).
    - Early-stops if the robot is stagnant for ~0.8s after the first 0.5s.
    """
    # --- world & model ---
    mj.set_mjcb_control(None)
    world = OlympicArena()
    robot = construct_mjspec_from_graph(robot_graph)
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)

    # --- track the 'core' geom(s) ---
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]
    if not to_track:
        return -1e6  # no core found → bad body

    # --- controller net ---
    input_size  = len(data.qpos) + len(data.qvel) + 2 + 3
    hidden_size = 8
    output_size = model.nu
    net = NeuralController(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=output_size,
                           weights=weights)

    # --- rollout ---
    joint_history = []
    x_hist = []
    dt = float(model.opt.timestep)

    prev_xy = np.array(to_track[0].xpos[:2], dtype=float)
    stagnant = 0
    # ~0.8s of no movement allowed (after 0.5s)
    stagnation_limit = max(1, int(1.5 / max(dt, 1e-6)))
    min_move = 0.0015  # meters between checks

    for t in range(steps):
        if (t % action_stride) == 0:
            # your controller MUST accept record=...; if not, remove the arg
            controller(model, data, to_track, net, record=record)
        mj.mj_step(model, data)

        # logs for fitness
        joint_history.append(data.ctrl.copy())
        x_hist.append(float(to_track[0].xpos[0]))

        # early-stagnation check
        if early_stop and data.time > 0.5:
            cur_xy = np.array(to_track[0].xpos[:2], dtype=float)
            if np.linalg.norm(cur_xy - prev_xy) < min_move:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant > stagnation_limit:
                break
            prev_xy = cur_xy

    joint_history = np.asarray(joint_history) if joint_history else np.zeros((1, model.nu))
    x_hist = np.asarray(x_hist) if x_hist else np.zeros(1)

    return fitness_adv(to_track, joint_history, x_hist)


def fitness_adv(to_track, joint_history: np.ndarray, x_hist: np.ndarray) -> float:
    """
    Progress-oriented fitness:
      + sum of positive forward deltas (progress)
      + final forward x
      - sideways drift
      +/- light regularization on joint activity
    """
    final_x = float(to_track[0].xpos[0])
    drift_y = abs(float(to_track[0].xpos[1]))

    # accumulated forward progress (only positive steps)
    progress = float(np.maximum(0.0, np.diff(x_hist)).sum()) if x_hist.size > 1 else 0.0

    # activity regularizers (kept small)
    osc = float(np.mean(np.std(joint_history, axis=0))) if joint_history.size else 0.0
    # encourage using range but not slamming extremes (simple L1 on ctrl changes)
    if joint_history.shape[0] > 1:
        du = np.diff(joint_history, axis=0)
        smooth_pen = float(np.mean(np.abs(du)))
    else:
        smooth_pen = 0.0

    # weights are mild; main driver is progress + final_x
    return 0.7 * progress + 0.5 * final_x + 0.05 * osc - 0.03 * smooth_pen - 0.30 * drift_y



def experiment(robot_graph: Any, mode: ViewerTypes = "viewer") -> np.ndarray:
    """Run the simulation with optimizer to find best weights."""
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    robot = construct_mjspec_from_graph(robot_graph)
    # Create world and spawn robot
    # robot=gecko()
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    # Compile full world (includes robot)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Neural controller setup
    input_size = len(data.qpos) + len(data.qvel) + 2
    hidden_size = 8
    output_size = model.nu
    dummy_net = NeuralController(input_size, hidden_size, output_size)
    num_params = dummy_net.num_params

    # Nevergrad optimizer
    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(SEED)
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=3000)

    # Objective function
    def objective(x):
        return -evaluate(x, robot_graph)  # evaluate uses world + compiled model

    # Run optimization
    recommendation = optimizer.minimize(objective)
    print("Best fitness:", -objective(recommendation.value))
    return recommendation.value



def smoke_test_once(num_modules=8, genotype_size=16):
    """Quick pipeline check: decode random body -> prescreen -> 200-step eval with random weights."""
    print("\n[SMOKE] starting quick pipeline check")
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    hpd = HighProbabilityDecoder(num_modules)
    # small genotype for speed
    type_p = RNG.random(genotype_size).astype(np.float32)
    conn_p = RNG.random(genotype_size).astype(np.float32)
    rot_p  = RNG.random(genotype_size).astype(np.float32)
    p_mats = nde.forward([type_p, conn_p, rot_p])
    graph  = hpd.probability_matrices_to_graph(*p_mats)

    passed, dx, v_rms = is_learner(graph)
    print(f"[SMOKE] prescreen pass={passed} | dx={dx:.3f} v_rms={v_rms:.3f}")

    # Tiny eval: compile once, run 200 steps with random controller weights
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(graph)
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]

    input_size = len(data.qpos) + len(data.qvel) + 2
    net = NeuralController(input_size, 8, model.nu)  # random weights
    steps = 200
    joint_hist = []
    for _ in range(steps):
        controller(model, data, to_track, net)
        mj.mj_step(model, data)
        joint_hist.append(data.ctrl.copy())
    fit = fitness_adv(to_track, np.array(joint_hist), np.array([d[0] for d in to_track]))
    print(f"[SMOKE] 200-step fitness={fit:.4f}, end_x={to_track[0].xpos[0]:.3f}")


def main() -> None:
    num_modules   = 20
    genotype_size = 64

    # Evolve body + controller jointly (outer = body, inner = controller)
    best_graph, best_weights, best_fit = evolve_body(
        num_modules=num_modules,
        genotype_size=genotype_size,
        outer_budget=120,   # try 100–200 if you have time
        inner_budget=600,  # per-body controller budget (keep modest)
        seed=SEED,
    )
    print(f"[RESULT] best fitness = {best_fit:.4f}")

    # Save body JSON (needed for Robot Olympics submission)
    save_graph_as_json(best_graph, DATA / "robot_graph.json")

    # === Launch the final run (unchanged) ===
    core = construct_mjspec_from_graph(best_graph)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    input_size = len(data.qpos) + len(data.qvel) + 2 + 3
    neural_net = NeuralController(input_size, 8, model.nu, best_weights)
    HISTORY.clear()
    mj.set_mjcb_control(lambda m, d: controller(m, d, to_track, neural_net))
    viewer.launch(model=model, data=data)
    show_xpos_history(HISTORY)



if __name__ == "__main__":
    main()
