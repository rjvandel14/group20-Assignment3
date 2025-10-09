"""Assignment 3 template code with Stepwise Controller."""

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
SPAWN_POS = [
        [-1, 0.0, 0.1],
        [1.0, 0.0, 0.1],
        [3, 0.0, 0.1],
    ]

NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

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

def fitness_function(history: list[float], penalty) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return cartesian_distance+penalty

def fitness(history: list[float], joint_history, ):
    final_pos = history[-1]
    displacement_y = abs(final_pos[1])  
    displacement_x = final_pos[0] 

    oscillation_reward = np.mean(np.std(joint_history, axis=0)) # variation of the joint angles

    saturation_penalty = np.mean(np.abs(np.abs(joint_history) - (np.pi / 2))) # to avoid getting stuck on max or min joint angles

    fitness = displacement_x + 0.08 * oscillation_reward - 0.08 * saturation_penalty - 0.3*displacement_y

    return fitness


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
    ym0, ymc = 0, SPAWN_POS[1][0]

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

class StepwiseController:
    def __init__(self, neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.1):
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

    controller = StepwiseController(neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.1)

    steps = 1500
    joint_history = []

    for _ in range(steps):
        controller.step(model, data)
        mj.mj_step(model, data)
        joint_history.append(data.ctrl.copy())

    # Return fitness computed from the tracker's recorded xpos
    return fitness_function(tracker.history["xpos"][0], penalty)

def experiment(robot_graph: Any, penalty) -> np.ndarray:
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS[0])

    # Compile model that actually contains the robot to get correct qpos/qvel sizes
    model_tmp = world.spec.compile()
    data_tmp = mj.MjData(model_tmp)
    mj.mj_resetData(model_tmp, data_tmp)
    mj.mj_forward(model_tmp, data_tmp)

    input_size = len(data_tmp.qpos) + len(data_tmp.qvel) + 2
    dummy_net = NeuralController(input_size, 8, model_tmp.nu)
    num_params = dummy_net.num_params

    parametrization = ng.p.Array(shape=(num_params,))
    parametrization.random_state.seed(SEED)
    optimizer = ng.optimizers.CMA(parametrization=num_params, budget=600)

    spawn_positions = SPAWN_POS
    def objective(x):
        weights = np.asarray(x)
        scores = []
        for sp in spawn_positions:
            try:
                f = evaluate(weights, robot_graph, sp, penalty)
            except Exception as e:
                print("Evaluation error at spawn", sp, ":", e)
                f = 1e6
            scores.append(f)

        return float(np.mean(scores))

    recommendation = optimizer.minimize(objective)
    print("Best aggregated fitness:", objective(recommendation.value))
    return recommendation.value


def main() -> None:
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    num_blocks = count_from_graph(robot_graph, "BRICK")
    num_hinges = count_from_graph(robot_graph, "HINGE")
    
    penalty = 0
    ratio = num_blocks/(num_hinges)
    if ratio > 1:
        penalty = 0.3

    save_graph_as_json(robot_graph, DATA / "robot_graph.json")
    core = construct_mjspec_from_graph(robot_graph)

    mj.set_mjcb_control(None)
    best_weights = experiment(robot_graph, penalty)

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
