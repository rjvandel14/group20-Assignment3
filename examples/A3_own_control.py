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
SPAWN_POS = [-0.8, 0.0, 0.1]
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

    return cartesian_distance+arch_penalty*ratio

def fitness(history: list[float], joint_history):
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


class NeuralController:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_params = (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)

        if weights is None:
            self.weights = np.random.randn(self.num_params) * 0.1
        else:
            self.weights = np.array(weights)

    def forward(self, inputs):
        def tanh(x):
            return np.tanh(x)

        idx = 0
        W1 = self.weights[idx: idx + self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        idx += self.input_size * self.hidden_size
        W2 = self.weights[idx: idx + self.hidden_size * self.hidden_size].reshape(self.hidden_size, self.hidden_size)
        idx += self.hidden_size * self.hidden_size
        W3 = self.weights[idx: idx + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)
        
        layer1 = tanh(np.dot(inputs, W1))
        layer2 = tanh(np.dot(layer1, W2))
        outputs = tanh(np.dot(layer2, W3))  

        return outputs.reshape(self.output_size)

def controller(model, data, to_track, neural_net: NeuralController):
    inputs = np.concatenate([data.qpos.copy(),data.qvel.copy(),[np.sin(data.time * 2 * np.pi)],[np.cos(data.time * 2 * np.pi)]])

    raw_output = neural_net.forward(inputs)
    target_angles = raw_output * (np.pi / 2) # map the output to possible joint angles

    smoothing = 0.1  # 0: no smooting, 1: only target angles 
    data.ctrl[:] = (1 - smoothing) * data.ctrl[:] + smoothing * target_angles

    HISTORY.append(to_track[0].xpos.copy())

def evaluate(weights,robot_graph):
    world = OlympicArena()
    mj.set_mjcb_control(None)
    robot = construct_mjspec_from_graph(robot_graph)
    #robot= gecko()
    world.spawn(robot.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mj.MjData(model)

    data.qpos[:] = 0.0 # reset position
    data.qvel[:] = 0.0 # reset velocity

    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    input_size = len(data.qpos) + len(data.qvel) + 2 
    neural_net = NeuralController(input_size=input_size,hidden_size=8,output_size=model.nu,weights=weights)

    steps = 2500
    joint_history = []

    for _ in range(steps):
        controller(model, data, to_track, neural_net)
        mj.mj_step(model, data)
        joint_history.append(data.ctrl.copy())

    return fitness_function(tracker.history["xpos"][0], robot_graph)


def experiment(robot_graph: Any) -> np.ndarray:
    mj.set_mjcb_control(None)
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
    optimizer = ng.optimizers.CMA(parametrization=num_params, budget=200)


    # Objective function
    def objective(x):
        return evaluate(x, robot_graph)

    # Run optimization
    recommendation = optimizer.minimize(objective)
    print("Best fitness:", objective(recommendation.value))
    return recommendation.value


def main() -> None:
    """Entry point."""
    num_modules = 20
    genotype_size = 64

    # # Random genotype
    # type_p_genes = RNG.random(genotype_size).astype(np.float32)
    # conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    # rot_p_genes = RNG.random(genotype_size).astype(np.float32)
    # genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    # # Decode robot
    # nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    # p_matrices = nde.forward(genotype)
    # hpd = HighProbabilityDecoder(num_modules)
    # robot_graph = hpd.probability_matrices_to_graph(
    #     p_matrices[0], p_matrices[1], p_matrices[2]
    # )

    # Sample a body that passes the non-learner prescreen
    robot_graph = sample_learner_robot_graph(num_modules=num_modules, genotype_size=genotype_size)

    save_graph_as_json(robot_graph, DATA / "robot_graph.json")
    core = construct_mjspec_from_graph(robot_graph)
    # core = gecko()
    # Clear old history
    HISTORY.clear()

    # Optimize weights
    best_weights = experiment(robot_graph=robot_graph)

    # Create world and spawn robot for simulation
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # Neural controller
    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, best_weights)
    tracker.update(data)
    stepwise_ctrl = StepwiseController(neural_net, tracker, ctrl_every=5, save_every=100, alpha=0.1)
    mj.set_mjcb_control(lambda m, d: stepwise_ctrl.step(m, d))

    # Launch viewer
    viewer.launch(model=model, data=data)

    # Show path
    show_xpos_history(HISTORY)


if __name__ == "__main__":
    main()

