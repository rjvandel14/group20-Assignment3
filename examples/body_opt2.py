
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

def evaluate(weights,robot_graph, steps: int|None = None):
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

    if steps is None:
        steps = CFG["eval_steps"]
    joint_history = []

    for _ in range(steps):
        controller(model, data, to_track, neural_net)
        mj.mj_step(model, data)
        joint_history.append(data.ctrl.copy())

    joint_history = np.array(joint_history)
    return fitness(to_track, joint_history)

def fitness(to_track, joint_history):
    final_pos = to_track[0].xpos.copy()
    displacement_y = abs(final_pos[1])  
    displacement_x = final_pos[0] 

    oscillation_reward = np.mean(np.std(joint_history, axis=0)) # variation of the joint angles

    saturation_penalty = np.mean(np.abs(np.abs(joint_history) - (np.pi / 2))) # to avoid getting stuck on max or min joint angles

    fitness = displacement_x + 0.08 * oscillation_reward - 0.08 * saturation_penalty - 0.3*displacement_y

    return fitness

def experiment(robot_graph: Any, mode: ViewerTypes = "viewer", budget: int | None = None) -> np.ndarray:
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
    if budget is None:
        budget = CFG["inner_budget"]
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=budget)

    # Objective function
    def objective(x):
        return -evaluate(x, robot_graph)  # evaluate uses world + compiled model

    # Run optimization
    recommendation = optimizer.minimize(objective)
    print("Best fitness:", -objective(recommendation.value))
    return recommendation.value

def evolve_body(mu=4, lam=8, gens=10, num_modules=20, genotype_size=64):
    # Parents are (3, G) arrays: [type_p, conn_p, rot_p]
    parents = [np.random.rand(3, genotype_size).astype(np.float32) for _ in range(mu)]
    parent_fitness = [evaluate_body(p, num_modules) for p in parents]

    print("\n======================================================================")
    print("(μ+λ) EVOLUTION: START")
    print("======================================================================\n")

    for gen in range(1, gens + 1):
        print(f"------ GENERATION {gen}/{gens} ------")
        offspring = []

        for _ in range(lam):
            parent = random.choice(parents)

            # ✅ HERE: create a proper child (3, G) float array
            child = np.clip(
                parent + np.random.normal(0, 0.05, parent.shape),
                0.0, 1.0
            ).astype(np.float32)

            off_fit = evaluate_body(child, num_modules)
            offspring.append((child, off_fit))

        # Select top μ among parents + offspring
        pool = list(zip(parents, parent_fitness)) + offspring
        pool.sort(key=lambda it: it[1], reverse=True)
        parents, parent_fitness = zip(*pool[:mu])
        parents, parent_fitness = list(parents), list(parent_fitness)

        print(f"  best fitness this gen = {parent_fitness[0]:.3f}\n")

    best_idx = int(np.argmax(parent_fitness))
    return parents[best_idx]

def evaluate_body(genotype, num_modules: int) -> float:
    # Accept either a (3, G) array or a list/tuple of 3 arrays
    if isinstance(genotype, np.ndarray):
        if genotype.ndim != 2 or genotype.shape[0] != 3:
            raise ValueError(f"Genotype must be shape (3, G); got {genotype.shape}")
        # ✅ HERE: unpack the three 1-D arrays for decoding
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
    for _ in range(K):
        best_weights = experiment(robot_graph, budget=CFG["inner_budget"])
        fit = evaluate(best_weights, robot_graph, steps=CFG["eval_steps"])
        scores.append(fit)
    return float(np.mean(scores))

# --- TRAINING CONFIG ---
CFG = {
    "inner_budget": 300,   # controller CMA steps (was 200). Try 500–1000 for long runs.
    "eval_steps": 4000,    # physics steps per evaluation (was 2500)
    "gens": 15,            # body ES generations (was 5/10)
    "mu": 6,               # parents (was 4)
    "lam": 18,             # offspring per gen (was 8)
    "repeats": 2,          # evaluate each body K times and average (reduces noise)
}


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
    #core = construct_mjspec_from_graph(robot_graph)  # rebuild before reuse
    # Bind robot geoms to track
    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Tracker
    # tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # Neural controller
    input_size = len(data.qpos) + len(data.qvel) + 2
    neural_net = NeuralController(input_size, 8, model.nu, best_weights)
    HISTORY.clear()
    mj.set_mjcb_control(lambda m, d: controller(m, d, to_track, neural_net))

    # Launch viewer
    viewer.launch(model=model, data=data)

    # Show path
    show_xpos_history(HISTORY)


if __name__ == "__main__":
    main()
