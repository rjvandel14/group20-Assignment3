"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from networkx import Graph
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleType,
)

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
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

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [
    [-0.8, 0.0, 0.1],
    [1.0, 0.0, 0.2],
    [3.0, 0.0, 0.2],
]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# Local scripts
from A3_plot_function import show_xpos_history


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


def fitness_function(history: list[tuple[float, float, float]],graph: Graph) -> float:
    sym_bonus = 0.3 * symmetry_score(graph)            # subtract later (good thing)
    pen_arch  = arch_penalty(graph, base=0.20)         # add
    pen_hinge = single_connection_hinge_penalty(graph, w_single=0.10)  # add
    pen_stab  = stability_penalty(graph, z_min=0.15, weight=0.2)       # add (tuned lower)

    penalty = (pen_arch + pen_hinge + pen_stab) - sym_bonus

    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return cartesian_distance+penalty


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def experiment(
    spawn_pos,
    robot: Any,
    controller: Controller,
    duration: int = 30,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena(
        load_precompiled=False,
    )

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(
        robot.spec,
        position=spawn_pos,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            cam_quat = np.zeros(4)
            mj.mju_euler2Quat(cam_quat, np.deg2rad([30, 0, 0]), "XYZ")
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
                cam_fovy=7,
                cam_pos=[2, -1, 2],
                cam_quat=cam_quat,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #


def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    scale = 8192
    genotype_size = 64
    type_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
    conn_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
    rot_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
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

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
    fitnesses = []

    for pos in SPAWN_POS:
        tracker = Tracker(
            mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
            name_to_bind="core",
        )
        ctrl = Controller(
            controller_callback_function=nn_controller,
            tracker=tracker,
        )

        experiment(spawn_pos=pos, robot=core, controller=ctrl, mode="simple")  # Use "simple" for faster evaluation

        # Compute fitness for this spawn position
        fit = fitness_function(tracker.history["xpos"][0], robot_graph)
        fitnesses.append(fit)

    # Average fitness
    average_fitness = float(np.mean(fitnesses))
    console.log(f"Average fitness over {len(SPAWN_POS)} spawn positions: {average_fitness}")

    show_xpos_history(
        tracker.history["xpos"][0],
        spawn_position=SPAWN_POS[0],
        target_position=TARGET_POSITION,
        save=True,
        show=True,
    )

    print(fitnesses)

    # fitness = fitness_function(tracker.history["xpos"][0], robot_graph)
    # msg = f"Fitness of generated robot: {fitness}"
    # console.log(msg)


if __name__ == "__main__":
    main()
