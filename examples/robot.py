import json
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json  # only save/load graph
from networkx.readwrite import json_graph
from ariel.utils.renderers import single_frame_renderer

  # if available
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena
from mujoco import viewer
import mujoco as mj
from ariel.utils.tracker import Tracker

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
SPAWN_POS = [-0.8, 0.0, 0.1]

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

# path to saved snapshot (replace)
save_dir = Path("./__data__/es_anytime")
# read last line from best_history.csv
history = save_dir / "best_history.csv"
with open(history, "r") as fh:
    lines = [l.strip() for l in fh if l.strip()]
last = lines[-1].split(",")
graph_path = Path(last[4])
weights_path = Path(last[5])

# load weights (1D array)
weights = np.loadtxt(weights_path, delimiter=",")

# load the graph (if you saved using the same JSON format)
with open(graph_path, "r") as fh:
    graph_json = json.load(fh)

# Tell NetworkX the edges are stored under "edges" (not "links")
robot_graph = json_graph.node_link_graph(graph_json, edges="edges")

# build model to infer sizes
world = OlympicArena()
robot = construct_mjspec_from_graph(robot_graph)
world.spawn(robot.spec, spawn_position=SPAWN_POS)   # any legal spawn
model = world.spec.compile()
data = mj.MjData(model)
mj.mj_resetData(model, data)
mj.mj_forward(model, data)

input_size = len(data.qpos) + len(data.qvel) + 2
output_size = model.nu
hidden_size = 8   # must match training

# instantiate network
neural_net = NeuralController(input_size, hidden_size, output_size, weights)

# use the network: example forward with zeros
inputs = np.zeros(input_size)
outs = neural_net.forward(inputs)
print("network output shape:", outs.shape)

tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
tracker.setup(world.spec, data)
tracker.update(data)  # initialize history

# --- Setup Stepwise controller with your trained network ---
stepwise_ctrl = StepwiseController(
    neural_net,     # your loaded NeuralController
    tracker,
    ctrl_every=5,   # control interval (as in training)
    save_every=100, # save tracker history interval
    alpha=0.1       # smoothing factor
)

# --- Tell MuJoCo to use your controller ---
mj.set_mjcb_control(lambda m, d: stepwise_ctrl.step(m, d))

# --- Launch the viewer ---
viewer.launch(model=model, data=data)
show_xpos_history(tracker.history["xpos"][0])

# After closing the viewer, you can inspect tracker history
print(tracker.history["xpos"][0])

