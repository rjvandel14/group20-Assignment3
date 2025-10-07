def run_evolution_strategy(
    mu=10, lambd=20, gens=10, *,
    steps=1500, mut_sigma=0.10, mut_prob=0.20,
    seed=42, out_dir="__data__"
):
    """
    One-call (μ+λ) Evolution Strategy for body genomes (3×64 in [0,1]).
    - Builds each body via body_opt.build_robot(genome)  -> (model, data, to_track)
    - Trains a small tanh MLP controller by local random search
    - Uses PD tracking + action subsampling during rollout
    - Saves best genome/weights to out_dir
    Returns: (best_genome, best_weights, best_fitness)
    """
    import os, time, math, contextlib
    import numpy as np
    import mujoco as mj
    from dataclasses import dataclass

    # ---- config (PD + control cadence) ----
    CONTROL_SUBSTEPS = 10
    KP, KD = 2.0, 0.05
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- local helpers (standalone) ----
    BODY_DIM = 64
    GENOME_SHAPE = (3, BODY_DIM)

    def random_genome():
        return rng.random(GENOME_SHAPE, dtype=np.float32)

    def mutate_genome(g, sigma=mut_sigma, prob=mut_prob):
        noise = rng.normal(0.0, sigma, size=g.shape).astype(np.float32)
        mask  = rng.random(g.shape) < prob
        out   = g + noise * mask
        np.clip(out, 0.0, 1.0, out=out)
        return out

    class TanhMLP:
        def __init__(self, in_dim, hidden, out_dim, rng_local):
            h1, h2 = hidden
            self.W1 = rng_local.normal(0, 1/np.sqrt(in_dim), size=(h1, in_dim)).astype(np.float32)
            self.b1 = np.zeros(h1, np.float32)
            self.W2 = rng_local.normal(0, 1/np.sqrt(h1), size=(h2, h1)).astype(np.float32)
            self.b2 = np.zeros(h2, np.float32)
            self.W3 = rng_local.normal(0, 1/np.sqrt(h2), size=(out_dim, h2)).astype(np.float32)
            self.b3 = np.zeros(out_dim, np.float32)

        @property
        def weights(self):
            return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3])

        @weights.setter
        def weights(self, flat):
            i=0
            s=self.W1.size; self.W1[:] = flat[i:i+s].reshape(self.W1.shape); i+=s
            s=self.b1.size; self.b1[:] = flat[i:i+s]; i+=s
            s=self.W2.size; self.W2[:] = flat[i:i+s].reshape(self.W2.shape); i+=s
            s=self.b2.size; self.b2[:] = flat[i:i+s]; i+=s
            s=self.W3.size; self.W3[:] = flat[i:i+s].reshape(self.W3.shape); i+=s
            s=self.b3.size; self.b3[:] = flat[i:i+s]

        def forward(self, x):
            z1 = np.tanh(self.W1 @ x + self.b1)
            z2 = np.tanh(self.W2 @ z1 + self.b2)
            return np.tanh(self.W3 @ z2 + self.b3)

    @contextlib.contextmanager
    def suppress_c_stderr():
        import os, sys
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = sys.stderr.fileno()
            saved = os.dup(stderr_fd)
            os.dup2(devnull, stderr_fd); os.close(devnull)
            try: yield
            finally: os.dup2(saved, stderr_fd); os.close(saved)
        except Exception:
            yield

    def pd_step(model, data, target):
        nu = model.nu
        qpos = data.qpos[:nu]
        qvel = data.qvel[:nu]
        err  = target - qpos
        torque = KP * err - KD * qvel
        data.ctrl[:nu] = np.clip(torque, -0.8, 0.8)

    def rollout(model, data, net, to_track, steps):
        end_x = end_y = end_z = 0.0
        target = None
        for t in range(steps):
            if t % CONTROL_SUBSTEPS == 0:
                obs = np.concatenate([
                    data.qpos.copy(), data.qvel.copy(),
                    [math.sin(data.time * 2*math.pi)], [math.cos(data.time * 2*math.pi)]
                ]).astype(np.float32)
                raw = net.forward(obs)
                target = np.clip(raw * (math.pi/2), -math.pi/2, math.pi/2)
            pd_step(model, data, target)
            mj.mj_step(model, data)
        # try site/body; fallback to root qpos
        try:
            end = to_track[0].xpos.copy()
            end_x, end_y, end_z = float(end[0]), float(end[1]), float(end[2])
        except Exception:
            end_x = float(data.qpos[0]) if data.qpos.size > 0 else 0.0
            end_y = float(data.qpos[1]) if data.qpos.size > 1 else 0.0
            end_z = float(data.qpos[2]) if data.qpos.size > 2 else 0.0
        return end_x, end_y, end_z

    def fitness(x, y, z):
        return x - 0.2*abs(y) - (2.0 if z < 0.03 else 0.0)

    # ---- body builder from your project ----
    from body_opt import build_robot  # expects RobotBuild(model, data, to_track)

    # ---- ES loop ----
    print("\n" + "="*70 + "\n(μ+λ) EVOLUTION: START\n" + "="*70)
    t0 = time.time()

    # init parents
    parents = [random_genome() for _ in range(mu)]
    parent_fit = [-1e9]*mu
    parent_w   = [None]*mu

    best_genome = None
    best_fit = -1e9
    best_w = None

    for gen in range(1, gens+1):
        print(f"\n------ GENERATION {gen}/{gens} ------")
        # make children
        children = [mutate_genome(parents[rng.integers(0, mu)]) for _ in range(lambd)]

        # evaluate a genome (build + tiny controller search)
        def eval_genome(g, seed_off):
            local_rng = np.random.default_rng(seed + seed_off)
            with suppress_c_stderr():
                rb = build_robot(g)
            m, d, track = rb.model, rb.data, rb.to_track
            # solver knobs
            m.opt.timestep = 0.002
            m.opt.iterations = 50
            m.opt.ls_iterations = 50
            # dims
            nu = m.nu
            obs_dim = d.qpos.size + d.qvel.size + 2
            # small local search on controller
            best_local_fit = -1e9
            best_local_w = None
            x=y=z=0.0
            restarts, local_steps, local_sigma = 6, 24, 0.05
            for r in range(restarts):
                net = TanhMLP(obs_dim, (64, 64), nu, local_rng)
                base_w = net.weights.copy()
                mj.mj_resetData(m, d)
                x0,y0,z0 = rollout(m, d, net, track, steps)
                f0 = fitness(x0,y0,z0)
                for s in range(local_steps):
                    trial = base_w + local_rng.normal(0.0, local_sigma, size=base_w.shape)
                    net.weights = trial
                    mj.mj_resetData(m, d)
                    x1,y1,z1 = rollout(m, d, net, track, steps)
                    f1 = fitness(x1,y1,z1)
                    if f1 > f0:
                        base_w, f0, (x0,y0,z0) = trial, f1, (x1,y1,z1)
                if f0 > best_local_fit:
                    best_local_fit, best_local_w, (x,y,z) = f0, base_w.copy(), (x0,y0,z0)
                    print(f"    controller restart {r+1}/{restarts} | best so far = {best_local_fit:.3f}")
            print(f"EVAL | END_X={x:.3f} END_Y={y:.3f} END_Z={z:.3f} | FIT={best_local_fit:.3f}")
            return best_local_fit, best_local_w

        # evaluate any unevaluated parents
        for i in range(mu):
            if parent_fit[i] <= -1e8:
                try:
                    f, w = eval_genome(parents[i], 10000 + gen*100 + i)
                except Exception as e:
                    print(f"  [parent fail] {i}: {e}")
                    f, w = -1e9, None
                parent_fit[i], parent_w[i] = f, w
                if f > best_fit:
                    best_fit, best_genome, best_w = f, parents[i].copy(), (w.copy() if w is not None else None)
                    print(f"*** NEW BEST = {best_fit:.3f} ***")

        # evaluate children
        child_fit, child_w = [], []
        for ci, c in enumerate(children):
            try:
                f, w = eval_genome(c, 20000 + gen*100 + ci)
            except Exception as e:
                print(f"  [child fail] {ci}: {e}")
                f, w = -1e9, None
            child_fit.append(f); child_w.append(w)
            if f > best_fit:
                best_fit, best_genome, best_w = f, c.copy(), (w.copy() if w is not None else None)
                print(f"*** NEW BEST = {best_fit:.3f} ***")

        # selection (elitist): take best μ from parents+children
        pool = parents + children
        pool_fit = parent_fit + child_fit
        pool_w = parent_w + child_w
        order = np.argsort(pool_fit)[::-1]
        parents = [pool[i] for i in order[:mu]]
        parent_fit = [pool_fit[i] for i in order[:mu]]
        parent_w = [pool_w[i] for i in order[:mu]]

        # save gen-best
        gi = int(np.argmax(parent_fit))
        np.save(os.path.join(out_dir, f"gen{gen:03d}_best_genome.npy"), parents[gi])
        if parent_w[gi] is not None:
            np.save(os.path.join(out_dir, f"gen{gen:03d}_best_weights.npy"), parent_w[gi])
        with open(os.path.join(out_dir, f"gen{gen:03d}_best_fit.txt"), "w") as f:
            f.write(str(parent_fit[gi]))

    dt = time.time() - t0
    print("\n" + "="*70 + f"\n(μ+λ) EVOLUTION: END | WALLTIME={dt:.1f}s\n" + "="*70)

    # final snapshot
    if best_genome is not None:
        np.save(os.path.join(out_dir, "final_best_genome.npy"), best_genome)
        if best_w is not None:
            np.save(os.path.join(out_dir, "final_best_weights.npy"), best_w)
        with open(os.path.join(out_dir, "final_best_fit.txt"), "w") as f:
            f.write(str(best_fit))
    return best_genome, best_w, best_fit

# In your A3_own_control.py (or a notebook)
best_genome, best_weights, best_fit = run_evolution_strategy(
    mu=10, lambd=20, gens=10, steps=1500, seed=42
)
