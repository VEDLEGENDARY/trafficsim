import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import pickle

# Simulation parameters
NUM_CARS = 10
CIRCLE_RADIUS = 20.0
BASE_SPEED = 8.0
SPEED_MIN = 0.9
SPEED_MAX = 1.1
SIM_DURATION = 180.0
DT = 0.05
SAMPLES_PER_GEN = 100
NUM_GENERATIONS = 10
SAVE_DIR = "gen_results"

np.random.seed(42)

class Car:
    def __init__(self, idx, angle, speed_mult):
        self.idx = idx
        self.angle = angle
        self.speed_mult = speed_mult
        self.base_speed = BASE_SPEED
        self.laps = 0
        self.last_angle = angle

    def update(self, dt):
        speed = self.base_speed * self.speed_mult
        dtheta = speed * dt / CIRCLE_RADIUS
        self.angle = (self.angle + dtheta) % (2 * np.pi)
        # Count laps (crossing angle 0)
        if (self.last_angle < np.pi) and (self.angle >= np.pi):
            self.laps += 0.5
        if (self.last_angle >= np.pi) and (self.angle < np.pi):
            self.laps += 0.5
        self.last_angle = self.angle

class SampleResult:
    def __init__(self, laps, car_states, seed):
        self.laps = laps
        self.car_states = car_states
        self.seed = seed

# Run a single simulation sample, return total laps and car state history
def run_sample(seed=None, record_frames=False):
    rng = np.random.default_rng(seed)
    cars = [
        Car(idx=i,
            angle=rng.uniform(0, 2 * np.pi),
            speed_mult=rng.uniform(SPEED_MIN, SPEED_MAX))
        for i in range(NUM_CARS)
    ]
    frames = []
    steps = int(SIM_DURATION / DT)
    for _ in range(steps):
        if record_frames:
            frames.append([(car.angle, car.speed_mult) for car in cars])
        for car in cars:
            car.update(DT)
    total_laps = sum(car.laps for car in cars)
    return SampleResult(total_laps, frames, seed)

def run_generation(gen_num):
    results = []
    for i in range(SAMPLES_PER_GEN):
        res = run_sample(seed=gen_num * 1000 + i, record_frames=False)
        results.append((res.laps, i, res.seed))
    results.sort(reverse=True)
    top_indices = [r[1] for r in results[:15]]
    top_seeds = [r[2] for r in results[:15]]
    top_results = [run_sample(seed=s, record_frames=True) for s in top_seeds]
    best_result = top_results[0]
    return best_result, top_results

def save_video(sample_result, filename):
    frames = sample_result.car_states
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-CIRCLE_RADIUS-5, CIRCLE_RADIUS+5)
    ax.set_ylim(-CIRCLE_RADIUS-5, CIRCLE_RADIUS+5)
    ax.axis('off')
    circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='gray', fill=False, linewidth=3)
    ax.add_patch(circle)
    car_dots = [ax.plot([], [], 'o', color=plt.cm.hsv(i / NUM_CARS), markersize=12)[0] for i in range(NUM_CARS)]
    images = []
    for frame in frames[::8]:
        for i, (angle, speed_mult) in enumerate(frame):
            x = CIRCLE_RADIUS * np.cos(angle)
            y = CIRCLE_RADIUS * np.sin(angle)
            car_dots[i].set_data([x], [y])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        argb = np.frombuffer(renderer.tostring_argb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        argb = argb.reshape((h, w, 4))
        rgb = argb[:, :, 1:4]
        images.append(rgb)
    imageio.mimsave(filename, images, fps=30)
    plt.close(fig)

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    all_best = []
    last_top15 = []
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"Generation {gen}...")
        best_result, top15_results = run_generation(gen)
        # Save top 1 video
        vid_name = os.path.join(SAVE_DIR, f"gen{gen:03d}_top1.mp4")
        save_video(best_result, vid_name)
        # Save top 1 sample
        with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_top1.pkl"), "wb") as f:
            pickle.dump(best_result, f)
        all_best.append(best_result)
        # Save top 15 from last generation
        if gen == NUM_GENERATIONS:
            for i, sample in enumerate(top15_results):
                with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_top{i+1:02d}.pkl"), "wb") as f:
                    pickle.dump(sample, f)
            last_top15 = top15_results
    print(f"Saved top 1 from each generation and top 15 from last generation in {SAVE_DIR}")

if __name__ == "__main__":
    main()
