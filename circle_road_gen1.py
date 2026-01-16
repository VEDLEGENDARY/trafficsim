import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio

# Simulation parameters
NUM_CARS = 10
CIRCLE_RADIUS = 100.0
BASE_SPEED = 8.0
SPEED_MIN = 0.9
SPEED_MAX = 1.1
SIM_DURATION = 180.0
DT = 0.05
SAMPLES_PER_GEN = 100

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
    def __init__(self, laps, car_states):
        self.laps = laps  # total laps by all cars
        self.car_states = car_states  # list of (angle, speed_mult) for each car at each frame

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
    return SampleResult(total_laps, frames)

def run_generation():
    results = []
    for i in range(SAMPLES_PER_GEN):
        res = run_sample(seed=i, record_frames=False)
        results.append((res.laps, i))
    # Find top 1 sample
    results.sort(reverse=True)
    best_idx = results[0][1]
    best_result = run_sample(seed=best_idx, record_frames=True)
    return best_result, best_idx, results[0][0]

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
    for frame in frames[::8]:  # Downsample for video speed
        for i, (angle, speed_mult) in enumerate(frame):
            x = CIRCLE_RADIUS * np.cos(angle)
            y = CIRCLE_RADIUS * np.sin(angle)
            car_dots[i].set_data([x], [y])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        argb = np.frombuffer(renderer.tostring_argb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        argb = argb.reshape((h, w, 4))
        # Convert ARGB to RGB by dropping alpha channel and reordering
        rgb = argb[:, :, 1:4]
        images.append(rgb)
    imageio.mimsave(filename, images, fps=30)
    plt.close(fig)

def main():
    print("Running 100 samples for 1 generation...")
    best_result, best_idx, best_laps = run_generation()
    print(f"Best sample: seed={best_idx}, total laps={best_laps:.2f}")
    outname = f"gen1_top1.mp4"
    save_video(best_result, outname)
    print(f"Saved video: {outname}")

if __name__ == "__main__":
    main()
