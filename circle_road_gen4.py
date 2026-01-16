import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import pickle
import sys

# --- Classes for pickle compatibility ---
class Car:
    def __init__(self, idx, angle, speed_mult):
        self.idx = idx
        self.angle = angle
        self.speed_mult = speed_mult
        self.base_speed = 8.0
        self.laps = 0
        self.last_angle = angle

    def update(self, dt):
        speed = self.base_speed * self.speed_mult
        dtheta = speed * dt / 20.0
        self.angle = (self.angle + dtheta) % (2 * np.pi)
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

NUM_CARS = 10
CIRCLE_RADIUS = 100.0
SAVE_DIR = "gen_results"

# --- Video rendering function (same as before) ---
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

# --- Main logic ---
def main():
    print("\n=== Video Generator for Circle Road Top Samples ===")
    print(f"Looking in folder: {SAVE_DIR}\n")
    # List all available .pkl files
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.pkl')]
    if not files:
        print("No saved samples found. Run the main simulation first.")
        return
    files.sort()
    print("Available samples:")
    for i, fname in enumerate(files):
        print(f"  [{i}] {fname}")
    print("\nEnter the number(s) of the sample(s) to generate video for (comma or space separated):")
    sel = input().replace(',', ' ').split()
    try:
        indices = [int(s) for s in sel]
    except Exception:
        print("Invalid input.")
        return
    for idx in indices:
        if idx < 0 or idx >= len(files):
            print(f"Index {idx} out of range.")
            continue
        pkl_path = os.path.join(SAVE_DIR, files[idx])
        with open(pkl_path, 'rb') as f:
            sample = pickle.load(f)
        outname = files[idx].replace('.pkl', '.mp4')
        outpath = os.path.join(SAVE_DIR, outname)
        print(f"Generating video: {outpath}")
        save_video(sample, outpath)
        print(f"  Done: {outpath}")
    print("\nAll requested videos generated.")

if __name__ == "__main__":
    main()
