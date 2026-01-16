import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
NUM_CARS = 10
CIRCLE_RADIUS = 20.0  # meters
BASE_SPEED = 8.0      # m/s (about 29 km/h)
SPEED_MIN = 0.9
SPEED_MAX = 1.1
SIM_DURATION = 30.0   # seconds (short for demo)
DT = 0.05             # timestep (seconds)

np.random.seed(42)

class Car:
    def __init__(self, idx, angle, speed_mult):
        self.idx = idx
        self.angle = angle  # radians
        self.speed_mult = speed_mult
        self.base_speed = BASE_SPEED
        self.color = plt.cm.hsv(idx / NUM_CARS)

    def update(self, dt):
        # Move car along the circle
        speed = self.base_speed * self.speed_mult
        dtheta = speed * dt / CIRCLE_RADIUS
        self.angle = (self.angle + dtheta) % (2 * np.pi)

# Initialize cars with random positions and speed multipliers
cars = [
    Car(idx=i,
        angle=np.random.uniform(0, 2 * np.pi),
        speed_mult=np.random.uniform(SPEED_MIN, SPEED_MAX))
    for i in range(NUM_CARS)
]

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-CIRCLE_RADIUS-5, CIRCLE_RADIUS+5)
ax.set_ylim(-CIRCLE_RADIUS-5, CIRCLE_RADIUS+5)
ax.axis('off')

# Draw the circle road
circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='gray', fill=False, linewidth=3)
ax.add_patch(circle)

car_dots = [ax.plot([], [], 'o', color=car.color, markersize=12)[0] for car in cars]

# Animation update function
def animate(frame):
    t = frame * DT
    for car, dot in zip(cars, car_dots):
        car.update(DT)
        x = CIRCLE_RADIUS * np.cos(car.angle)
        y = CIRCLE_RADIUS * np.sin(car.angle)
        dot.set_data([x], [y])  # set_data expects sequences
    return car_dots

frames = int(SIM_DURATION / DT)
ani = animation.FuncAnimation(fig, animate, frames=frames, interval=DT*1000, blit=True)
plt.title('Circle Road Traffic Flow (Random Speeds)')
plt.show()
