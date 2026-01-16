"""
Circle Road Traffic Flow Optimization - Final V5
Combines all steps: realistic traffic sim with collision avoidance,
genetic algorithm learning speed/distance, 1 inducer + 9 learners.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CARS = 10  # 1 inducer + 9 learners
CIRCLE_RADIUS = 100.0
CIRCUMFERENCE = 2 * np.pi * CIRCLE_RADIUS
BASE_SPEED = 10.0  # m/s
SPEED_MIN = 0.5  # emergency brake
SPEED_MAX = 1.3  # max multiplier
SIM_DURATION = 180.0
DT = 0.05
SAMPLES_PER_GEN = 100
NUM_GENERATIONS = 10  # Start with 10 to test
SAVE_DIR = "gen_results_v5"
MIN_SAFE_DISTANCE = 5.0  # meters - collision threshold
REACTION_DELAY = 1.0  # 1 second delay for acceleration

# Physics
ACCEL_RATE = 2.0  # m/s^2
DECEL_RATE = 4.0  # m/s^2 (braking is faster)

# Neural network genome size: weights for deciding speed_mult and target_distance
# Inputs: front_speed_ratio, gap_distance_ratio, own_speed_ratio (3 inputs)
# Hidden layer: 6 neurons
# Outputs: speed_mult_delta, target_distance (2 outputs)
INPUT_SIZE = 3
HIDDEN_SIZE = 6
OUTPUT_SIZE = 2
GENOME_SIZE = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

np.random.seed(42)

# =============================================================================
# NEURAL NETWORK
# =============================================================================
def create_random_genome():
    """Create a random genome (neural network weights)."""
    return np.random.randn(GENOME_SIZE) * 0.5

def genome_to_weights(genome):
    """Convert flat genome to weight matrices."""
    idx = 0
    w1 = genome[idx:idx + INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE * HIDDEN_SIZE
    b1 = genome[idx:idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE
    w2 = genome[idx:idx + HIDDEN_SIZE * OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += HIDDEN_SIZE * OUTPUT_SIZE
    b2 = genome[idx:idx + OUTPUT_SIZE]
    return w1, b1, w2, b2

def neural_forward(genome, inputs):
    """Forward pass through neural network."""
    w1, b1, w2, b2 = genome_to_weights(genome)
    # Hidden layer with tanh activation
    hidden = np.tanh(np.dot(inputs, w1) + b1)
    # Output layer with tanh, then scale
    output = np.tanh(np.dot(hidden, w2) + b2)
    # output[0]: speed_mult_delta in [-1, 1] -> maps to [0.5, 1.3]
    # output[1]: target_distance in [-1, 1] -> maps to [5, 30] meters
    speed_mult = 0.9 + output[0] * 0.4  # 0.5 to 1.3
    target_dist = 17.5 + output[1] * 12.5  # 5 to 30 meters
    return speed_mult, target_dist

# =============================================================================
# CAR CLASS
# =============================================================================
class Car:
    def __init__(self, idx, angle, is_inducer=False, genome=None):
        self.idx = idx
        self.angle = angle  # position on circle (radians)
        self.is_inducer = is_inducer
        self.genome = genome
        self.base_speed = BASE_SPEED
        self.speed_mult = np.random.uniform(0.9, 1.1) if is_inducer else 1.0
        self.current_speed = self.base_speed * self.speed_mult
        self.target_speed = self.current_speed
        self.laps = 0.0
        self.distance_traveled = 0.0
        self.last_angle = angle
        self.accel_delay_timer = 0.0  # reaction delay for acceleration
        self.pending_accel = False
        self.collided = False

    def get_position(self):
        """Get (x, y) position on circle."""
        return CIRCLE_RADIUS * np.cos(self.angle), CIRCLE_RADIUS * np.sin(self.angle)

    def arc_distance_to(self, other):
        """Get arc distance to another car (positive = other is ahead)."""
        diff = (other.angle - self.angle) % (2 * np.pi)
        return diff * CIRCLE_RADIUS

    def update(self, dt, front_car=None):
        """Update car state for one timestep."""
        if self.collided:
            return

        if self.is_inducer:
            # Inducer: random speed changes
            if np.random.random() < 0.01:  # 1% chance per step to change speed
                self.speed_mult = np.random.uniform(0.9, 1.1)
            self.target_speed = self.base_speed * self.speed_mult
        else:
            # Learner: use neural network to decide speed
            if front_car is not None and self.genome is not None:
                gap = self.arc_distance_to(front_car)
                if gap < 0:
                    gap += CIRCUMFERENCE
                # Normalize inputs
                front_speed_ratio = front_car.current_speed / self.base_speed
                gap_ratio = gap / 50.0  # normalize to ~1 for 50m gap
                own_speed_ratio = self.current_speed / self.base_speed
                inputs = np.array([front_speed_ratio, gap_ratio, own_speed_ratio])
                
                speed_mult, target_dist = neural_forward(self.genome, inputs)
                
                # Emergency braking if too close
                if gap < MIN_SAFE_DISTANCE * 2:
                    speed_mult = max(0.5, speed_mult * (gap / (MIN_SAFE_DISTANCE * 2)))
                
                # Reaction delay for acceleration only
                if speed_mult > self.speed_mult:
                    if not self.pending_accel:
                        self.pending_accel = True
                        self.accel_delay_timer = REACTION_DELAY
                    if self.accel_delay_timer > 0:
                        self.accel_delay_timer -= dt
                        speed_mult = self.speed_mult  # Can't accelerate yet
                    else:
                        self.pending_accel = False
                else:
                    # Braking is instant (no delay)
                    self.pending_accel = False
                    self.accel_delay_timer = 0
                
                self.speed_mult = np.clip(speed_mult, SPEED_MIN, SPEED_MAX)
            self.target_speed = self.base_speed * self.speed_mult

        # Gradual speed change (realistic acceleration/deceleration)
        if self.current_speed < self.target_speed:
            self.current_speed = min(self.current_speed + ACCEL_RATE * dt, self.target_speed)
        elif self.current_speed > self.target_speed:
            self.current_speed = max(self.current_speed - DECEL_RATE * dt, self.target_speed)

        # Move along circle
        dist = self.current_speed * dt
        dtheta = dist / CIRCLE_RADIUS
        self.angle = (self.angle + dtheta) % (2 * np.pi)
        self.distance_traveled += dist

        # Count laps
        self.laps = self.distance_traveled / CIRCUMFERENCE

# =============================================================================
# SIMULATION
# =============================================================================
class SampleResult:
    def __init__(self, total_laps, learner_laps, car_states, genome, seed, collision_count):
        self.total_laps = total_laps
        self.learner_laps = learner_laps
        self.car_states = car_states
        self.genome = genome
        self.seed = seed
        self.collision_count = collision_count

def run_sample(genome, seed=None, record_frames=False):
    """Run a single simulation sample."""
    rng = np.random.default_rng(seed)
    
    # Initialize cars with equal spacing
    cars = []
    for i in range(NUM_CARS):
        angle = (i / NUM_CARS) * 2 * np.pi
        if i == 0:
            # First car is the inducer
            cars.append(Car(i, angle, is_inducer=True))
        else:
            # Learner cars share the same genome
            cars.append(Car(i, angle, is_inducer=False, genome=genome))
    
    frames = []
    steps = int(SIM_DURATION / DT)
    collision_count = 0
    
    for step in range(steps):
        if record_frames and step % 4 == 0:  # Record every 4th frame
            frames.append([
                (c.angle, c.speed_mult, c.current_speed, c.is_inducer, c.laps, c.collided)
                for c in cars
            ])
        
        # Sort cars by angle to find who's in front of whom
        sorted_cars = sorted(cars, key=lambda c: c.angle)
        
        # Update each car
        for i, car in enumerate(sorted_cars):
            # Find front car (next car in sorted order, wrapping around)
            front_idx = (i + 1) % len(sorted_cars)
            front_car = sorted_cars[front_idx]
            car.update(DT, front_car)
        
        # Check for collisions
        for i, car in enumerate(cars):
            if car.collided:
                continue
            for j, other in enumerate(cars):
                if i == j or other.collided:
                    continue
                gap = car.arc_distance_to(other)
                if gap < 0:
                    gap += CIRCUMFERENCE
                if gap < MIN_SAFE_DISTANCE or (CIRCUMFERENCE - gap) < MIN_SAFE_DISTANCE:
                    # Collision! Penalize but continue
                    collision_count += 1
    
    # Calculate fitness (total laps by learners only)
    learner_laps = sum(c.laps for c in cars if not c.is_inducer)
    total_laps = sum(c.laps for c in cars)
    
    return SampleResult(total_laps, learner_laps, frames, genome, seed, collision_count)

def evaluate_genome(args):
    """Evaluate a genome (for multiprocessing)."""
    genome, seed = args
    result = run_sample(genome, seed, record_frames=False)
    # Fitness = learner laps - collision penalty
    fitness = result.learner_laps - result.collision_count * 0.5
    return fitness, genome

# =============================================================================
# GENETIC ALGORITHM
# =============================================================================
def crossover(parent1, parent2):
    """Uniform crossover."""
    child = np.zeros(GENOME_SIZE)
    for i in range(GENOME_SIZE):
        if np.random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    # Blend some genes
    blend_mask = np.random.random(GENOME_SIZE) < 0.3
    alpha = np.random.random(GENOME_SIZE)
    child[blend_mask] = alpha[blend_mask] * parent1[blend_mask] + (1 - alpha[blend_mask]) * parent2[blend_mask]
    return child

def mutate(genome, rate=0.1, intensity=0.3):
    """Gaussian mutation."""
    child = genome.copy()
    for i in range(GENOME_SIZE):
        if np.random.random() < rate:
            child[i] += np.random.randn() * intensity
    return child

def run_generation(gen_num, population):
    """Run one generation and return sorted results."""
    results = []
    
    # Evaluate all genomes
    args_list = [(g, gen_num * 10000 + i) for i, g in enumerate(population)]
    
    # Use multiprocessing
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as executor:
        futures = [executor.submit(evaluate_genome, args) for args in args_list]
        for future in as_completed(futures):
            fitness, genome = future.result()
            results.append((fitness, genome))
    
    # Sort by fitness (higher is better)
    results.sort(reverse=True, key=lambda x: x[0])
    return results

def evolve_population(results, pop_size):
    """Create next generation from results."""
    # Elite selection: keep top 15%
    elite_count = max(5, pop_size // 7)
    elites = [r[1] for r in results[:elite_count]]
    
    new_pop = list(elites)
    
    # Tournament selection + crossover for the rest
    while len(new_pop) < pop_size:
        # Tournament selection
        tournament_size = 5
        candidates = np.random.choice(len(results), tournament_size, replace=False)
        parent1 = results[min(candidates)][1]
        candidates = np.random.choice(len(results), tournament_size, replace=False)
        parent2 = results[min(candidates)][1]
        
        child = crossover(parent1, parent2)
        child = mutate(child, rate=0.15, intensity=0.3)
        new_pop.append(child)
    
    # Add some fresh random genomes
    fresh_count = max(2, pop_size // 20)
    for i in range(fresh_count):
        new_pop[-(i+1)] = create_random_genome()
    
    return new_pop[:pop_size]

# =============================================================================
# VIDEO GENERATION
# =============================================================================
def save_video(sample_result, filename, gen_num=0):
    """Generate video from sample result."""
    frames = sample_result.car_states
    if not frames:
        print(f"  No frames to save for {filename}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
    ax.set_ylim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    # Draw circle road
    circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='#3d3d5c', fill=False, linewidth=8)
    ax.add_patch(circle)
    
    # Car markers
    car_dots = []
    for i in range(NUM_CARS):
        color = '#ff6b6b' if i == 0 else '#00ff88'  # Red for inducer, green for learners
        dot, = ax.plot([], [], 'o', color=color, markersize=10)
        car_dots.append(dot)
    
    # Text annotations
    title_text = ax.text(0, CIRCLE_RADIUS + 15, '', ha='center', va='bottom', 
                         color='white', fontsize=12, fontweight='bold')
    info_text = ax.text(-CIRCLE_RADIUS - 15, CIRCLE_RADIUS + 10, '', ha='left', va='top',
                        color='#888888', fontsize=9)
    
    images = []
    for frame_idx, frame in enumerate(frames):
        for i, (angle, speed_mult, speed, is_inducer, laps, collided) in enumerate(frame):
            x = CIRCLE_RADIUS * np.cos(angle)
            y = CIRCLE_RADIUS * np.sin(angle)
            car_dots[i].set_data([x], [y])
            if collided:
                car_dots[i].set_color('#ffff00')
        
        t = frame_idx * DT * 4  # Accounting for frame skip
        learner_laps = sum(f[4] for f in frame if not f[3])
        title_text.set_text(f'Gen {gen_num} | t={t:.1f}s | Learner Laps: {learner_laps:.1f}')
        
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        argb = np.frombuffer(renderer.tostring_argb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        argb = argb.reshape((h, w, 4))
        rgb = argb[:, :, 1:4]
        images.append(rgb.copy())
    
    imageio.mimsave(filename, images, fps=30)
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  CIRCLE ROAD TRAFFIC OPTIMIZATION - FINAL V5")
    print("=" * 60)
    print(f"  Cars: {NUM_CARS} (1 inducer + {NUM_CARS-1} learners)")
    print(f"  Circle radius: {CIRCLE_RADIUS}m")
    print(f"  Simulation: {SIM_DURATION}s per sample")
    print(f"  Samples/gen: {SAMPLES_PER_GEN}")
    print(f"  Generations: {NUM_GENERATIONS}")
    print("=" * 60 + "\n")
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Initialize population
    population = [create_random_genome() for _ in range(SAMPLES_PER_GEN)]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"Generation {gen}/{NUM_GENERATIONS}...", end=" ", flush=True)
        
        results = run_generation(gen, population)
        
        best_fitness = results[0][0]
        avg_fitness = np.mean([r[0] for r in results])
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        print(f"Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f}")
        
        # Save best genome and generate video
        best_genome = results[0][1]
        best_result = run_sample(best_genome, gen * 10000, record_frames=True)
        
        vid_name = os.path.join(SAVE_DIR, f"gen{gen:03d}_best.mp4")
        save_video(best_result, vid_name, gen)
        
        with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_best.pkl"), "wb") as f:
            pickle.dump(best_result, f)
        
        # Save top 15 from last generation
        if gen == NUM_GENERATIONS:
            for i in range(min(15, len(results))):
                genome = results[i][1]
                result = run_sample(genome, gen * 10000 + i, record_frames=True)
                with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_top{i+1:02d}.pkl"), "wb") as f:
                    pickle.dump(result, f)
        
        # Evolve population
        population = evolve_population(results, SAMPLES_PER_GEN)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Generation 1 best: {best_fitness_history[0]:.2f}")
    print(f"  Generation {NUM_GENERATIONS} best: {best_fitness_history[-1]:.2f}")
    improvement = best_fitness_history[-1] - best_fitness_history[0]
    print(f"  Improvement: {improvement:+.2f} ({improvement/max(0.01, best_fitness_history[0])*100:+.1f}%)")
    print(f"  Results saved in: {SAVE_DIR}")
    print("=" * 60 + "\n")
    
    # Save fitness history
    with open(os.path.join(SAVE_DIR, "fitness_history.pkl"), "wb") as f:
        pickle.dump({'best': best_fitness_history, 'avg': avg_fitness_history}, f)

if __name__ == "__main__":
    main()
