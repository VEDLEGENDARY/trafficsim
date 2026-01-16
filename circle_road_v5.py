"""
Circle Road Traffic Flow Optimization - Final V5 (Numba + Multiprocessing)
Uses Numba JIT compilation for speed, simulates realistic traffic with reaction delays.
The inducer fluctuates 0.9x-1.2x, learners must learn distance/speed to prevent cascading slowdowns.
"""
import numpy as np
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
import multiprocessing as mp
import time as timer

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CARS = 10  # 1 inducer + 9 learners
CIRCLE_RADIUS = 100.0
CIRCUMFERENCE = 2 * np.pi * CIRCLE_RADIUS
BASE_SPEED = 10.0  # m/s (~36 km/h)
SIM_DURATION = 180.0
DT = 0.05
SAMPLES_PER_GEN = 100
NUM_GENERATIONS = 50  # Full run
SAVE_DIR = "gen_results_v5"
MIN_SAFE_DISTANCE = 4.0  # meters
MAX_WORKERS = min(8, max(1, os.cpu_count() - 2))  # Cap at 8 to prevent memory issues

# Physics
ACCEL_RATE = 2.5  # m/s^2
DECEL_RATE = 4.0  # m/s^2
REACTION_DELAY = 1.0  # 1 second delay before can accelerate after front car speeds up

# Neural network: simple linear weights for speed/distance decisions
# Inputs: gap_ratio, front_speed_ratio, own_speed_ratio (3)
# Outputs: target_speed_mult, preferred_gap (2)
INPUT_SIZE = 3
OUTPUT_SIZE = 2
GENOME_SIZE = 8  # 6 weights + 2 biases

# Car array indices (for Numba)
C_ANGLE = 0
C_SPEED = 1
C_TARGET_SPEED = 2
C_SPEED_MULT = 3
C_DISTANCE = 4  # total distance traveled
C_IS_INDUCER = 5
C_ACCEL_TIMER = 6  # reaction delay timer
C_PENDING_ACCEL = 7
CAR_FIELDS = 8

np.random.seed(42)

# =============================================================================
# NUMBA JIT-COMPILED SIMULATION CORE
# =============================================================================

@njit(cache=True)
def neural_forward(genome, gap_ratio, front_speed_ratio, own_speed_ratio):
    """Simple linear neural network without np.dot (Numba compatible)."""
    # Manual matrix multiplication: 3 inputs -> 2 outputs
    # Weights: 6 values (3x2), Biases: 2 values
    w00, w01 = genome[0], genome[1]
    w10, w11 = genome[2], genome[3]
    w20, w21 = genome[4], genome[5]
    b0, b1 = genome[6], genome[7]
    
    # Output 0: speed_mult_raw
    raw0 = gap_ratio * w00 + front_speed_ratio * w10 + own_speed_ratio * w20 + b0
    # Output 1: pref_gap_raw
    raw1 = gap_ratio * w01 + front_speed_ratio * w11 + own_speed_ratio * w21 + b1
    
    # Tanh activation and scale
    out0 = np.tanh(raw0)
    out1 = np.tanh(raw1)
    
    # out[0]: speed_mult in [0.9, 1.1]
    # out[1]: preferred_gap in [8, 40] meters
    speed_mult = 1.0 + out0 * 0.1  # 0.9 to 1.1
    pref_gap = 24.0 + out1 * 16.0  # 8 to 40 meters
    return speed_mult, pref_gap


@njit(cache=True)
def run_simulation(genome, seed):
    """Run a complete simulation and return (total_learner_laps, collision_count)."""
    np.random.seed(seed)
    
    # Initialize cars array
    cars = np.zeros((NUM_CARS, CAR_FIELDS), dtype=np.float64)
    
    # Equal spacing around circle
    for i in range(NUM_CARS):
        cars[i, C_ANGLE] = (i / NUM_CARS) * 2 * np.pi
        cars[i, C_SPEED] = BASE_SPEED
        cars[i, C_TARGET_SPEED] = BASE_SPEED
        cars[i, C_SPEED_MULT] = 1.0
        cars[i, C_DISTANCE] = 0.0
        cars[i, C_IS_INDUCER] = 1.0 if i == 0 else 0.0
        cars[i, C_ACCEL_TIMER] = 0.0
        cars[i, C_PENDING_ACCEL] = 0.0
    
    collision_count = 0
    steps = int(SIM_DURATION / DT)
    inducer_change_timer = 0.0
    inducer_target_mult = 1.0
    
    for step in range(steps):
        # === INDUCER BEHAVIOR ===
        inducer_change_timer += DT
        if inducer_change_timer > 1.5 + np.random.random() * 2.0:  # Change every 1.5-3.5 seconds (more frequent)
            inducer_change_timer = 0.0
            inducer_target_mult = 0.85 + np.random.random() * 0.3  # 0.85 to 1.15 (wider range)
        
        # More aggressive speed transitions (0.2 instead of 0.1)
        cars[0, C_SPEED_MULT] += (inducer_target_mult - cars[0, C_SPEED_MULT]) * 0.2
        cars[0, C_TARGET_SPEED] = BASE_SPEED * cars[0, C_SPEED_MULT]
        
        # === FIND FRONT CAR FOR EACH CAR ===
        # Sort by angle to determine order
        angles = cars[:, C_ANGLE].copy()
        order = np.argsort(angles)
        
        # === UPDATE EACH CAR ===
        for idx in range(NUM_CARS):
            i = order[idx]
            
            # Find front car (next in circular order)
            front_idx = order[(idx + 1) % NUM_CARS]
            
            # Calculate gap (arc distance to front car)
            angle_diff = (cars[front_idx, C_ANGLE] - cars[i, C_ANGLE]) % (2 * np.pi)
            gap = angle_diff * CIRCLE_RADIUS
            if gap < 0.1:
                gap = CIRCUMFERENCE  # Full circle if same position
            
            # === LEARNER DECISION ===
            if cars[i, C_IS_INDUCER] < 0.5:  # Learner
                # Normalize inputs
                gap_ratio = gap / 50.0  # Normalize to ~1 for 50m gap
                front_speed_ratio = cars[front_idx, C_SPEED] / BASE_SPEED
                own_speed_ratio = cars[i, C_SPEED] / BASE_SPEED
                
                target_mult, pref_gap = neural_forward(genome, gap_ratio, front_speed_ratio, own_speed_ratio)
                
                # === COLLISION AVOIDANCE ===
                if gap < MIN_SAFE_DISTANCE * 2:
                    # Emergency brake - linear reduction based on gap
                    target_mult = max(0.5, target_mult * (gap / (MIN_SAFE_DISTANCE * 2)))
                
                # If gap is smaller than preferred, slow down
                if gap < pref_gap:
                    gap_factor = gap / pref_gap
                    target_mult = min(target_mult, front_speed_ratio * gap_factor + 0.1)
                
                # === REACTION DELAY (only for speeding up) ===
                if target_mult > cars[i, C_SPEED_MULT]:
                    if cars[i, C_PENDING_ACCEL] < 0.5:
                        cars[i, C_PENDING_ACCEL] = 1.0
                        cars[i, C_ACCEL_TIMER] = REACTION_DELAY
                    
                    if cars[i, C_ACCEL_TIMER] > 0:
                        cars[i, C_ACCEL_TIMER] -= DT
                        target_mult = cars[i, C_SPEED_MULT]  # Can't accelerate yet
                    else:
                        cars[i, C_PENDING_ACCEL] = 0.0
                else:
                    # Braking is instant
                    cars[i, C_PENDING_ACCEL] = 0.0
                    cars[i, C_ACCEL_TIMER] = 0.0
                
                # Clamp speed multiplier to 0.9-1.1 range
                cars[i, C_SPEED_MULT] = max(0.9, min(1.1, target_mult))
            
            # === APPLY SPEED ===
            cars[i, C_TARGET_SPEED] = BASE_SPEED * cars[i, C_SPEED_MULT]
            
            # Gradual acceleration/deceleration
            if cars[i, C_SPEED] < cars[i, C_TARGET_SPEED]:
                cars[i, C_SPEED] = min(cars[i, C_SPEED] + ACCEL_RATE * DT, cars[i, C_TARGET_SPEED])
            else:
                cars[i, C_SPEED] = max(cars[i, C_SPEED] - DECEL_RATE * DT, cars[i, C_TARGET_SPEED])
            
            cars[i, C_SPEED] = max(0.0, cars[i, C_SPEED])
            
            # === MOVE ===
            dist = cars[i, C_SPEED] * DT
            dtheta = dist / CIRCLE_RADIUS
            cars[i, C_ANGLE] = (cars[i, C_ANGLE] + dtheta) % (2 * np.pi)
            cars[i, C_DISTANCE] += dist
        
        # === CHECK COLLISIONS ===
        for i in range(NUM_CARS):
            for j in range(i + 1, NUM_CARS):
                angle_diff = abs(cars[i, C_ANGLE] - cars[j, C_ANGLE])
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                arc_dist = angle_diff * CIRCLE_RADIUS
                if arc_dist < MIN_SAFE_DISTANCE:
                    collision_count += 1
    
    # Calculate total laps by learners
    learner_laps = 0.0
    for i in range(NUM_CARS):
        if cars[i, C_IS_INDUCER] < 0.5:
            learner_laps += cars[i, C_DISTANCE] / CIRCUMFERENCE
    
    return learner_laps, collision_count


def create_random_genome():
    """Create random genome."""
    return np.random.randn(GENOME_SIZE) * 0.5


def evaluate_genome(args):
    """Evaluate a single genome (for multiprocessing)."""
    genome, base_seed = args
    total_laps = 0.0
    total_collisions = 0
    n_runs = 3  # Average over 3 runs
    
    for i in range(n_runs):
        laps, collisions = run_simulation(genome, base_seed + i * 1000)
        total_laps += laps
        total_collisions += collisions
    
    avg_laps = total_laps / n_runs
    avg_collisions = total_collisions / n_runs
    
    # Fitness = laps - collision penalty
    fitness = avg_laps - avg_collisions * 2.0
    return fitness, genome, avg_laps, avg_collisions


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

def crossover(p1, p2):
    """BLX-alpha crossover."""
    child = np.zeros(GENOME_SIZE)
    for i in range(GENOME_SIZE):
        if np.random.random() < 0.5:
            alpha = np.random.uniform(-0.1, 1.1)
            child[i] = alpha * p1[i] + (1 - alpha) * p2[i]
        else:
            child[i] = p1[i] if np.random.random() < 0.5 else p2[i]
    return child


def mutate(genome, rate=0.2, intensity=0.3):
    """Gaussian mutation."""
    child = genome.copy()
    for i in range(GENOME_SIZE):
        if np.random.random() < rate:
            child[i] += np.random.randn() * intensity
    return child


def run_generation(gen_num, population):
    """Run one generation with multiprocessing."""
    args_list = [(g, gen_num * 10000 + i) for i, g in enumerate(population)]
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_genome, args) for args in args_list]
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by fitness (higher is better)
    results.sort(reverse=True, key=lambda x: x[0])
    return results


def evolve_population(results, pop_size):
    """Create next generation."""
    elite_count = max(5, pop_size // 7)
    elites = [r[1] for r in results[:elite_count]]
    
    new_pop = list(elites)
    
    # Tournament selection + crossover
    while len(new_pop) < pop_size:
        t_size = 5
        candidates = random.sample(range(len(results)), t_size)
        p1 = results[min(candidates)][1]  # min index = best fitness (sorted descending)
        candidates = random.sample(range(len(results)), t_size)
        p2 = results[min(candidates)][1]  # min index = best fitness (sorted descending)
        
        child = crossover(p1, p2)
        child = mutate(child, rate=0.25, intensity=0.4)  # Higher mutation for exploration
        new_pop.append(child)
    
    # Fresh randoms
    for i in range(max(2, pop_size // 20)):
        new_pop[-(i + 1)] = create_random_genome()
    
    return new_pop[:pop_size]


# =============================================================================
# SAMPLE RESULT CLASS (for pickle)
# =============================================================================

class SampleResult:
    def __init__(self, total_laps, learner_laps, car_states, genome, seed, collision_count):
        self.total_laps = total_laps
        self.learner_laps = learner_laps
        self.car_states = car_states
        self.genome = genome
        self.seed = seed
        self.collision_count = collision_count


def run_sample_with_recording(genome, seed):
    """Run simulation with frame recording for video."""
    np.random.seed(seed)
    
    cars = np.zeros((NUM_CARS, CAR_FIELDS), dtype=np.float64)
    for i in range(NUM_CARS):
        cars[i, C_ANGLE] = (i / NUM_CARS) * 2 * np.pi
        cars[i, C_SPEED] = BASE_SPEED
        cars[i, C_TARGET_SPEED] = BASE_SPEED
        cars[i, C_SPEED_MULT] = 1.0
        cars[i, C_DISTANCE] = 0.0
        cars[i, C_IS_INDUCER] = 1.0 if i == 0 else 0.0
        cars[i, C_ACCEL_TIMER] = 0.0
        cars[i, C_PENDING_ACCEL] = 0.0
    
    frames = []
    collision_count = 0
    steps = int(SIM_DURATION / DT)
    inducer_change_timer = 0.0
    inducer_target_mult = 1.0
    
    for step in range(steps):
        # Record frame every 4 steps
        if step % 4 == 0:
            frame = []
            for i in range(NUM_CARS):
                frame.append((
                    cars[i, C_ANGLE],
                    cars[i, C_SPEED_MULT],
                    cars[i, C_SPEED],
                    cars[i, C_IS_INDUCER] > 0.5,
                    cars[i, C_DISTANCE] / CIRCUMFERENCE,
                    False  # collided flag (simplified)
                ))
            frames.append(frame)
        
        # Same simulation logic as run_simulation
        inducer_change_timer += DT
        if inducer_change_timer > 2.0 + np.random.random() * 3.0:
            inducer_change_timer = 0.0
            inducer_target_mult = 0.9 + np.random.random() * 0.3
        
        cars[0, C_SPEED_MULT] += (inducer_target_mult - cars[0, C_SPEED_MULT]) * 0.1
        cars[0, C_TARGET_SPEED] = BASE_SPEED * cars[0, C_SPEED_MULT]
        
        angles = cars[:, C_ANGLE].copy()
        order = np.argsort(angles)
        
        for idx in range(NUM_CARS):
            i = order[idx]
            front_idx = order[(idx + 1) % NUM_CARS]
            
            angle_diff = (cars[front_idx, C_ANGLE] - cars[i, C_ANGLE]) % (2 * np.pi)
            gap = angle_diff * CIRCLE_RADIUS
            if gap < 0.1:
                gap = CIRCUMFERENCE
            
            if cars[i, C_IS_INDUCER] < 0.5:
                gap_ratio = gap / 50.0
                front_speed_ratio = cars[front_idx, C_SPEED] / BASE_SPEED
                own_speed_ratio = cars[i, C_SPEED] / BASE_SPEED
                
                # Manual neural forward (same as JIT version)
                w00, w01 = genome[0], genome[1]
                w10, w11 = genome[2], genome[3]
                w20, w21 = genome[4], genome[5]
                b0, b1 = genome[6], genome[7]
                
                raw0 = gap_ratio * w00 + front_speed_ratio * w10 + own_speed_ratio * w20 + b0
                raw1 = gap_ratio * w01 + front_speed_ratio * w11 + own_speed_ratio * w21 + b1
                
                out0 = np.tanh(raw0)
                out1 = np.tanh(raw1)
                
                target_mult = 0.95 + out0 * 0.25
                pref_gap = 24.0 + out1 * 16.0
                
                if gap < MIN_SAFE_DISTANCE * 2:
                    target_mult = max(0.5, target_mult * (gap / (MIN_SAFE_DISTANCE * 2)))
                
                if gap < pref_gap:
                    gap_factor = gap / pref_gap
                    target_mult = min(target_mult, front_speed_ratio * gap_factor + 0.1)
                
                if target_mult > cars[i, C_SPEED_MULT]:
                    if cars[i, C_PENDING_ACCEL] < 0.5:
                        cars[i, C_PENDING_ACCEL] = 1.0
                        cars[i, C_ACCEL_TIMER] = REACTION_DELAY
                    
                    if cars[i, C_ACCEL_TIMER] > 0:
                        cars[i, C_ACCEL_TIMER] -= DT
                        target_mult = cars[i, C_SPEED_MULT]
                    else:
                        cars[i, C_PENDING_ACCEL] = 0.0
                else:
                    cars[i, C_PENDING_ACCEL] = 0.0
                    cars[i, C_ACCEL_TIMER] = 0.0
                
                cars[i, C_SPEED_MULT] = max(0.5, min(1.3, target_mult))
            
            cars[i, C_TARGET_SPEED] = BASE_SPEED * cars[i, C_SPEED_MULT]
            
            if cars[i, C_SPEED] < cars[i, C_TARGET_SPEED]:
                cars[i, C_SPEED] = min(cars[i, C_SPEED] + ACCEL_RATE * DT, cars[i, C_TARGET_SPEED])
            else:
                cars[i, C_SPEED] = max(cars[i, C_SPEED] - DECEL_RATE * DT, cars[i, C_TARGET_SPEED])
            
            cars[i, C_SPEED] = max(0.0, cars[i, C_SPEED])
            
            dist = cars[i, C_SPEED] * DT
            dtheta = dist / CIRCLE_RADIUS
            cars[i, C_ANGLE] = (cars[i, C_ANGLE] + dtheta) % (2 * np.pi)
            cars[i, C_DISTANCE] += dist
    
    learner_laps = sum(cars[i, C_DISTANCE] / CIRCUMFERENCE for i in range(NUM_CARS) if cars[i, C_IS_INDUCER] < 0.5)
    total_laps = sum(cars[i, C_DISTANCE] / CIRCUMFERENCE for i in range(NUM_CARS))
    
    return SampleResult(total_laps, learner_laps, frames, genome, seed, collision_count)


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def save_video(sample_result, filename, gen_num=0):
    """Generate video from sample result."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    
    frames = sample_result.car_states
    if not frames:
        print(f"  No frames for {filename}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    images = []
    for frame_idx, frame in enumerate(frames[::2]):  # Skip every other frame
        ax.clear()
        ax.set_xlim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
        ax.set_ylim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw circle road
        circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='#3d3d5c', fill=False, linewidth=8)
        ax.add_patch(circle)
        
        # Draw cars
        for i, (angle, speed_mult, speed, is_inducer, laps, collided) in enumerate(frame):
            x = CIRCLE_RADIUS * np.cos(angle)
            y = CIRCLE_RADIUS * np.sin(angle)
            color = '#ff6b6b' if is_inducer else '#00ff88'
            if speed < BASE_SPEED * 0.7:
                color = '#ffaa00'  # Yellow if slow
            ax.plot(x, y, 'o', color=color, markersize=12)
        
        # Info text
        t = frame_idx * 2 * DT * 4
        learner_laps = sum(f[4] for f in frame if not f[3])
        ax.text(0, CIRCLE_RADIUS + 12, f'Gen {gen_num} | t={t:.1f}s | Learner Laps: {learner_laps:.1f}',
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold',
                transform=ax.transData)
        
        fig.patch.set_facecolor('#1a1a2e')
        fig.canvas.draw()
        
        # Get image from canvas
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        images.append(img)
    
    imageio.mimsave(filename, images, fps=30)
    plt.close(fig)
    print(f"    Saved: {filename}")


# =============================================================================
# MAIN
# =============================================================================

def load_latest_generation():
    """Load the latest generation from saved pkl files, or return None if starting fresh."""
    if not os.path.exists(SAVE_DIR):
        return None, 0, [], []
    
    # Find all gen*_population.pkl files
    pkl_files = [f for f in os.listdir(SAVE_DIR) if f.startswith('gen') and f.endswith('_population.pkl')]
    if not pkl_files:
        return None, 0, [], []
    
    # Extract generation numbers and find the latest
    gen_numbers = []
    for f in pkl_files:
        try:
            gen_num = int(f[3:6])  # Extract gen number from "genXXX_population.pkl"
            gen_numbers.append(gen_num)
        except:
            continue
    
    if not gen_numbers:
        return None, 0, [], []
    
    latest_gen = max(gen_numbers)
    
    # Load the full population data
    pkl_path = os.path.join(SAVE_DIR, f'gen{latest_gen:03d}_population.pkl')
    try:
        with open(pkl_path, 'rb') as f:
            pop_data = pickle.load(f)
            population = pop_data['population']
            best_fitness_history = pop_data.get('best_fitness_history', [])
            best_laps_history = pop_data.get('best_laps_history', [])
            print(f"  Loaded generation {latest_gen} with {len(population)} genomes")
            return population, latest_gen, best_fitness_history, best_laps_history
    except Exception as e:
        print(f"  Warning: Could not load population file: {e}")
        return None, 0, [], []
    
    return None, 0, [], []
    
    print(f"  Resuming from generation {latest_gen}")
    print(f"  Loaded {len([g for g in population if g is not None])} genomes from previous run")
    
    return population, latest_gen, best_fitness_history, best_laps_history


def main():
    print("\n" + "=" * 60)
    print("  CIRCLE ROAD TRAFFIC OPTIMIZATION - V5 (Numba)")
    print("=" * 60)
    print(f"  Cars: {NUM_CARS} (1 inducer + {NUM_CARS-1} learners)")
    print(f"  Circle radius: {CIRCLE_RADIUS}m | Duration: {SIM_DURATION}s")
    print(f"  Samples/gen: {SAMPLES_PER_GEN} | Workers: {MAX_WORKERS}")
    print(f"  Generations: {NUM_GENERATIONS}")
    print("=" * 60)
    
    # Warm up JIT
    print("  JIT compiling...", end=" ", flush=True)
    _ = run_simulation(create_random_genome(), 0)
    print("done!")
    print("=" * 60 + "\n")
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Try to load from previous run
    population, start_gen, best_fitness_history, best_laps_history = load_latest_generation()
    if population is None:
        population = [create_random_genome() for _ in range(SAMPLES_PER_GEN)]
        start_gen = 0
        best_fitness_history = []
        best_laps_history = []
    
    for gen in range(start_gen + 1, start_gen + NUM_GENERATIONS + 1):
        t0 = timer.time()
        
        results = run_generation(gen, population)
        
        best_fitness, best_genome, best_laps, best_collisions = results[0]
        avg_fitness = np.mean([r[0] for r in results])
        avg_laps = np.mean([r[2] for r in results])
        
        best_fitness_history.append(best_fitness)
        best_laps_history.append(best_laps)
        
        dt = timer.time() - t0
        
        print(f"  Gen {gen:3d}/{start_gen + NUM_GENERATIONS} | {dt:4.1f}s | Best: {best_laps:.1f} laps ({best_collisions:.0f} col) | Avg: {avg_laps:.1f} laps")
        
        # Delete old pkl files from previous generation to save space
        if gen > 1:
            old_gen = gen - 1
            # Delete all files from previous gen except if it's a milestone (every 10)
            if old_gen % 10 != 0:
                for old_file in os.listdir(SAVE_DIR):
                    if old_file.startswith(f'gen{old_gen:03d}') and old_file.endswith('.pkl'):
                        try:
                            os.remove(os.path.join(SAVE_DIR, old_file))
                        except:
                            pass
            else:
                # Even for milestones, delete population pkl to save space (keep only best)
                pop_file = os.path.join(SAVE_DIR, f'gen{old_gen:03d}_population.pkl')
                if os.path.exists(pop_file):
                    try:
                        os.remove(pop_file)
                    except:
                        pass
        
        # Save best
        best_result = run_sample_with_recording(best_genome, gen * 10000)
        with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_best.pkl"), "wb") as f:
            pickle.dump(best_result, f)
        
        # Save entire population for proper resumption (genomes only, not results)
        pop_data = {
            'generation': gen,
            'population': [r[1] for r in results],  # Just the genomes, not full results
            'best_fitness_history': best_fitness_history,
            'best_laps_history': best_laps_history
        }
        with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_population.pkl"), "wb") as f:
            pickle.dump(pop_data, f)
        
        # Save video every 10 generations or last gen
        if gen % 10 == 0 or gen == start_gen + NUM_GENERATIONS:
            vid_name = os.path.join(SAVE_DIR, f"gen{gen:03d}_best.mp4")
            save_video(best_result, vid_name, gen)
        
        # Save top 15 from last gen
        if gen == start_gen + NUM_GENERATIONS:
            for i in range(min(15, len(results))):
                genome = results[i][1]
                result = run_sample_with_recording(genome, gen * 10000 + i)
                with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_top{i+1:02d}.pkl"), "wb") as f:
                    pickle.dump(result, f)
        
        # Evolve
        population = evolve_population(results, SAMPLES_PER_GEN)
    
    # Summary
    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)
    if len(best_laps_history) > 0:
        print(f"  Gen {start_gen + 1} best laps: {best_laps_history[0]:.1f}")
        print(f"  Gen {start_gen + NUM_GENERATIONS} best laps: {best_laps_history[-1]:.1f}")
        improvement = best_laps_history[-1] - best_laps_history[0]
        print(f"  Improvement: {improvement:+.1f} laps")
    print(f"  Results saved in: {SAVE_DIR}")
    print("=" * 60 + "\n")
    
    with open(os.path.join(SAVE_DIR, "fitness_history.pkl"), "wb") as f:
        pickle.dump({'best_fitness': best_fitness_history, 'best_laps': best_laps_history}, f)


if __name__ == "__main__":
    mp.freeze_support()
    main()
