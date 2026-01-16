"""
Circle Road Traffic Flow Optimization - Simplified V6
Learns to maintain smooth speed and safe following distance.
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
CAR_SIZE_RATIO = 1.0  # Ratio to scale car size (1.0 = default, <1 smaller, >1 larger)
# =============================================================================
NUM_CARS = 15
CIRCLE_RADIUS = 50.0
CIRCUMFERENCE = 2 * np.pi * CIRCLE_RADIUS
BASE_SPEED = 10.0  # m/s
SIM_DURATION = 300.0
DT = 0.05
SAMPLES_PER_GEN = 20
NUM_GENERATIONS = 50
SAVE_DIR = "gen_results_v6"
MAX_WORKERS = min(8, max(1, os.cpu_count() - 2))

# Neural network: 3 inputs -> 1 output (just target speed)
GENOME_SIZE = 4  # 3 weights + 1 bias

# Car state indices
C_ANGLE = 0
C_SPEED = 1
C_DISTANCE = 2
C_IS_INDUCER = 3
C_PREV_SPEED = 4
CAR_FIELDS = 5

np.random.seed(42)

# =============================================================================
# NUMBA JIT-COMPILED SIMULATION
# =============================================================================

@njit(cache=True)
def neural_forward(genome, gap_distance, front_speed, own_speed):
    """Neural network: decides target speed based on gap and front car speed."""
    # Normalize inputs to [0, 1] range
    gap_norm = min(gap_distance / 40.0, 1.5)  # Normalize gap
    front_norm = front_speed / (BASE_SPEED * 1.5)  # Front car speed
    own_norm = own_speed / (BASE_SPEED * 1.5)  # Own speed
    
    # Weights
    w0, w1, w2, b = genome[0], genome[1], genome[2], genome[3]
    
    # Linear + sigmoid for smooth output
    raw = gap_norm * w0 + front_norm * w1 + own_norm * w2 + b
    
    # Sigmoid to [0, 1], then scale to [0.3, 1.3] * BASE_SPEED
    sigmoid = 1.0 / (1.0 + np.exp(-raw))
    speed_mult = 0.3 + sigmoid * 1.0  # 0.3 to 1.3
    
    return speed_mult * BASE_SPEED


@njit(cache=True)
def run_simulation(genome, seed):
    """Run simulation and return fitness metrics."""
    np.random.seed(seed)
    
    # Initialize cars
    cars = np.zeros((NUM_CARS, CAR_FIELDS), dtype=np.float64)
    for i in range(NUM_CARS):
        cars[i, C_ANGLE] = (i / NUM_CARS) * 2 * np.pi
        cars[i, C_SPEED] = BASE_SPEED
        cars[i, C_DISTANCE] = 0.0
        cars[i, C_IS_INDUCER] = 1.0 if i == 0 else 0.0
        cars[i, C_PREV_SPEED] = BASE_SPEED
    
    steps = int(SIM_DURATION / DT)
    
    # Inducer behavior
    inducer_timer = 0.0
    inducer_target = 1.0
    
    # Metrics
    collision_count = 0
    speed_change_sum = 0.0  # Total speed changes (jerking)
    too_slow_count = 0  # Count of times going < 80% base speed
    total_speed_ratio = 0.0  # Track average speed relative to base
    sample_count = 0
    
    for step in range(steps):
        # Inducer fluctuates speed MORE to create challenge
        inducer_timer += DT
        if inducer_timer > 2.0:  # Change every 2 seconds
            inducer_timer = 0.0
            inducer_target = 0.6 + np.random.random() * 0.8  # 0.6 to 1.4 multiplier
        
        # Smooth transition
        cars[0, C_SPEED] += (inducer_target * BASE_SPEED - cars[0, C_SPEED]) * 0.15
        
        # Sort cars by angle
        angles = cars[:, C_ANGLE].copy()
        order = np.argsort(angles)
        
        # Update each car
        for idx in range(NUM_CARS):
            i = order[idx]
            front_idx = order[(idx + 1) % NUM_CARS]
            
            # Calculate gap to front car
            angle_diff = (cars[front_idx, C_ANGLE] - cars[i, C_ANGLE]) % (2 * np.pi)
            gap = angle_diff * CIRCLE_RADIUS
            if gap < 0.1:
                gap = CIRCUMFERENCE
            
            # Learner uses neural network
            if cars[i, C_IS_INDUCER] < 0.5:
                target_speed = neural_forward(
                    genome,
                    gap,
                    cars[front_idx, C_SPEED],
                    cars[i, C_SPEED]
                )
                
                # Smoothly adjust speed (prevent instant changes)
                speed_diff = target_speed - cars[i, C_SPEED]
                cars[i, C_SPEED] += speed_diff * 0.3  # Faster adjustment for responsiveness
                
                # Clamp to reasonable range
                cars[i, C_SPEED] = max(0.5, min(cars[i, C_SPEED], BASE_SPEED * 1.5))
                
                # Track metrics
                speed_change = abs(cars[i, C_SPEED] - cars[i, C_PREV_SPEED])
                speed_change_sum += speed_change
                
                # Count slow driving
                if cars[i, C_SPEED] < BASE_SPEED * 0.8:
                    too_slow_count += 1
                
                # Track average speed ratio
                total_speed_ratio += cars[i, C_SPEED] / BASE_SPEED
                sample_count += 1
                
                cars[i, C_PREV_SPEED] = cars[i, C_SPEED]
            
            # Move car
            dist = cars[i, C_SPEED] * DT
            dtheta = dist / CIRCLE_RADIUS
            cars[i, C_ANGLE] = (cars[i, C_ANGLE] + dtheta) % (2 * np.pi)
            cars[i, C_DISTANCE] += dist
        
        # Check collisions (< 3m is collision)
        for i in range(NUM_CARS):
            for j in range(i + 1, NUM_CARS):
                angle_diff = abs(cars[i, C_ANGLE] - cars[j, C_ANGLE])
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                arc_dist = angle_diff * CIRCLE_RADIUS
                if arc_dist < 3.0:
                    collision_count += 1
    
    # Calculate metrics
    total_distance = 0.0
    for i in range(NUM_CARS):
        if cars[i, C_IS_INDUCER] < 0.5:
            total_distance += cars[i, C_DISTANCE]
    
    learner_laps = total_distance / CIRCUMFERENCE / (NUM_CARS - 1)
    avg_speed_change = speed_change_sum / sample_count
    avg_speed_ratio = total_speed_ratio / sample_count
    slow_ratio = too_slow_count / sample_count
    
    return learner_laps, collision_count, avg_speed_change, avg_speed_ratio, slow_ratio


# =============================================================================
# EVALUATION
# =============================================================================

def create_random_genome():
    return np.random.randn(GENOME_SIZE) * 1.0


def evaluate_genome(args):
    """Evaluate genome over multiple runs."""
    genome, base_seed = args
    
    total_laps = 0.0
    total_collisions = 0
    total_speed_change = 0.0
    total_speed_ratio = 0.0
    total_slow_ratio = 0.0
    
    n_runs = 3
    for i in range(n_runs):
        laps, collisions, speed_change, speed_ratio, slow_ratio = run_simulation(
            genome, base_seed + i * 1000
        )
        laps_penalized = max(0.0, laps - collisions * 0.3)
        total_laps += laps_penalized
        total_collisions += collisions
        total_speed_change += speed_change
        total_speed_ratio += speed_ratio
        total_slow_ratio += slow_ratio
    
    avg_laps = total_laps / n_runs
    avg_collisions = total_collisions / n_runs
    avg_speed_change = total_speed_change / n_runs
    avg_speed_ratio = total_speed_ratio / n_runs
    avg_slow_ratio = total_slow_ratio / n_runs

    # Induce heavy penalty: reduce avg_laps by 0.3 per collision
    avg_laps_penalized = avg_laps  # Already penalized per sample above
    
    # NEW FITNESS FUNCTION:
    # We want cars to go FAST and SMOOTH while avoiding collisions
    
    # Base fitness: reward speed (laps completed)
    fitness = avg_laps_penalized * 10.0  # Scale up for better resolution
    
    # CRITICAL: Massive collision penalty (forces safe driving)
    fitness -= avg_collisions * 50.0
    
    # Tier 1 (0-15 laps): Just avoid collisions
    # (Already handled by collision penalty)
    
    # Tier 2 (15-30 laps): Penalize jerky driving
    if avg_laps > 15:
        jerkiness_penalty = avg_speed_change * 200.0  # Heavy penalty for speed changes
        fitness -= jerkiness_penalty
    
    # Tier 3 (30+ laps): Penalize slow driving heavily
    if avg_laps > 30:
        # Want speed ratio close to 1.0 (matching base speed)
        speed_deviation = abs(avg_speed_ratio - 1.0)
        fitness -= speed_deviation * 50.0
        
        # Extra penalty for being too slow
        if avg_speed_ratio < 0.9:
            fitness -= (0.9 - avg_speed_ratio) * 100.0
    
    return fitness, genome, avg_laps, avg_collisions, avg_speed_change, avg_speed_ratio


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

def crossover(p1, p2):
    """BLX-alpha crossover."""
    child = np.zeros(GENOME_SIZE)
    for i in range(GENOME_SIZE):
        alpha = np.random.uniform(-0.1, 1.1)
        child[i] = alpha * p1[i] + (1 - alpha) * p2[i]
    return child


def mutate(genome, rate=0.3, intensity=0.5):
    """Gaussian mutation."""
    child = genome.copy()
    for i in range(GENOME_SIZE):
        if np.random.random() < rate:
            child[i] += np.random.randn() * intensity
    return child


def run_generation(gen_num, population):
    """Evaluate population in parallel."""
    args_list = [(g, gen_num * 10000 + i) for i, g in enumerate(population)]
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_genome, args) for args in args_list]
        for future in as_completed(futures):
            results.append(future.result())
    
    results.sort(reverse=True, key=lambda x: x[0])
    return results


def evolve_population(results, pop_size):
    """Create next generation."""
    elite_count = max(5, pop_size // 10)
    elites = [r[1] for r in results[:elite_count]]
    
    new_pop = list(elites)
    
    # Tournament selection + crossover
    while len(new_pop) < pop_size:
        t_size = 5
        candidates = random.sample(range(len(results)), t_size)
        p1 = results[min(candidates)][1]
        candidates = random.sample(range(len(results)), t_size)
        p2 = results[min(candidates)][1]
        
        child = crossover(p1, p2)
        child = mutate(child, rate=0.3, intensity=0.5)
        new_pop.append(child)
    
    # Add random genomes
    for i in range(max(3, pop_size // 15)):
        new_pop[-(i + 1)] = create_random_genome()
    
    return new_pop[:pop_size]


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def run_sample_with_recording(genome, seed):
    """Run simulation with frame recording for video."""
    np.random.seed(seed)
    
    cars = np.zeros((NUM_CARS, CAR_FIELDS), dtype=np.float64)
    for i in range(NUM_CARS):
        cars[i, C_ANGLE] = (i / NUM_CARS) * 2 * np.pi
        cars[i, C_SPEED] = BASE_SPEED
        cars[i, C_DISTANCE] = 0.0
        cars[i, C_IS_INDUCER] = 1.0 if i == 0 else 0.0
        cars[i, C_PREV_SPEED] = BASE_SPEED
    
    frames = []
    steps = int(SIM_DURATION / DT)
    inducer_timer = 0.0
    inducer_target = 1.0
    
    for step in range(steps):
        # Record every step (~48 FPS)
        frame = []
        for i in range(NUM_CARS):
            frame.append((
                cars[i, C_ANGLE],
                cars[i, C_SPEED] / BASE_SPEED,
                cars[i, C_SPEED],
                cars[i, C_IS_INDUCER] > 0.5,
                cars[i, C_DISTANCE] / CIRCUMFERENCE
            ))
        frames.append(frame)
        
        # Inducer behavior (same as simulation)
        inducer_timer += DT
        if inducer_timer > 2.0:
            inducer_timer = 0.0
            inducer_target = 0.7 + np.random.random() * 0.6
        
        cars[0, C_SPEED] += (inducer_target * BASE_SPEED - cars[0, C_SPEED]) * 0.15
        
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
                # Use genome to compute target speed
                w0, w1, w2, b = genome[0], genome[1], genome[2], genome[3]
                gap_norm = min(gap / 40.0, 1.5)
                front_norm = cars[front_idx, C_SPEED] / (BASE_SPEED * 1.5)
                own_norm = cars[i, C_SPEED] / (BASE_SPEED * 1.5)
                
                raw = gap_norm * w0 + front_norm * w1 + own_norm * w2 + b
                sigmoid = 1.0 / (1.0 + np.exp(-raw))
                speed_mult = 0.3 + sigmoid * 1.0
                target_speed = speed_mult * BASE_SPEED
                
                speed_diff = target_speed - cars[i, C_SPEED]
                cars[i, C_SPEED] += speed_diff * 0.3
                cars[i, C_SPEED] = max(0.5, min(cars[i, C_SPEED], BASE_SPEED * 1.5))
                cars[i, C_PREV_SPEED] = cars[i, C_SPEED]
            
            dist = cars[i, C_SPEED] * DT
            dtheta = dist / CIRCLE_RADIUS
            cars[i, C_ANGLE] = (cars[i, C_ANGLE] + dtheta) % (2 * np.pi)
            cars[i, C_DISTANCE] += dist
    
    learner_laps = sum(cars[i, C_DISTANCE] / CIRCUMFERENCE 
                       for i in range(NUM_CARS) if cars[i, C_IS_INDUCER] < 0.5) / (NUM_CARS - 1)
    
    return frames, learner_laps, genome


def save_video(frames, filename, gen_num, laps):
    """Generate video from frames."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import imageio.v2 as imageio
    except ImportError:
        print(f"  Skipping video (matplotlib/imageio not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    images = []
    
    for frame_idx, frame in enumerate(frames):
        ax.clear()
        ax.set_xlim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
        ax.set_ylim(-CIRCLE_RADIUS - 20, CIRCLE_RADIUS + 20)
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw road
        circle = plt.Circle((0, 0), CIRCLE_RADIUS, color='#3d3d5c', fill=False, linewidth=8)
        ax.add_patch(circle)

        # Draw cars as rectangles (car shapes)
        car_length = 3.5 * CAR_SIZE_RATIO
        car_width = 1.5 * CAR_SIZE_RATIO
        for i, (angle, speed_mult, speed, is_inducer, car_laps) in enumerate(frame):
            x = CIRCLE_RADIUS * np.cos(angle)
            y = CIRCLE_RADIUS * np.sin(angle)
            theta = angle + np.pi / 2  # Tangent direction
            if is_inducer:
                color = '#ff6b6b'  # Red for inducer
            elif speed < BASE_SPEED * 0.8:
                color = '#ffaa00'  # Yellow if slow
            else:
                color = '#00ff88'  # Green if good speed

            # Rectangle center at (x, y), rotated to match tangent
            rect = plt.Rectangle(
                (x - car_length/2 * np.cos(theta),
                 y - car_length/2 * np.sin(theta)),
                car_length, car_width,
                angle=np.degrees(theta),
                color=color, ec='white', lw=1.5
            )
            ax.add_patch(rect)
        
        # Info text
        t = frame_idx * 4 * DT
        ax.text(0, CIRCLE_RADIUS + 12, 
                f'Gen {gen_num} | t={t:.1f}s | Avg Laps: {laps:.1f}',
                ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
        
        fig.patch.set_facecolor('#1a1a2e')
        fig.canvas.draw()
        
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        images.append(img)
    
    imageio.mimsave(filename, images, fps=25)
    plt.close(fig)
    print(f"    Saved video: {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  CIRCLE ROAD TRAFFIC OPTIMIZATION - V6 (Simplified)")
    print("=" * 70)
    print(f"  Cars: {NUM_CARS} (1 inducer + {NUM_CARS-1} learners)")
    print(f"  Circle: {CIRCLE_RADIUS}m radius | Duration: {SIM_DURATION}s")
    print(f"  Population: {SAMPLES_PER_GEN} | Generations: {NUM_GENERATIONS}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 70)
    
    # Warm up JIT
    print("\n  JIT compiling...", end=" ", flush=True)
    _ = run_simulation(create_random_genome(), 0)
    print("done!\n")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize
    population = [create_random_genome() for _ in range(SAMPLES_PER_GEN)]
    best_fitness_history = []
    best_laps_history = []
    
    for gen in range(1, NUM_GENERATIONS + 1):
        t0 = timer.time()
        
        results = run_generation(gen, population)
        
        best_fitness, best_genome, best_laps, best_collisions, best_smoothness, best_speed = results[0]
        avg_laps = np.mean([r[2] for r in results])
        
        best_fitness_history.append(best_fitness)
        best_laps_history.append(best_laps)
        
        dt = timer.time() - t0
        
        print(f"  Gen {gen:3d} | {dt:4.1f}s | "
              f"Best: {best_laps:5.1f} laps, {best_collisions:4.0f} col, "
              f"jerk={best_smoothness:.4f}, speed={best_speed:.2f}x | Avg: {avg_laps:5.1f} laps")
        
        # Save best genome
        with open(os.path.join(SAVE_DIR, f"gen{gen:03d}_best.pkl"), "wb") as f:
            pickle.dump({
                'genome': best_genome,
                'laps': best_laps,
                'collisions': best_collisions,
                'generation': gen
            }, f)
        
        # Generate videos for gen 1 and gen 50
        if gen == 1 or gen == NUM_GENERATIONS:
            print(f"  Recording gen {gen} video...")
            frames, laps, genome = run_sample_with_recording(best_genome, gen * 10000)
            vid_path = os.path.join(SAVE_DIR, f"gen{gen:03d}_best.mp4")
            save_video(frames, vid_path, gen, laps)
        
        # Evolve
        population = evolve_population(results, SAMPLES_PER_GEN)
    
    # Save history
    with open(os.path.join(SAVE_DIR, "history.pkl"), "wb") as f:
        pickle.dump({
            'best_fitness': best_fitness_history,
            'best_laps': best_laps_history
        }, f)
    
    print("\n" + "=" * 70)
    print("  COMPLETE!")
    print(f"  Gen 1:  {best_laps_history[0]:.1f} laps")
    print(f"  Gen {NUM_GENERATIONS}: {best_laps_history[-1]:.1f} laps")
    print(f"  Improvement: {best_laps_history[-1] - best_laps_history[0]:+.1f} laps")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()