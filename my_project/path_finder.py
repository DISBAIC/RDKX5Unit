import heapq
import math
from ftplib import print_line

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Circle, Rectangle, Wedge
import time

class Movement:
    def __init__(self, angle, distance):
        self.angle = angle
        self.distance = distance

def export_optimized_path(game_map: list[list[bool]], start: tuple[int, int], end: tuple[int, int], car_radius: int, fire_radius: int) -> list[Movement]:
    """
    导出优化后的路径，直接返回Movement对象列表
    
    Parameters:
    - game_map: 2D boolean map, True indicates obstacle, False indicates passable
    - start: Start coordinates (x, y)
    - end: End coordinates (x, y)
    - car_radius: Vehicle radius for collision detection
    - fire_radius: Firing radius, final position should be 1-2 times fire_radius from target
    
    Returns:
    - list[Movement]: List of movement objects with angle and distance
    """
    
    if not game_map or not game_map[0]:
        return []
    
    height = len(game_map)
    width = len(game_map[0])
    
    def is_valid_position(x, y):
        """Check if position is valid (considering vehicle radius and no-go zone)"""
        # Check boundaries
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        
        # Check No-Go Zone: vehicle cannot enter within fire_radius of end point
        dist_to_end = math.sqrt((x - end[0])**2 + (y - end[1])**2)
        if dist_to_end < fire_radius:
            return False
        
        # Check circular area centered at (x,y) with car_radius for obstacles
        for dy in range(-car_radius, car_radius + 1):
            for dx in range(-car_radius, car_radius + 1):
                if dx*dx + dy*dy <= car_radius*car_radius:  # Circular collision detection
                    check_x, check_y = x + dx, y + dy
                    if 0 <= check_x < width and 0 <= check_y < height:
                        if game_map[check_y][check_x]:  # Hit obstacle
                            return False
        return True
    
    def analyze_polar_obstacles():
        """Analyze obstacles using ray casting from end point"""
        obstacle_rays = []
        
        # Cast rays in all directions (every degree for high precision)
        for angle_deg in range(0, 360, 1):  # 1-degree resolution
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            # Cast ray from end point outward
            hit_obstacle = False
            hit_distance = float('inf')
            
            # Ray casting with fine step resolution
            step_size = 0.5  # Fine resolution for accurate detection
            max_ray_distance = 2 * fire_radius + car_radius + 5  # Extend beyond our interest range
            
            for step in range(int(max_ray_distance / step_size)):
                ray_distance = step * step_size
                ray_x = end[0] + dx * ray_distance
                ray_y = end[1] + dy * ray_distance
                
                # Check map boundaries
                if ray_x < 0 or ray_x >= width or ray_y < 0 or ray_y >= height:
                    hit_distance = ray_distance
                    hit_obstacle = True
                    break
                
                # Check for obstacle
                grid_x = int(ray_x)
                grid_y = int(ray_y)
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    if game_map[grid_y][grid_x]:  # Hit obstacle
                        hit_distance = ray_distance
                        hit_obstacle = True
                        break
            
            obstacle_rays.append({
                'angle': angle_deg,
                'distance': hit_distance,
                'hit_obstacle': hit_obstacle
            })
        
        return obstacle_rays
    
    def find_clear_sectors(obstacle_rays, min_gap_degrees=45):
        """Find sectors with obstacle gaps >= min_gap_degrees"""
        clear_sectors = []
        
        if not obstacle_rays:
            return clear_sectors
        
        # Sort by angle to ensure proper order
        obstacle_rays.sort(key=lambda x: x['angle'])
        
        # Find angles where obstacles are within our range of interest
        obstacle_angles = []
        min_obstacle_distance = fire_radius  # We care about obstacles closer than this
        
        for ray in obstacle_rays:
            if ray['hit_obstacle'] and ray['distance'] <= min_obstacle_distance:
                obstacle_angles.append(ray['angle'])
        
        if not obstacle_angles:
            # No close obstacles, entire circle is clear
            return [(0, 360)]
        
        # Find gaps between consecutive obstacle angles
        obstacle_angles.sort()
        
        for i in range(len(obstacle_angles)):
            current_angle = obstacle_angles[i]
            next_angle = obstacle_angles[(i + 1) % len(obstacle_angles)]
            
            # Calculate gap between current and next obstacle
            if next_angle > current_angle:
                gap = next_angle - current_angle
                sector_start = current_angle
                sector_end = next_angle
            else:
                # Handle wrap-around case (e.g., from 350° to 10°)
                gap = (360 - current_angle) + next_angle
                sector_start = current_angle
                sector_end = next_angle
            
            # If gap is large enough, this is a clear sector
            if gap >= min_gap_degrees:
                # Normalize angles to [0, 360) range
                if sector_end > 360:
                    # Handle wrapping case
                    clear_sectors.append((sector_start, 360))
                    clear_sectors.append((0, sector_end - 360))
                else:
                    clear_sectors.append((sector_start, sector_end))
        
        return clear_sectors
    
    def is_position_in_clear_sector(pos, clear_sectors):
        """Check if a position is within any clear sector"""
        # If no clear sectors, position is invalid
        if not clear_sectors:
            return False
            
        dx = pos[0] - end[0]
        dy = pos[1] - end[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # 精确检查每个扇区
        for sector_start, sector_end in clear_sectors:
            # 处理正常扇区
            if sector_end <= 360:
                if sector_start <= angle <= sector_end:
                    return True
            else:
                # 处理跨越0°的扇区 (如 350° to 370° 表示 350° to 360° 和 0° to 10°)
                if angle >= sector_start or angle <= (sector_end - 360):
                    return True
        
        return False
    
    def get_target_positions():
        """Get valid target positions in clear sectors"""
        targets = []
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        # Analyze obstacles in polar coordinates and find clear sectors
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        
        print(f"DEBUG: Found {len(clear_sectors)} clear sectors with gaps >= 45°")
        
        # 严格检查：如果没有合法扇区，立即返回空列表
        if not clear_sectors:
            print("DEBUG: No clear sectors found, no valid target positions")
            return []
        
        # Search for valid positions around target point
        search_range = max_dist + car_radius
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                target_x = end[0] + dx
                target_y = end[1] + dy
                
                # Check if within map boundaries
                if 0 <= target_x < width and 0 <= target_y < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    # Check distance constraint: must be in the annular region AND outside the no-go zone
                    if min_dist <= dist <= max_dist and dist > fire_radius and is_valid_position(target_x, target_y):
                        # 严格检查：只有在合法扇区内的位置才被添加
                        if is_position_in_clear_sector((target_x, target_y), clear_sectors):
                            targets.append((target_x, target_y))
        
        return targets
    
    def heuristic(pos1, pos2):
        """A* algorithm heuristic function (Euclidean distance)"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(pos):
        """Get adjacent positions in 8 directions, but only in clear sectors"""
        x, y = pos
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # up down left right
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
        ]
        
        # Get clear sectors for checking
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            # Position must be valid AND in a clear sector
            if is_valid_position(new_x, new_y):
                # Additional check: position must be in clear sector OR be the start position
                if pos == start or is_position_in_clear_sector((new_x, new_y), clear_sectors):
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def a_star_search(start_pos, target_pos):
        """A* pathfinding algorithm with aggressive early termination"""
        open_set = []
        heapq.heappush(open_set, (0, start_pos))
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, target_pos)}
        
        open_set_hash = {start_pos}
        
        # Get clear sectors for early termination check
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        print(f"DEBUG: A* starting with {len(clear_sectors)} clear sectors available")
        
        steps_count = 0
        while open_set:
            steps_count += 1
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Aggressive early termination: if current position is in clear sector and meets distance constraints
            if current != start_pos:  # Don't terminate at start position
                dist_to_end = math.sqrt((current[0] - end[0])**2 + (current[1] - end[1])**2)
                
                # Check if current position meets all criteria for early termination
                if (min_dist <= dist_to_end <= max_dist and 
                    dist_to_end > fire_radius and  # Outside no-go zone
                    is_position_in_clear_sector(current, clear_sectors)):
                    
                    # Immediate early termination: found first valid clear sector point
                    print(f"DEBUG: Immediate early termination at {current} after {steps_count} steps")
                    
                    # Reconstruct path to current position
                    path = []
                    temp_current = current
                    while temp_current in came_from:
                        path.append(temp_current)
                        temp_current = came_from[temp_current]
                    path.append(start_pos)
                    path.reverse()
                    return path
            
            # Normal termination: reached original target
            if current == target_pos:
                print(f"DEBUG: Reached original target {target_pos}")
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                # Calculate movement cost
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(dx*dx + dy*dy)  # Euclidean distance
                
                temp_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, target_pos)
                    
                    if neighbor not in open_set_hash:
                        # Check if this neighbor is a potential early termination point
                        neighbor_dist_to_end = math.sqrt((neighbor[0] - end[0])**2 + (neighbor[1] - end[1])**2)
                        is_clear_sector_point = (neighbor != start_pos and
                                               min_dist <= neighbor_dist_to_end <= max_dist and 
                                               neighbor_dist_to_end > fire_radius and  
                                               is_position_in_clear_sector(neighbor, clear_sectors))
                        
                        if is_clear_sector_point:
                            # Give highest priority to clear sector points for immediate termination
                            priority_boost = 50  # Very high boost for aggressive early termination
                            heapq.heappush(open_set, (f_score[neighbor] - priority_boost, neighbor))
                            print(f"DEBUG: High priority clear sector candidate {neighbor}")
                        else:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
                        open_set_hash.add(neighbor)
        
        return None  # No path found
    
    def path_to_movements(path):
        """Convert path coordinates to Movement objects with angle and distance"""
        if not path or len(path) < 2:
            return []
        
        def can_move_directly(start_pos, end_pos):
            """Check if we can move directly from start_pos to end_pos"""
            x1, y1 = start_pos
            x2, y2 = end_pos
            
            # Use Bresenham-like algorithm to check all points along the line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx == 0 and dy == 0:
                return True
            
            # Calculate number of steps needed
            steps = max(dx, dy)
            
            # Check each point along the path
            for i in range(steps + 1):
                # Linear interpolation
                t = i / max(steps, 1)
                check_x = int(x1 + t * (x2 - x1))
                check_y = int(y1 + t * (y2 - y1))
                
                # Check if this position is valid
                if not is_valid_position(check_x, check_y):
                    return False
            
            return True
        
        # First, simplify the path by connecting distant points directly
        simplified_path = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current_pos = path[i]
            
            # Find the farthest point we can reach directly
            max_reachable = i + 1
            for j in range(i + 2, len(path)):
                if can_move_directly(current_pos, path[j]):
                    max_reachable = j
                else:
                    break
            
            # Add the farthest reachable point
            simplified_path.append(path[max_reachable])
            i = max_reachable
        
        print(f"DEBUG: Simplified path from {len(path)} to {len(simplified_path)} points")
        
        # Convert simplified path to Movement objects
        movements = []
        
        for i in range(len(simplified_path) - 1):
            current_pos = simplified_path[i]
            next_pos = simplified_path[i + 1]
            
            # Calculate movement vector
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            
            # Calculate distance
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate angle (in degrees, 0° = East, 90° = North)
            angle = math.degrees(math.atan2(dy, dx))
            
            # Normalize angle to [0, 360) range
            if angle < 0:
                angle += 360
            
            # Create Movement object
            movement = Movement(angle, distance)
            movements.append(movement)
            
            print(f"DEBUG: Movement {i+1}: angle={angle:.1f}°, distance={distance:.2f}")
        
        return movements
    
    # Main logic
    if not is_valid_position(start[0], start[1]):
        print("DEBUG: Invalid start position!")
        return []
    
    # Get valid target positions
    print("DEBUG: Getting target positions...")
    target_positions = get_target_positions()
    print(f"DEBUG: Found {len(target_positions)} valid target positions")
    
    if not target_positions:
        print("DEBUG: No valid target positions found!")
        return []
    
    # Find the closest valid target position to start point
    best_target = min(target_positions, key=lambda pos: heuristic(start, pos))
    print(f"DEBUG: Selected best target: {best_target}")
    
    # Use A* algorithm to find path
    print("DEBUG: Starting A* pathfinding...")
    path = a_star_search(start, best_target)
    if not path:
        print("DEBUG: A* failed to find path!")
        return []
    
    print(f"DEBUG: A* found path with {len(path)} points")
    
    # Convert path to Movement objects
    movements = path_to_movements(path)
    print(f"DEBUG: Generated {len(movements)} movement commands")
    
    return movements

def angle_fix(movements:list[Movement]) -> list[Movement]:
    if (movements):
        for m in movements:
            m.angle = 360.0-m.angle
    return movements

# Test function
def generate_random_map(width, height, obstacle_density=0.15, min_obstacle_size=3, max_obstacle_size=8):
    """
    生成随机地图
    
    Parameters:
    - width, height: Map dimensions
    - obstacle_density: Percentage of map covered by obstacles (0.0 to 1.0)
    - min_obstacle_size, max_obstacle_size: Size range for rectangular obstacles
    
    Returns:
    - 2D boolean map
    """
    test_map = [[False for _ in range(width)] for _ in range(height)]
    
    # Calculate total cells and target obstacle cells
    total_cells = width * height
    target_obstacle_cells = int(total_cells * obstacle_density)
    current_obstacle_cells = 0
    
    # Generate random obstacles until we reach target density
    attempts = 0
    max_attempts = target_obstacle_cells * 2  # Prevent infinite loop
    
    while current_obstacle_cells < target_obstacle_cells and attempts < max_attempts:
        attempts += 1
        
        # Random obstacle size
        obs_width = random.randint(min_obstacle_size, max_obstacle_size)
        obs_height = random.randint(min_obstacle_size, max_obstacle_size)
        
        # Random position (ensure obstacle fits within map)
        obs_x = random.randint(0, width - obs_width)
        obs_y = random.randint(0, height - obs_height)
        
        # Add obstacle
        obstacle_added = 0
        for y in range(obs_y, min(obs_y + obs_height, height)):
            for x in range(obs_x, min(obs_x + obs_width, width)):
                if not test_map[y][x]:  # Only count new obstacle cells
                    test_map[y][x] = True
                    obstacle_added += 1
        
        current_obstacle_cells += obstacle_added
    
    # Add some scattered single-cell obstacles for variety
    scatter_count = int(total_cells * 0.02)  # 2% scattered obstacles
    for _ in range(scatter_count):
        if current_obstacle_cells >= target_obstacle_cells:
            break
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if not test_map[y][x]:
            test_map[y][x] = True
            current_obstacle_cells += 1
    
    print(f"Generated random map: {current_obstacle_cells}/{total_cells} = {current_obstacle_cells/total_cells:.1%} obstacles")
    return test_map

def test_export_optimized_path():
    """Test the export_optimized_path function with random map"""
    # Create a random test map
    width, height = 80, 60
    test_map = generate_random_map(width, height, obstacle_density=0.2)
    
    # Set random start and end positions
    # Ensure start and end are in clear areas
    def find_clear_position():
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(5, width - 6)
            y = random.randint(5, height - 6)
            # Check if area around position is clear
            is_clear = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        if test_map[y + dy][x + dx]:
                            is_clear = False
                            break
                if not is_clear:
                    break
            if is_clear:
                return (x, y)
        # Fallback to corner positions
        return (5, 5)
    
    start = find_clear_position()
    end = find_clear_position()
    
    # Ensure start and end are far enough apart
    min_distance = 20
    max_attempts = 50
    attempts = 0
    while math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) < min_distance and attempts < max_attempts:
        end = find_clear_position()
        attempts += 1
    
    # Set parameters
    car_radius = 2
    fire_radius = 4
    
    print(f"Random test: Start={start}, End={end}, Distance={math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2):.1f}")
    
    # Test the function
    print("Testing export_optimized_path function...")
    movements = angle_fix(export_optimized_path(test_map, start, end, car_radius, fire_radius))

    if movements:
        print(f"SUCCESS: Found {len(movements)} movement commands:")
        total_distance = 0
        for i, movement in enumerate(movements):
            print(f"  Movement {i+1}: Angle={movement.angle:.1f}°, Distance={movement.distance:.2f}")
            total_distance += movement.distance
        print(f"Total travel distance: {total_distance:.2f}")
    else:
        print("FAILED: No movement commands generated")
    
    return movements

def visualize_path_and_movements(game_map: list[list[bool]], start: tuple[int, int], end: tuple[int, int], 
                                car_radius: int, fire_radius: int, movements: list[Movement] = None):
    """
    可视化显示地图、路径和移动命令
    
    Parameters:
    - game_map: 2D boolean map, True indicates obstacle, False indicates passable
    - start: Start coordinates (x, y)
    - end: End coordinates (x, y)
    - car_radius: Vehicle radius for collision detection
    - fire_radius: Firing radius
    - movements: List of movement objects (optional, will be calculated if not provided)
    """
    if not movements:
        movements = export_optimized_path(game_map, start, end, car_radius, fire_radius)
    
    height = len(game_map)
    width = len(game_map[0])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Path Planning Visualization', fontsize=16)
    
    # Plot 1: Map with obstacles and path
    ax1.set_title('Map with Obstacles and Path')
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Draw obstacles
    for y in range(height):
        for x in range(width):
            if game_map[y][x]:
                rect = Rectangle((x, y), 1, 1, facecolor='red', alpha=0.7)
                ax1.add_patch(rect)
    
    # Draw start point
    start_circle = Circle(start, car_radius, facecolor='green', alpha=0.7, label='Start')
    ax1.add_patch(start_circle)
    ax1.plot(start[0], start[1], 'go', markersize=8)
    ax1.text(start[0] + 1, start[1] + 1, 'START', fontsize=10, fontweight='bold')
    
    # Draw end point
    end_circle = Circle(end, 0.5, facecolor='blue', alpha=0.7, label='Target')
    ax1.add_patch(end_circle)
    ax1.plot(end[0], end[1], 'bo', markersize=8)
    ax1.text(end[0] + 1, end[1] + 1, 'TARGET', fontsize=10, fontweight='bold')
    
    # Draw fire radius (no-go zone)
    fire_circle = Circle(end, fire_radius, facecolor='none', edgecolor='red', 
                        linestyle='--', linewidth=2, alpha=0.7, label='No-Go Zone')
    ax1.add_patch(fire_circle)
    
    # Draw firing range
    fire_range_inner = Circle(end, fire_radius, facecolor='none', edgecolor='orange', 
                             linestyle=':', linewidth=1, alpha=0.5)
    fire_range_outer = Circle(end, 2 * fire_radius, facecolor='none', edgecolor='orange', 
                             linestyle=':', linewidth=1, alpha=0.5, label='Fire Range')
    ax1.add_patch(fire_range_inner)
    ax1.add_patch(fire_range_outer)
    
    # Draw path if movements exist
    if movements:
        current_pos = start
        path_x = [current_pos[0]]
        path_y = [current_pos[1]]
        
        for i, movement in enumerate(movements):
            # Calculate next position
            angle_rad = math.radians(movement.angle)
            next_x = current_pos[0] + movement.distance * math.cos(angle_rad)
            next_y = current_pos[1] + movement.distance * math.sin(angle_rad)
            
            path_x.append(next_x)
            path_y.append(next_y)
            
            # Draw movement arrow
            ax1.arrow(current_pos[0], current_pos[1], 
                     next_x - current_pos[0], next_y - current_pos[1],
                     head_width=1, head_length=1, fc='purple', ec='purple', alpha=0.7)
            
            # Add movement number
            mid_x = (current_pos[0] + next_x) / 2
            mid_y = (current_pos[1] + next_y) / 2
            ax1.text(mid_x, mid_y, str(i+1), fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            current_pos = (next_x, next_y)
        
        # Draw complete path line
        ax1.plot(path_x, path_y, 'purple', linewidth=2, alpha=0.8, label='Path')
        
        # Draw final position
        final_circle = Circle(current_pos, car_radius, facecolor='purple', alpha=0.5, label='Final Position')
        ax1.add_patch(final_circle)
        ax1.plot(current_pos[0], current_pos[1], 'mo', markersize=8)
        ax1.text(current_pos[0] + 1, current_pos[1] + 1, 'FINAL', fontsize=10, fontweight='bold')
    
    ax1.legend()
    ax1.invert_yaxis()  # Invert Y axis to match array indexing
    
    # Plot 2: Movement commands polar plot
    ax2.set_title('Movement Commands')
    
    if movements:
        # Create polar subplot
        ax2.clear()
        ax2 = fig.add_subplot(122, projection='polar')
        ax2.set_title('Movement Commands (Polar View)')
        
        # Plot movements in polar coordinates
        angles = []
        distances = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(movements)))
        
        for i, movement in enumerate(movements):
            angle_rad = math.radians(movement.angle)
            angles.append(angle_rad)
            distances.append(movement.distance)
            
            # Draw movement vector
            ax2.arrow(0, 0, angle_rad, movement.distance, 
                     head_width=0.1, head_length=0.5, fc=colors[i], ec=colors[i], alpha=0.7)
            
            # Add movement label
            ax2.text(angle_rad, movement.distance + 0.5, f'{i+1}', 
                    fontsize=8, ha='center', va='center')
        
        # Add movement details as text
        info_text = "Movement Details:\n"
        total_distance = 0
        for i, movement in enumerate(movements):
            info_text += f"#{i+1}: {movement.angle:.1f}° {movement.distance:.2f}u\n"
            total_distance += movement.distance
        info_text += f"Total: {total_distance:.2f}u"
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No movements to display', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def test_with_visualization():
    """带可视化的测试函数 - 使用随机地图"""
    print("Testing with visualization using random map...")
    
    # Create a random test map with higher complexity
    width, height = 200, 185
    test_map = generate_random_map(width, height, obstacle_density=0.08, min_obstacle_size=5, max_obstacle_size=20)
    
    # Find good start and end positions
    def find_clear_position():
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(5, width - 6)
            y = random.randint(5, height - 6)
            # Check if area around position is clear (larger area for better positioning)
            is_clear = True
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        if test_map[y + dy][x + dx]:
                            is_clear = False
                            break
                if not is_clear:
                    break
            if is_clear:
                return (x, y)
        # Fallback to corner positions
        return (random.randint(5, 10), random.randint(5, 10))
    
    # Generate multiple start/end combinations and pick the best one
    best_start, best_end = None, None
    best_distance = 0
    
    for _ in range(10):  # Try 10 different combinations
        candidate_start = find_clear_position()
        candidate_end = find_clear_position()
        distance = math.sqrt((candidate_end[0] - candidate_start[0])**2 + (candidate_end[1] - candidate_start[1])**2)
        
        if distance > best_distance and distance >= 25:  # Minimum distance requirement
            best_distance = distance
            best_start = candidate_start
            best_end = candidate_end
    
    # Use best combination or fallback
    start = best_start if best_start else (5, 5)
    end = best_end if best_end else (width - 10, height - 10)
    
    # Set parameters
    car_radius = 10
    fire_radius = 10  # Random fire radius for variety

    print(f"Random test with visualization:")
    print(f"  Map size: {width}x{height}")
    print(f"  Start: {start}, End: {end}")
    print(f"  Distance: {math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2):.1f}")
    print(f"  Car radius: {car_radius}, Fire radius: {fire_radius}")
    
    # Get movements
    movements = export_optimized_path(test_map, start, end, car_radius, fire_radius)
    
    # Display results
    if movements:
        print(f"SUCCESS: Found {len(movements)} movement commands:")
        total_distance = 0
        for i, movement in enumerate(movements):
            print(f"  Movement {i+1}: Angle={360.0-movement.angle:.1f}°, Distance={movement.distance:.2f}")
            total_distance += movement.distance
        print(f"Total travel distance: {total_distance:.2f}")
        
        # Show visualization
        visualize_path_and_movements(test_map, start, end, car_radius, fire_radius, movements)
    else:
        print("FAILED: No movement commands generated")
        # Still show the map
        visualize_path_and_movements(test_map, start, end, car_radius, fire_radius, [])

def search_path(game_map: list[list[bool]], start: tuple[int, int], end: tuple[int, int], car_radius: int, fire_radius: int) -> list[Movement]:
    return angle_fix(export_optimized_path(game_map,start,end,car_radius,fire_radius))

if __name__ == "__main__":
    # Set random seed for reproducible results (optional)
    random.seed(time.time())  # Comment this out for truly random maps each time

    print("=== Random Map Tests ===")
    print("Testing path planning with randomly generated maps\n")
    
    # Run multiple random tests
    print("=== Quick Random Tests ===")
    for i in range(3):
        print(f"\n--- Random Test {i+1} ---")
        test_export_optimized_path()
    
    print("\n=== Detailed Test with Visualization ===")
    test_with_visualization()
    
    # Optional: Run test with different map complexities
    print("\n=== Testing Different Map Complexities ===")
    complexities = [0.1, 0.2, 0.3]
    for i, density in enumerate(complexities):
        print(f"\n--- Complexity Test {i+1}: {density:.0%} obstacles ---")
        width, height = 200, 185
        test_map = generate_random_map(width, height, obstacle_density=density)
        start = (5, 5)
        end = (width-10, height-10)
        movements = export_optimized_path(test_map, start, end, 2, 4)
        if movements:
            total_dist = sum(m.distance for m in movements)
            print(f"  SUCCESS: {len(movements)} movements, total distance: {total_dist:.2f}")
        else:
            print("  FAILED: No path found")
