import numpy as np
import matplotlib.pyplot as plt


def display_grid_and_path(inst, evaluator, path, title_suffix=""):
    """
    Display a grid with nodes and path coverage visualization.

    Args:
        inst: AgcspInstance object containing grid_nodes and obstacle_nodes
        evaluator: AgcspEvaluator object to calculate coverage
        path: List of (row, col) tuples representing the path
        title_suffix: Additional text to add to the title
    """
    covered_nodes = evaluator.get_covered_nodes_list(path) if len(path) > 0 else []
    total_nodes = inst.target_node_count
    covered_count = len(covered_nodes)
    coverage_percentage = (covered_count / total_nodes) * 100 if total_nodes > 0 else 0

    # Check for obstacle coverage
    covered_obstacles = []
    obstacle_warning = ""
    if hasattr(inst, 'obstacle_nodes') and len(inst.obstacle_nodes) > 0:
        covered_set = set(map(tuple, covered_nodes))
        covered_obstacles = [obs for obs in inst.obstacle_nodes if tuple(obs) in covered_set]
        if covered_obstacles:
            obstacle_warning = f" - {len(covered_obstacles)} obstacles covered!"

    plt.figure(figsize=(10, 8))

    # Plot valid nodes
    plt.scatter(inst.field_nodes[:, 0], inst.field_nodes[:, 1], c='#AAAAAA', s=3,
                label='Uncovered Nodes', alpha=0.7)

    if len(covered_nodes) > 0:
        plt.scatter(covered_nodes[:, 0], covered_nodes[:, 1], c='#00AA44', s=4,
                   label='Covered Nodes', alpha=0.8)

    # Plot obstacles
    if hasattr(inst, 'obstacle_nodes') and len(inst.obstacle_nodes) > 0:
        plt.scatter(inst.obstacle_nodes[:, 0], inst.obstacle_nodes[:, 1], c='#CC0000', s=4,
                   marker='s', label='Obstacles', linewidths=0.5)

        # Highlight covered obstacles
        if covered_obstacles:
            covered_obstacles_arr = np.array(covered_obstacles)
            plt.scatter(covered_obstacles_arr[:, 0], covered_obstacles_arr[:, 1],
                       c='#FF6600', s=10, marker='D',
                       linewidths=2, label='Covered Obstacles')

    # Plot path
    if len(path) > 0:
        path_arr = np.array(path)
        plt.plot(path_arr[:, 0], path_arr[:, 1], color='#9900CC', linewidth=2,
                label='Sprayer Path', marker='o', markersize=4, markerfacecolor='white',
                markeredgecolor='#9900CC', markeredgewidth=1.5)

        # Plot start point (green star)
        plt.scatter(path_arr[0, 0], path_arr[0, 1], c='#00DD00', s=150,
                   marker='*', label='Start', edgecolors='black', linewidths=1.5, zorder=5)

        # Plot end point (red square)
        if len(path) > 1:
            plt.scatter(path_arr[-1, 0], path_arr[-1, 1], c='#DD0000', s=80,
                       marker='s', label='End', edgecolors='black', linewidths=1.5, zorder=5)

    plt.axis('equal')
    plt.legend(loc='upper right', framealpha=0.9, fontsize=8)

    title = f'{title_suffix}\nCoverage: {covered_count}/{total_nodes} nodes ({coverage_percentage:.1f}%){obstacle_warning}'
    plt.title(title, fontsize=10, fontweight='bold')

    if title_suffix:
        print(f"{title_suffix}")
    print(f"Coverage: {covered_count}/{total_nodes} nodes ({coverage_percentage:.1f}%)")
    if hasattr(inst, 'obstacle_nodes') and len(inst.obstacle_nodes) > 0:
        print(f"Obstacles: {len(covered_obstacles)}/{len(inst.obstacle_nodes)} covered")
        if covered_obstacles:
            print("WARNING: Sprayer is covering obstacles!")

    plt.show()
