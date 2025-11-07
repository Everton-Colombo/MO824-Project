import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Literal

from agricultural_csp.instance import AgcspInstance
from agricultural_csp.evaluator import AgcspEvaluator
from agricultural_csp.solution import AgcspSolution
from agricultural_csp.solver.agcsp_ts import (
    AgcspTS, 
    TSStrategy, 
    PhasedOptimizationParams, 
    TerminationCriteria, 
    DebugOptions
)
from agricultural_csp.solver.constructive_heuristics.base_heuristics import (
    ConstructiveHeuristicType
)

Node = Tuple[float, float]

def save_plot_to_file(inst: AgcspInstance, evaluator: AgcspEvaluator, path: List[Node], 
                      title_suffix: str, filepath: str):
    
    path_list = [tuple(p) for p in path]
    covered_nodes = evaluator.get_covered_nodes_list(path_list) if len(path_list) > 0 else []
    
    total_nodes = inst.target_node_count
    covered_count = len(covered_nodes)
    coverage_percentage = (covered_count / total_nodes) * 100 if total_nodes > 0 else 0
    
    covered_obstacles = []
    obstacle_warning = ""
    if hasattr(inst, 'obstacle_nodes_original') and len(inst.obstacle_nodes_original) > 0:
        covered_set = set(map(tuple, covered_nodes))
        covered_obstacles = [obs for obs in inst.obstacle_nodes_original if tuple(obs) in covered_set]
        if covered_obstacles:
            obstacle_warning = f" - {len(covered_obstacles)} obstacles covered!"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if inst.field_nodes.size > 0:
        ax.scatter(inst.field_nodes[:, 0], inst.field_nodes[:, 1], c='#AAAAAA', s=3, 
                   label='Uncovered Nodes', alpha=0.7, zorder=1)
    
    if len(covered_nodes) > 0:
        ax.scatter(covered_nodes[:, 0], covered_nodes[:, 1], c='#00AA44', s=4, 
                   label='Covered Nodes', alpha=0.8, zorder=2)
    
    if hasattr(inst, 'obstacle_nodes_original') and len(inst.obstacle_nodes_original) > 0:
        ax.scatter(inst.obstacle_nodes_original[:, 0], inst.obstacle_nodes_original[:, 1], c='#CC0000', s=4, 
                   marker='s', label='Obstacles', linewidths=0.5, zorder=3)
        
        if covered_obstacles:
            covered_obstacles_arr = np.array(covered_obstacles)
            ax.scatter(covered_obstacles_arr[:, 0], covered_obstacles_arr[:, 1], 
                       c='#FF6600', s=10, marker='D', 
                       linewidths=2, label='Covered Obstacles', zorder=7)
    
    if len(path) > 0:
        path_arr = np.array(path_list)
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='#9900CC', linewidth=2, 
                label='Sprayer Path', marker='o', markersize=4, markerfacecolor='white', 
                markeredgecolor='#9900CC', markeredgewidth=1.5, zorder=4)
        
        ax.scatter(path_arr[0, 0], path_arr[0, 1], c='#00DD00', s=150, 
                   marker='*', label='Start', edgecolors='black', linewidths=1.5, zorder=5)
        
        if len(path_list) > 1:
            ax.scatter(path_arr[-1, 0], path_arr[-1, 1], c='#DD0000', s=80, 
                       marker='s', label='End', edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.axis('equal')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    
    title = f'{title_suffix}\nCoverage: {covered_count}/{total_nodes} nodes ({coverage_percentage:.1f}%){obstacle_warning}'
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight', format='jpg')
    except Exception as e:
        print(f"  AVISO: Falha ao salvar gráfico {filepath}. Erro: {e}")
    
    plt.close(fig)

def run_single_instance(config_bundle: Dict[str, Any]) -> Dict[str, Any]:
    
    instance_name = config_bundle['instance_name']
    info = config_bundle['info']
    paths = config_bundle['paths']
    
    base_strategy_config = config_bundle['strategy']
    termination_criteria = config_bundle['termination_criteria']
    debug_options = config_bundle['debug_options']
    tenure = config_bundle['tenure']
    
    print(f"[Iniciando] {instance_name} (Tenure: {tenure})")

    instance_filepath = os.path.join(paths['instances'], f"{instance_name}.pkl")
    try:
        with open(instance_filepath, 'rb') as f:
            instance = pickle.load(f)
    except Exception as e:
        return {'Instance': instance_name, 'Status': 'Error (Load failed)', 'Error': str(e)}

    evaluator = AgcspEvaluator(instance)
    
    phased_opt = PhasedOptimizationParams(
        phase_iterations=base_strategy_config['phased_optimization']['iterations'],
        degradation_tolerances=base_strategy_config['phased_optimization']['tolerances']
    )
    
    strategy = TSStrategy(
        constructive_heuristic=base_strategy_config['constructive_heuristic'],
        search_strategy=base_strategy_config['search_strategy'],
        phased_optimization=phased_opt,
        tabu_radius=base_strategy_config['tabu_radius'],
        move_min_distance=base_strategy_config['move_min_distance']
    )
    
    ts = AgcspTS(
        instance=instance, 
        tenure=tenure, 
        strategy=strategy, 
        termination_criteria=termination_criteria, 
        debug_options=debug_options
    )

    try:
        start_time = time.time()
        
        initial_sol = ts._constructive_heuristic(strategy.constructive_heuristic)
        initial_obj = evaluator.objfun(initial_sol)
        (initial_cov_pen, initial_dist, initial_man) = evaluator.objfun_components(initial_sol)
        
        final_sol = ts.solve(initial_solution=initial_sol)
        
        exec_time = time.time() - start_time
        
        final_obj = evaluator.objfun(final_sol)
        (final_cov_pen, final_dist, final_man) = evaluator.objfun_components(final_sol)
        final_coverage_prop = evaluator.coverage_proportion(final_sol)
        
        plot_title = f"{instance_name}\nObj: {final_obj:.2f} | Coverage: {final_coverage_prop*100:.1f}%"
        plot_filepath = os.path.join(paths['results'], f"{instance_name}_final.jpg")
        save_plot_to_file(instance, evaluator, final_sol.path, plot_title, plot_filepath)
        
        print(f"[Concluído] {instance_name} | Obj: {final_obj:.2f} | Tempo: {exec_time:.2f}s")
        
        return {
            'Instance': instance_name,
            'Status': 'Success',
            'Stop Reason': ts.stop_reason,
            'Total Iters': ts._iters,
            'Exec Time (s)': exec_time,
            'Size': info['size'],
            'Obstacles': info['obstacle_type'],
            'Border': 'Yes' if info['border'] else 'No',
            'Sprayer': info['sprayer_length'],
            'Initial Obj': initial_obj,
            'Final Obj': final_obj,
            'Improvement (%)': (initial_obj - final_obj) / initial_obj * 100 if initial_obj > 0 else 0,
            'Final Coverage (%)': final_coverage_prop * 100,
            'Final Dist': final_dist,
            'Final Maneuver': final_man,
            'Plot': plot_filepath
        }

    except Exception as e:
        import traceback
        error_str = traceback.format_exc()
        print(f"[ERRO] {instance_name} falhou: {e}")
        return {'Instance': instance_name, 'Status': 'Error (Solve failed)', 'Error': error_str}

def main():
    print("Iniciando processo de batch (execução noturna)...")
    
    INSTANCES_DIR = 'agricultural_csp/instances'
    RESULTS_DIR = 'agricultural_csp/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    TERMINATION_CRITERIA = TerminationCriteria(max_time_secs=1800)
    
    DEBUG_OPTIONS = DebugOptions(verbose=False, log_history=False)
    
    TENURE_MAP = {
        'small': 10,
        'medium': 25,
        'large': 40
    }

    BASE_STRATEGY = {
        'constructive_heuristic': ConstructiveHeuristicType.FSM_COVERAGE_PLANNER,
        'search_strategy': 'first',
        'phased_optimization': {
            'iterations': [10, 10, 10],
            'tolerances': [0.0, 0.0, 0.0]
        },
        'tabu_radius': 3,
        'move_min_distance': 5
    }

    metadata_filepath = os.path.join(INSTANCES_DIR, 'instances_metadata.pkl')
    try:
        with open(metadata_filepath, 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de metadados não encontrado em {metadata_filepath}")
        return

    print(f"Metadados carregados. {len(metadata)} instâncias para processar.")
    
    tasks = []
    for instance_name, info in metadata.items():
        tenure = TENURE_MAP.get(info['size'], 10)
        
        task_config = {
            'instance_name': instance_name,
            'info': info,
            'paths': {'instances': INSTANCES_DIR, 'results': RESULTS_DIR},
            'strategy': BASE_STRATEGY,
            'termination_criteria': TERMINATION_CRITERIA,
            'debug_options': DEBUG_OPTIONS,
            'tenure': tenure
        }
        tasks.append(task_config)

    results_list = []
    start_batch_time = time.time()
    
    max_workers = max(1, os.cpu_count() - 1) 
    print(f"Iniciando ProcessPoolExecutor com {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_instance, task): task['instance_name'] for task in tasks}
        
        for future in as_completed(futures):
            instance_name = futures[future]
            try:
                result_data = future.result()
                results_list.append(result_data)
            except Exception as exc:
                print(f"Exceção no worker da instância {instance_name}: {exc}")
                results_list.append({'Instance': instance_name, 'Status': 'Error (Executor)', 'Error': str(exc)})

    end_batch_time = time.time()
    print("\n" + "="*80)
    print("Processamento em Lote Concluído")
    print(f"Tempo total: {(end_batch_time - start_batch_time) / 60:.2f} minutos.")
    
    results_df = pd.DataFrame(results_list)
    results_filepath = os.path.join(RESULTS_DIR, "batch_run_results.csv")
    results_df.to_csv(results_filepath, index=False)

    print(f"Resultados salvos em: {results_filepath}")
    print("\nResumo dos Resultados:")
    print(results_df[['Instance', 'Status', 'Exec Time (s)', 'Initial Obj', 'Final Obj', 'Final Coverage (%)']])

if __name__ == "__main__":
    main()