import random
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics

# Carregamento do arquivo CNF
def load_cnf_file(filename):
    clauses = []
    num_variables = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('%') or line.startswith('0'):
                continue
            elif line.startswith('p cnf'):
                parts = line.split()
                num_variables = int(parts[2])
            elif line.strip() != "":
                clause = list(map(int, line.split()))
                if clause[-1] == 0: clause.pop()
                clauses.append(clause)
    return num_variables, clauses

# Funções Base e Índices Invertidos
def generate_initial_solution(num_variables):
    return [random.choice([True, False]) for _ in range(num_variables)]

def build_inverted_indices(num_variables, clauses):
    pos_map = [[] for _ in range(num_variables)]
    neg_map = [[] for _ in range(num_variables)]
    for c_idx, clause in enumerate(clauses):
        for literal in clause:
            var_idx = abs(literal) - 1
            if literal > 0:
                pos_map[var_idx].append(c_idx)
            else:
                neg_map[var_idx].append(c_idx)
    return pos_map, neg_map

def initialize_state(solution, clauses):
    clause_sat_count = [0] * len(clauses)
    energy = 0
    for c_idx, clause in enumerate(clauses):
        sat_literals = 0
        for literal in clause:
            var_idx = abs(literal) - 1
            is_true = solution[var_idx]
            if (literal > 0 and is_true) or (literal < 0 and not is_true):
                sat_literals += 1
        clause_sat_count[c_idx] = sat_literals
        if sat_literals == 0:
            energy += 1 
    return clause_sat_count, energy

def flip_variable(var_idx, solution, clause_sat_count, energy, pos_map, neg_map):
    is_true_now = not solution[var_idx]
    solution[var_idx] = is_true_now 
    
    gains = pos_map[var_idx] if is_true_now else neg_map[var_idx]
    losses = neg_map[var_idx] if is_true_now else pos_map[var_idx]
    
    for c_idx in gains:
        if clause_sat_count[c_idx] == 0:
            energy -= 1 
        clause_sat_count[c_idx] += 1
        
    for c_idx in losses:
        clause_sat_count[c_idx] -= 1
        if clause_sat_count[c_idx] == 0:
            energy += 1 
            
    return energy

# Equações de Resfriamento escolhidas
def cooling_custom(it, itMax, t_factor=3):
    T = (1 - (it / itMax)) ** t_factor
    return max(T, 0.000001)

def cooling_schedule_1(it, itMax, T0=100, TN=0.01):
    if itMax == 0: return T0
    return T0 * ((TN / T0) ** (it / itMax))

# Algoritmo Simulated Annealing
def simulated_annealing(clauses, num_variables, cooling_func, thermal_equilibrium, max_evals=200000): #maxIterações!!!!!!!!
    pos_map, neg_map = build_inverted_indices(num_variables, clauses)
    current_sol = generate_initial_solution(num_variables)
    clause_sat_count, current_energy = initialize_state(current_sol, clauses)
    
    # SALVANDO A ENERGIA INICIAL
    initial_energy = current_energy 
    best_sol, best_energy = current_sol[:], current_energy
    
    num_bits_to_flip = max(1, int(num_variables * 0.05))
    
    evals = 1
    it = 0
    itMax = max_evals // thermal_equilibrium
    history = []

    while evals < max_evals:
        temp = cooling_func(it, itMax)
        
        for _ in range(thermal_equilibrium):
            if evals >= max_evals: break
            
            indices_to_flip = random.sample(range(num_variables), num_bits_to_flip)
            n_energy = current_energy
            
            for idx in indices_to_flip:
                n_energy = flip_variable(idx, current_sol, clause_sat_count, n_energy, pos_map, neg_map)
                
            evals += 1
            delta = n_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / temp if temp > 0 else -1e10):
                current_energy = n_energy
                if current_energy < best_energy:
                    best_sol, best_energy = current_sol[:], current_energy
            else:
                for idx in indices_to_flip:
                    _ = flip_variable(idx, current_sol, clause_sat_count, current_energy, pos_map, neg_map)
            
            history.append(current_energy)
            
            if best_energy == 0:
                break
                
        it += 1
        if best_energy == 0:
            break
            
    # RETORNANDO A ENERGIA INICIAL JUNTO COM OS OUTROS DADOS
    return best_sol, best_energy, history, initial_energy

# Protocolo de Experimentos
def run_experiments():
    files = ['uf20-01.cnf', 'uf100-01.cnf','uf250-01.cnf'] # Altere para o arquivo que deseja testar
    configs_eq = [1, 100] #parametro de equilibrio termico
    funcs = {"Custom_Formula": cooling_custom, "Schedule_1_PDF": cooling_schedule_1}
    
    for filename in files:
        print(f"\n{'='*60}\nARQUIVO: {filename}\n{'='*60}")
        try:
            num_vars, clauses = load_cnf_file(filename)
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Pulando...")
            continue

        for eq in configs_eq:
            for f_name, f_logic in funcs.items():
                print(f"\n>> Config: {f_name} | Equilíbrio: {eq}")
                
                final_results = []
                histories = []

                for i in range(10):
                    # RECEBENDO A ENERGIA INICIAL DA FUNÇÃO
                    _, final_energy, hist, initial_energy = simulated_annealing(clauses, num_vars, f_logic, eq)
                    
                    final_results.append(final_energy)
                    histories.append(hist)
                    
                    # NOVO PRINT COM AS DUAS INFORMAÇÕES LADO A LADO
                    print(f"  Exec {i+1:02d}: FO Inicial = {initial_energy:03d} | FO Final (Melhor) = {final_energy:03d}")

                mean_fo = statistics.mean(final_results)
                std_fo = statistics.stdev(final_results) if len(final_results) > 1 else 0
                print(f"  Estatísticas da FO Final -> MÉDIA: {mean_fo:.2f} | DESVIO-PADRÃO: {std_fo:.2f}")

                plt.figure(figsize=(8, 4))
                plt.plot(histories[0])
                plt.title(f"Convergência {filename}\n({f_name}, eq={eq})")
                plt.xlabel("Avaliações")
                plt.ylabel("Cláusulas não satisfeitas")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"plot_conv_{filename}_{f_name}_eq{eq}.png")
                plt.close()

                plt.figure(figsize=(5, 4))
                plt.boxplot(final_results)
                plt.title(f"Boxplot {filename}\n({f_name}, eq={eq})")
                plt.ylabel("FO Final")
                plt.tight_layout()
                plt.savefig(f"boxplot_{filename}_{f_name}_eq{eq}.png")
                plt.close()

if __name__ == "__main__":
    run_experiments()