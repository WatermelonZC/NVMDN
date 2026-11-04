

class ALNSConfig:
    destroy_count = 10       # Fixed count, capped by the instance task count.
    initial_deterioration = 0.02
    initial_acceptance = 0.5
    final_temperature_ratio = 0.01
    # Compatibility for legacy helpers; the current ALNS does not use these.
    d_num_coefficient = 0.3
    T_coefficient = 1
    rho = 0.1                # reaction factor
    l_s = 5                  # update interval
    sigma_1 = 33             # score 1
    sigma_2 = 13             # score 2
    sigma_3 = 9              # score 3
    shaw_weight = 0.3        # shaw coefficient

    C = 10                   # computation time
