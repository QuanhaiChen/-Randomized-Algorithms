import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def importance_sampling(m, sigma, N=10000):
    samples = np.random.normal(m, sigma, N)
    f = (samples > 2).astype(float)
    
    p_x = stats.norm.pdf(samples, 0, 1)
    q_x = stats.norm.pdf(samples, m, sigma)
    w = p_x / q_x
    
    mu_std = np.mean(f * w)
    mu_sn = np.sum(f * w) / np.sum(w)
    
    return mu_std, mu_sn

# =============================================================================
# m = 4
# sigma = 0.5
# 
# var_f = importance_sampling_1(m, sigma)
# 
# print(f"m = {m}, sigma = {sigma}")
# print(f"Variance of the estimator: {var_f}")
# =============================================================================
    

def estimate_Z(N = 10000):
    samples = np.random.normal(0, 1, N)
    terms = np.sqrt(2 * np.pi) * np.exp(-np.abs(samples)**3 + (samples**2)/2)
    Z_hat = np.mean(terms)
    return Z_hat

# =============================================================================
# N = 10000000
# Z_hat = estimate_Z(N)
# print(f"Estimated Z: {Z_hat}")
# =============================================================================

# =============================================================================
# M = 1000
# N = 10000
# m_values = [0, 1, 2, 3, 4]
# sigma_values = [0.5, 1, 2, 3]
# 
# results = []
# for m in m_values:
#     for sigma in sigma_values:
#         var_std, var_sn = [], []
#         for i in range(M):
#             mu_std, mu_sn = importance_sampling(m, sigma, N)
#             var_std.append(mu_std)
#             var_sn.append(mu_sn)
#         var_std = np.var(var_std)
#         var_sn = np.var(var_sn)
#         results.append((m, sigma, var_std, var_sn))
# 
# for res in results:
#     print(res)
# =============================================================================


def systematic_gibbs(lattice, beta):
    L = lattice.shape[0]
    new_lattice = lattice.copy()
    for i in range(L):
        for j in range(L):
            top = new_lattice[(i-1) % L, j]
            bottom = new_lattice[(i+1) % L, j]
            left = new_lattice[i, (j-1) % L]
            right = new_lattice[i, (j+1) % L]
            S = top + bottom + left + right
            p = 1 / (1 + np.exp(-2 * beta * S))
            new_lattice[i, j] = 1 if np.random.rand() < p else -1
    return new_lattice

def random_gibbs(lattice, beta):
    L = lattice.shape[0]
    new_lattice = lattice.copy()
    indices = np.arange(L * L)
    np.random.shuffle(indices)
    for idx in indices:
        i, j = divmod(idx, L)
        top = new_lattice[(i-1) % L, j]
        bottom = new_lattice[(i+1) % L, j]
        left = new_lattice[i, (j-1) % L]
        right = new_lattice[i, (j+1) % L]
        S = top + bottom + left + right
        p = 1 / (1 + np.exp(-2 * beta * S))
        new_lattice[i, j] = 1 if np.random.rand() < p else -1
    return new_lattice

def simulate_ising(initial_lattice, T, N, mode):
    # kB = 1.380649e-23
    # beta = 1 / (kB * T)
    beta = 1 / T
    magnetization = []
    lattice = initial_lattice.copy()
    for _ in range(N):
        if mode == 1:
            lattice = systematic_gibbs(lattice, beta)
        elif mode == 2:
            lattice = random_gibbs(lattice, beta)
        magnetization.append(np.sum(lattice))
    return np.array(magnetization)

def integrated_autocorr_time(signal, max_lag=1000):
    n = len(signal)
    if n <= 1:
        return 0.0
    mean = np.mean(signal)
    var = np.var(signal)
    if var == 0:
        return 0.0 
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= n:
            acf[lag] = 0.0
            continue
        acov = np.mean((signal[:n-lag] - mean) * (signal[lag:] - mean))
        acf[lag] = acov / var
    cutoff = np.where(acf[1:] < 0)[0]
    cutoff = cutoff[0] + 1 if cutoff.size > 0 else max_lag
    return 1 + 2 * np.sum(acf[1:cutoff])

# =============================================================================
# L = 8
# T = 2
# N = [100000]
# iat_values = []
# burn_in = [int(n/2) for n in N]
# simulation_time = 1
# systematic_result = []
# random_result = []
# 
# for j in range(len(N)):
#     iat_systematic = []
#     iat_random = []
#     for i in range(simulation_time):
#         initial_lattice = np.random.choice([-1, 1], size=(L, L))
#         mag_systematic = simulate_ising(initial_lattice, T, N[j], 1)[burn_in[j]:]
#         mag_random = simulate_ising(initial_lattice, T, N[j], 2)[burn_in[j]:]
#         
#         iat_systematic.append(integrated_autocorr_time(mag_systematic, N[j]))
#         iat_random.append(integrated_autocorr_time(mag_random, N[j]))
#     
#     systematic_result.append(np.mean(iat_systematic))
#     random_result.append(np.mean(iat_random))
#     print(f"Temperature: {T}, L: {L}")
#     print(f"Systematic IAT: {np.mean(iat_systematic):.2f}")
#     print(f"Random IAT: {np.mean(iat_random):.2f}")
#     
#     plt.hist(mag_systematic, bins=50, alpha=0.5, label="Systematic")
#     plt.hist(mag_random, bins=50, alpha=0.5, label="Random")
#     plt.xlabel("Magnetization")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()
# =============================================================================


# =============================================================================
# plt.figure(figsize=(10, 5))
# plt.plot(N, systematic_result, marker='o')
# plt.xlabel('Length of Data Series')
# plt.ylabel('Integrated Autocorrelation Time')
# plt.title('Convergence of IAT Estimates')
# plt.grid(True)
# plt.show()
# 
# plt.figure(figsize=(10, 5))
# plt.plot(N, random_result, marker='o')
# plt.xlabel('Length of Data Series')
# plt.ylabel('Integrated Autocorrelation Time')
# plt.title('Convergence of IAT Estimates')
# plt.grid(True)
# plt.show()
# =============================================================================

# =============================================================================
# plt.hist(mag_systematic, bins=50, alpha=0.5, label="Systematic")
# plt.hist(mag_random, bins=50, alpha=0.5, label="Random")
# plt.xlabel("Magnetization")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()
# =============================================================================

def metropolis_step(lattice, beta):
    L = lattice.shape[0]
    i, j = np.random.randint(0, L, size=2)
    current_spin = lattice[i, j]
    top = lattice[(i-1) % L, j]
    bottom = lattice[(i+1) % L, j]
    left = lattice[i, (j-1) % L]
    right = lattice[i, (j+1) % L]
    neighbor_sum = top + bottom + left + right
    
    delta_E = 2 * current_spin * neighbor_sum
    if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
        lattice[i, j] *= -1  
    return lattice

def simulate_metropolis(initial_lattice, T, N, burn_in):
    beta = 1 / T
    lattice = initial_lattice.copy()
    magnetization = []
    
    for _ in range(burn_in):
        lattice = metropolis_step(lattice, beta)
    
    for _ in range(N):
        lattice = metropolis_step(lattice, beta)
        magnetization.append(np.sum(lattice))
    
    return np.array(magnetization)

# =============================================================================
# L = 8
# T = 1.5
# N = 500000
# burn_in = 10000
# 
# initial_lattice = np.random.choice([-1, 1], size=(L, L))
# mag_metropolis = simulate_metropolis(initial_lattice, T, N, burn_in)
# iat_metropolis = integrated_autocorr_time(mag_metropolis)
# 
# mag_gibbs = simulate_ising(initial_lattice, T, N, mode=2)[burn_in:]
# #iat_gibbs = integrated_autocorr_time(mag_gibbs)
# 
# print(f"Metropolis IAT: {iat_metropolis:.4f}")
# #print(f"Gibbs IAT: {iat_gibbs:.4f}")
# =============================================================================
