import numpy as np
import matplotlib.pyplot as plt

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

def simulate_ising(initial_lattice, T, N, mode, thin=1000):
    beta = 1 / T
    magnetization = []
    lattice = initial_lattice.copy()
    for step in range(N):
        if mode == 1:
            lattice = systematic_gibbs(lattice, beta)
        elif mode == 2:
            lattice = random_gibbs(lattice, beta)
        if step % thin == 0:  
            magnetization.append(np.sum(lattice))
    return np.array(magnetization)

def integrated_autocorr_time(signal, max_lag_fraction=0.1):
    n = len(signal)
    if n <= 1:
        return 0.0
    mean = np.mean(signal)
    var = np.var(signal)
    if var == 0:
        return 0.0
    max_lag = int(n * max_lag_fraction)
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

L = 6
T = 2
N = [1000000]
thin = 100  
burn_in = [int(n / 2) for n in N]
simulation_time = 1
systematic_result = []
random_result = []

for j in range(len(N)):
    iat_systematic = []
    iat_random = []
    for i in range(simulation_time):
        initial_lattice = np.random.choice([-1, 1], size=(L, L))
        mag_systematic = simulate_ising(initial_lattice, T, N[j], 1, thin)
        mag_random = simulate_ising(initial_lattice, T, N[j], 2, thin)
        burn_in_samples = burn_in[j] // thin
        mag_systematic = mag_systematic[burn_in_samples:]
        mag_random = mag_random[burn_in_samples:]
        iat_systematic.append(integrated_autocorr_time(mag_systematic))
        iat_random.append(integrated_autocorr_time(mag_random))
    systematic_result.append(np.mean(iat_systematic))
    random_result.append(np.mean(iat_random))
    print(f"Temperature: {T}, L: {L}")
    print(f"Systematic IAT: {np.mean(iat_systematic):.4f}")
    print(f"Random IAT: {np.mean(iat_random):.4f}")
    plt.hist(mag_systematic, bins=50, alpha=0.5, label="Systematic")
    plt.hist(mag_random, bins=50, alpha=0.5, label="Random")
    plt.xlabel("Magnetization")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()