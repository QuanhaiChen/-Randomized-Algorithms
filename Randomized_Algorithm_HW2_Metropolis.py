import numpy as np

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

def simulate_metropolis(initial_lattice, T, N, burn_in, thin=1000):
    beta = 1 / T
    lattice = initial_lattice.copy()
    magnetization = []
    
    for _ in range(burn_in):
        lattice = metropolis_step(lattice, beta)
    
    for step in range(N):
        lattice = metropolis_step(lattice, beta)
        if step % thin == 0:  
            magnetization.append(np.sum(lattice))
    
    return np.array(magnetization)

L = 4
T = 1.5
N = 10000000
thin = 1000
burn_in = 10000

initial_lattice = np.random.choice([-1, 1], size=(L, L))
mag_metropolis = simulate_metropolis(initial_lattice, T, N, burn_in, thin)
iat_metropolis = integrated_autocorr_time(mag_metropolis)

#mag_gibbs = simulate_ising(initial_lattice, T, N, mode=2)[burn_in:]
#iat_gibbs = integrated_autocorr_time(mag_gibbs)

print(f"Temperature: {T}, L: {L}")
print(f"Metropolis IAT: {iat_metropolis:.4f}")
#print(f"Gibbs IAT: {iat_gibbs:.4f}")