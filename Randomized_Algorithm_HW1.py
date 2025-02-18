import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
import time

# takes input N to generate a sample with pi(x) = e^-x
def generate_xbar(N):
    temp = np.random.exponential(1, N)
    return np.mean(temp)
    
def xbar_histogram(N, simulation_times = 1000):
    deviations = []
    for i in range(simulation_times):
        xbar = generate_xbar(N)
        deviations.append(np.sqrt(N)*(xbar-1))

    plt.hist(deviations, bins=100)
    plt.title(f'Histogram of $\\sqrt{{N}}(\\bar{{x}}_N-1$ for N={N}')
    plt.xlabel('$\\sqrt{N}(\\bar{X}_N - 1$')
    plt.ylabel('Frequency')
    plt.show()
    
    sm.qqplot(np.array(deviations), line = '45')
    plt.title(f'QQ Plot for $N={N}$')
    plt.show()
    

def xbar_histograms(N_values = [10, 100, 1000, 5000, 10000]):
    simulation_times = 1000
    for N in N_values:
        xbar_histogram(N, simulation_times)
    
def estimate_Q(N, simulation_times = 10000, threshold = 0.1):
    count = 0
    for i in range(simulation_times):
        xbar = generate_xbar(N)
        if xbar - 1 > threshold:
            count += 1
    QN = count/simulation_times
    print(f"Estimated probability Q_N for N={N} is: {QN:.6f}")
    
def improve_xbar(N):
    return np.random.gamma(N, 1)/N

def estimate_improve_Q(N, simulation_times = 10000, threshold = 0.1):
    count = 0
    for i in range(simulation_times):
        xbar = improve_xbar(N)
        if xbar - 1 > threshold:
            count += 1
    QN = count/simulation_times
    return QN
    
def decay_plot(N_values = np.geomspace(10, 1000, num=100, dtype=int), threshold=0.1):
    p_N_values = [estimate_improve_Q(N, threshold=threshold) for N in N_values]
    decay_rates = [1/N * np.log(p_N) for N, p_N in zip(N_values, p_N_values)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, decay_rates, marker='o')
    plt.xscale('log')
    plt.xlabel('Sample Size N')
    plt.ylabel(r'$\frac{1}{N} \log p_N$')
    plt.title('Decay of Probability as N Increases')
    plt.grid(True)
    temp = np.log(1+threshold)-threshold
    plt.plot([N_values[0], N_values[-1]], [temp, temp], linestyle='--', color='r')
    #temp2 = -(threshold**2)/2
    #plt.plot([N_values[0], N_values[-1]], [temp2, temp2], linestyle='--', color='g')
    plt.show()
    
def pNvarQN():
    N = [10, 50, 100, 500, 1000, 5000]
    pN = (0.1-np.log(1.1))**N
    varQN = np.sqrt(pN*(1-pN))/N
    print(pN)
    print(varQN)
    plt.figure(figsize=(10, 6))
    plt.plot(N, pN, marker='o', label='p_N')
    plt.plot(N, varQN, marker='o', label='var(Q_N)')
    plt.xlabel('Sample Size N')
    plt.title('pN and standard deviation of QN')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def inversion(N = 10000):
    uniform_samples = np.random.uniform(0, 1, N)
    pi_inversion_samples = uniform_samples**2
    
    plt.figure(figsize=(10, 6))
    plt.hist(pi_inversion_samples, bins=50, alpha=0.75, density=True)
    plt.title('Histogram of Generated Samples from $\\pi(x) = \\frac{1}{2\\sqrt{x}}$')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()
    
    pi_inversion_samples.sort()
    
    plt.figure(figsize=(8,8))
    plt.scatter(np.linspace(0, 1, N)**2, pi_inversion_samples, alpha=0.6)
    plt.plot([0,1], [0,1], 'r--')
    plt.title('QQ Plot Comparing Sampled Data Against $\\pi(x)$')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical Quantiles')
    plt.grid(True)
    plt.show()
    
def twoGaussian(N):
    u1 = np.random.uniform(0, 1, N)
    u2 = np.random.uniform(0, 1, N)
    R = np.sqrt(-2*np.log(u1))
    theta = 2*np.pi*u2
    return R*np.cos(theta), R*np.sin(theta)

def twoGaussianHist(N):
    g1, g2 = twoGaussian(N)
    plt.hist2d(g1,g2, bins=50, cmap='viridis')
    plt.colorbar()
    plt.grid(True)
    plt.show()

def qqplotGaussian(N):
    g1, g2 = twoGaussian(N)
    
    scipy.stats.probplot(g1, dist="norm", plot=plt)
    
    plt.show()
    
    scipy.stats.probplot(g2, dist="norm", plot=plt)
    
    plt.show()

def uniformDisk(N):
    u1 = np.random.uniform(0, 1, N)
    u2 = np.random.uniform(0, 1, N)
    R = np.sqrt(u1)
    theta = 2*np.pi*u2
    g1, g2 = R*np.cos(theta), R*np.sin(theta)
    
# =============================================================================
#     plt.figure(figsize=(10,8))
#     plt.hexbin(g1, g2, gridsize=50, cmap='viridis', extent=(-1,1,-1,1))
#     plt.hist2d(g1, g2, bins=50, range=[[-1,1],[-1,1]])
#     plt.colorbar()
#     plt.title('2D Histogram of Uniform Samples on the Unit Disk')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.axis('equal')
#     plt.show()
# =============================================================================
    return g1, g2, 2*N

def uniformDiskRejection():
    i = 0
    while i < 10000:
        u1 = 2*np.random.uniform(0,1)-1
        u2 = 2*np.random.uniform(0,1)-1
        
        if u1**2 + u2**2 <= 1:
            return u1, u2
        
        i += 1
    raise ValueError('Didn\'t find the rejection')

def sqrt2CoverSquare():
    fig, ax = plt.subplots()
    points = 1000
    theta = np.linspace(0, 2 * np.pi, points)

    x1 = np.cos(theta)
    y1 = np.sin(theta)

    x2 = np.sqrt(2) * np.cos(theta)
    y2 = np.sqrt(2) * np.sin(theta)

    ax.plot(x1, y1, label='Unit Circle (Radius = 1)')

    ax.plot(x2, y2, 'r--', label='Scaled Circle (Radius = $\\sqrt{2}$)')

    square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='green', label='Bounding Square')
    ax.add_patch(square)

    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.grid(True)

    plt.title("Unit Circle and Scaled Circle Covering Bounding Square")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    
def naturalChoice(N):
    result_u1 = []
    result_u2 = []
    acceptedK = 1/np.sqrt(2)
    uniformCalled = 0
    while len(result_u1) < N:
        u1 = 2 * np.random.uniform(0, 1) - 1
        u2 = 2 * np.random.uniform(0, 1) - 1
        uniformCalled += 2
        
        if u1**2 + u2**2 <= 1:
            if np.random.rand() < acceptedK:
                result_u1.append(u1)
                result_u2.append(u2)
    
    
# =============================================================================
#     plt.figure(figsize=(10,8))
#     plt.hist2d(result_u1, result_u2, bins=50, cmap='viridis')
#     plt.colorbar()
#     plt.title("2D Histogram of Points Within the Scaled Circle")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.axis('equal')
#     plt.grid(True)
#     plt.show()
# =============================================================================
    
    return result_u1, result_u2, uniformCalled
    

N = 100000

startTime = time.perf_counter()
_, __, rejectionUniform = naturalChoice(N)
endTime = time.perf_counter()
print(f"Rejection Sampling: {endTime-startTime} seconds, {rejectionUniform} uniform calls")
    
startTime = time.perf_counter()
_, __, directUniform = uniformDisk(N)
endTime = time.perf_counter()
print(f"Direct Sampling: {endTime-startTime} seconds, {directUniform} Uniform calls")

    
    
    
    