import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

"""
There are only functions are not commented. Every calculation for every part of the task is commented, 
so it is convenient to calculate them one by one. Otherwise it would be graphs and numbers spam
"""


def generate_histogram(data, bins=10):
    n = len(data)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_width = bin_edges[1] - bin_edges[0]  # can just use 1/bins, but in such way more universal
    normalized_counts = counts / n / bin_width  # also can just density=True, but need not normalized for errors
    errors = np.sqrt((n * counts - counts ** 2) / (n ** 3))
    errors = errors * n * bin_width

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, normalized_counts, width=bin_width, alpha=0.6, label=f'N={n}')
    plt.errorbar(bin_centers, normalized_counts, yerr=errors, fmt='.', color='red', label='Error bars')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Density Fluctuations with Error Bars')
    plt.legend()
    plt.show()


def sample_fluctuations(n, bin_index=5, samples=1000, bins=10):

    bin_heights = []
    bin_heights_normalized = []

    for _ in range(samples):
        data = np.random.uniform(0, 1, n)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_heights.append(counts[bin_index])
        bin_width = bin_edges[1] - bin_edges[0]
        normalized_counts = counts / n / bin_width
        bin_heights_normalized.append(normalized_counts[bin_index])

    normalized_counts, bin_edges = np.histogram(bin_heights_normalized, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expectation_value = np.sum(bin_centers * normalized_counts * bin_width)
    print(f'expectation_value: {expectation_value}', f'N_mean_squared = {(np.mean(bin_heights)) ** 2}',
          f'N_squared_mean = {np.mean(np.array(bin_heights)**2)}')

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, normalized_counts, width=bin_width, alpha=0.7)
#    plt.hist(bin_heights, bins=bins, density=True, alpha=0.7, color='blue')
    plt.xlabel('Bin Height (0.5 < x < 0.6)')
    plt.ylabel('Probability Density')
    plt.title(f'Fluctuations in Histogram Bin (N={n}, M={samples})')
    plt.show()


def calculate_compressibility(n, n_mean_squared, n_squared_mean, volume=0.1):
    isothermal_compressibility = volume * (n_squared_mean - n_mean_squared) / n_mean_squared
    expected_compressibility = 1 / n
    print(isothermal_compressibility, expected_compressibility)
    return isothermal_compressibility, expected_compressibility


def inverse_transform_sampling():

    uniform_samples = np.random.uniform(0, 1)
    exponential_samples = -np.log(uniform_samples)

    return exponential_samples


def gaussian_distribution(x):
    if x >= 0:
        f = np.sqrt(2 / np.pi) * np.exp(-x ** 2 / 2)
    else:
        print('Incorrect x')
        return 1
    return f


def enveloping_function(x):
    if x >= 0:
        g = np.exp(-x)
    else:
        print('Incorrect x')
        return 1
    return g


def new_enveloping_function(x):
    if 0 <= x <= 1:
        g = 1 / 2
    elif x > 1:
        g = 1 / 2 / x ** 2
    else:
        print('Incorrect x')
        return 1
    return g


def new_inverse_transform_sampling():

    uniform_sample = np.random.uniform(0, 1)
    if uniform_sample <= 0.5:
        inverse_sample = 2 * uniform_sample
    else:
        inverse_sample = 1 / 2 / (1 - uniform_sample)

    return inverse_sample


def rejection_sampling(n):
    iterations_number = 0
    x_set = []
    arg_max = opt.minimize(lambda x: -gaussian_distribution(x) / enveloping_function(x), x0=0.5, method='BFGS')
    # actually can be solved analytically, but I decided to do it this way
    arg_max = arg_max.x
    env_const = gaussian_distribution(arg_max) / enveloping_function(arg_max)

    while len(x_set) <= n:
        env_dis_number = inverse_transform_sampling()
        uniform_number = np.random.uniform(0, 1)
        if uniform_number <= gaussian_distribution(env_dis_number) / env_const \
                / enveloping_function(env_dis_number):
            x_set.append((-1) ** np.random.randint(2) * env_dis_number)
        iterations_number += 1

    acceptance_rate = n / iterations_number
    return np.array(x_set), acceptance_rate


def new_rejection_sampling(n):
    iterations_number = 0
    x_set = []
    arg_max = 0
    """arg_max = opt.minimize(lambda x: -gaussian_distribution(x) / new_enveloping_function(x), x0=2, method='BFGS')
    arg_max = arg_max.x
    print(arg_max)
    
    This method also works for my envelope function, but it gives a few complains in the process, so I decided to solve 
    it analytically after all"""
    env_const = gaussian_distribution(arg_max) / new_enveloping_function(arg_max)

    while len(x_set) <= n:
        env_dis_number = new_inverse_transform_sampling()
        uniform_number = np.random.uniform(0, 1)
        if uniform_number <= gaussian_distribution(env_dis_number) / env_const \
                / new_enveloping_function(env_dis_number):
            x_set.append((-1) ** np.random.randint(2) * env_dis_number)
        iterations_number += 1

    acceptance_rate = n / iterations_number
    return np.array(x_set), acceptance_rate


def bayesian_analysis(n, num_bins=50):
    data = []

    for _ in range(n):
        data.append(inverse_transform_sampling())

    counts, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]

    pi = (counts + 1) / (n + num_bins + 1)
    lambda_pi = np.sqrt(1 / (n + num_bins + 2) * pi * (1 - pi))
    counts = 1 / bin_width * pi
    lambda_hi = 1 / bin_width * lambda_pi

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), color='C0', alpha=0.6, label='Posterior Mean')
    plt.errorbar(bin_centers, counts, yerr=lambda_hi, fmt='None', color='black', label='68% CI')
    plt.xlabel('Data Value')
    plt.ylabel('Count')
    plt.title('Bayesian Histogram with Error Bars')
    plt.legend()
    plt.show()


def calculate_velocity_magnitude(n):
    v1, _ = rejection_sampling(n)
    v2, _ = rejection_sampling(n)
    v3, _ = rejection_sampling(n)
    v = np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2)

    print("Mean values: ", np.mean(v1), np.mean(v2), np.mean(v3))
    print("Std deviations: ", np.std(v1), np.std(v2), np.std(v3))
    return v


"generate_histogram(data = np.random.uniform(0, 1, 10**2))"
"""sample_fluctuations(n=10**3)
sample_fluctuations(n=10**5)"""

"""calculate_compressibility(10**3, 9871.813449, 9955.811)
calculate_compressibility(10**5, 99993280.11289601, 100002900.74)"""

"""x_set, acceptance_rate = rejection_sampling(10 ** 4)
x_set_new, new_acceptance_rate = new_rejection_sampling(10 ** 4)

plt.figure(figsize=(8, 6))
plt.hist(x_set, bins=50, density=True, alpha=0.7, color='blue')
plt.xlabel('Velocity distribution')
plt.ylabel('Probability Density')
plt.title(f'Velocity distribution')
plt.show()
print(f'Old: {acceptance_rate}')

plt.figure(figsize=(8, 6))
plt.hist(x_set_new, bins=50, density=True, alpha=0.7, color='blue')
plt.xlabel('Velocity distribution')
plt.ylabel('Probability Density')
plt.title(f'Velocity distribution')
plt.show()
print(f'New: {new_acceptance_rate}')"""

velocity_magnitude = calculate_velocity_magnitude(10**5)

plt.figure(figsize=(8, 6))
plt.hist(velocity_magnitude, bins=100, density=True, alpha=0.7, color='blue', label='Simulated PDF')
v = np.linspace(0, np.max(velocity_magnitude), 1000)

sigma = 1
mb_pdf = (v**2 / sigma**3) * np.exp(-v**2 / (2 * sigma**2)) * np.sqrt(2 / np.pi)
plt.plot(v, mb_pdf, 'r-', label='Maxwell-Boltzmann PDF')
plt.xlabel('Velocity distribution')
plt.ylabel('Probability Density')
plt.title(f'Velocity distribution')
plt.show()

generate_histogram(velocity_magnitude, 100)

exponential_distribution = []
for i in range(10**4):
    exponential_distribution.append(inverse_transform_sampling())

velocity_set, _ = rejection_sampling(10**4)
new_velocity_set, _ = new_rejection_sampling(10**4)
generate_histogram(exponential_distribution, 100)
generate_histogram(velocity_set, 100)
generate_histogram(new_velocity_set, 100)

bayesian_analysis(10**4, 100)
