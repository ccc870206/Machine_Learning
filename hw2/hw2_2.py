import math


def load_data():
    with open('testfile.txt', 'r') as f:
        arr = []
        for line in f.readlines():
            arr.append(line.strip())

    return arr, int(input("a:")), int(input("b:"))


def beta_distribution(p, a, b):
    return pow(p, a - 1) * pow(1 - p, b - 1) * math.gamma(a + b) / (math.gamma(a) * math.gamma(b))


def Cmn(m, n):
    return math.factorial(m) // math.factorial(m-n) // math.factorial(n)


def binomial_distribution(x, N, p):
    return Cmn(N, x)*pow(p, x)*pow(1-p, N-x)


def compute_posterior(outcome, a, b):
    m = sum([int(i) for i in outcome if i == '1'])
    N = len(outcome)
    p = m/N
    likelihood = binomial_distribution(m, N, p)
    posterior_a = m + a
    posterior_b = N - m + b
    posterior = beta_distribution(p, posterior_a, posterior_b)
    return p, posterior_a, posterior_b, likelihood


def online_learning():
    inputs, a, b = load_data()
    prior_a = a
    prior_b = b
    for i in range(len(inputs)):
        print('case', str(i+1)+':', inputs[i])
        p, posterior_a, posterior_b, likelihood = compute_posterior(inputs[i], prior_a, prior_b)
        print('Likelihood:', likelihood)
        print('Beta prior:      a =', str(prior_a), 'b =', str(prior_b))
        print('Beta posterior:  a =', str(posterior_a), 'b =', str(posterior_b))
        print()
        prior_a = posterior_a
        prior_b = posterior_b


if __name__ == "__main__":
    online_learning()


