import matplotlib.pyplot as plt

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from application import Sender, Receiver0, Receiver1
from math import comb, ceil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

VALID_SINGLET_OUTCOMES = {"0011", "0101", "0110", "1010", "1001", "1100"}

def run_protocol(M, mu=0.272, lambda_=0.94, p0=0.05, p1=0.005):
    cfg = StackNetworkConfig.from_file("config.yaml")

    programs = {
        "S": Sender(M=M, p0=p0, p1=p1),
        "R0": Receiver0("R0", M=M, mu=mu, p0=p0, p1=p1),
        "R1": Receiver1("R1", M=M, mu=mu, lambda_=lambda_, p0=p0, p1=p1),
    }

    result = run(
        config=cfg,
        programs=programs,
        num_times=1,
    )

    sender = result[0][0]
    r0_result = result[1][0]
    r1_result = result[2][0]

    sender_measurements = sender.get("measurements")
    r0_measurements = r0_result.get("measurements")
    r1_measurements = r1_result.get("measurements")

    min_len = min(len(sender_measurements), len(r0_measurements), len(r1_measurements))
    if min_len < M:
        print(
            f"WARNING: Short measurement lists (S: {len(sender_measurements)}, R0: {len(r0_measurements)}, R1: {len(r1_measurements)}). Using {min_len} rounds.")
        return {}

    num_leakage = 0
    for i in range(M):
        full_measurement = (
                str(sender_measurements[i][0]) +
                str(sender_measurements[i][1]) +
                str(r0_measurements[i]) +
                str(r1_measurements[i])
        )
        if full_measurement not in VALID_SINGLET_OUTCOMES:
            num_leakage += 1

    q = num_leakage / M

    return r0_result.get("accepted"), r1_result.get("final_decision"), q, sender.get("xS")

def run_single_trial(args):
    M, mu, lambda_, p0, p1 = args
    return run_protocol(M, mu=mu, lambda_=lambda_, p0=p0, p1=p1)

def simulate_failure_probability():
    M_values = list(range(20, 401, 20))
    num_trials = 100
    mu = 0.272
    lambda_ = 0.94
    p0 = 0.05
    p1 = 0.005

    failure_rates = []
    exact_rates = []
    mc_errors = []
    noisy_rates = []
    avg_qs = []
    theoretical_failure_probs = []

    for M in M_values:
        print("M: ", M)
        failures = 0
        leakage_qs = []

        args = [(M, mu, lambda_, p0, p1)] * num_trials

        # Use all available CPU cores
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
            results = list(executor.map(run_single_trial, args))

        for y0, y1, q, xS in results:
            leakage_qs.append(q)
            if y0 is None or y1 is None or y0 != y1:
                failures += 1

        pf_mc = failures / num_trials
        pf_exact = exact_failure_probability_no_noise(M, mu=mu)

        pf_noisy_nalytical = analytical_noisy_failure_probability(M, lambda_, mu)

        theoretical_failure_probs.append(pf_noisy_nalytical)

        avg_q = sum(leakage_qs) / len(leakage_qs)
        avg_qs.append(avg_q)
        P = 1 - (1 - avg_q) ** M
        pf_noisy = (1 - P) * pf_exact + P

        noisy_rates.append(pf_noisy)
        failure_rates.append(pf_mc)
        exact_rates.append(pf_exact)

        se = (pf_mc * (1 - pf_mc) / num_trials) ** 0.5
        mc_errors.append(se)


    # Plot 1: Failure probability vs M
    plt.figure(figsize=(8, 5))
    plt.errorbar(M_values, failure_rates, yerr=mc_errors, fmt='x', color='red')
    plt.plot(M_values, failure_rates, marker='x', color='red', linestyle='None', label="Monte Carlo")
    plt.plot(M_values, exact_rates, marker='o', color='green', linestyle='None', label="Exact", fillstyle='none')
    plt.plot(M_values, noisy_rates, marker='s', color='blue', linestyle='None', label="Noisy Model", fillstyle='none')
    plt.plot(M_values, theoretical_failure_probs, marker='^', color='purple', linestyle='None', label="Bit-flip Theoretical Model", fillstyle='none')
    plt.xlabel("Number of four-qubit singlet states (m)")
    plt.ylabel("Failure probability")
    plt.title(f"Bit-flip Model Failure Probability (No Faulty), N = {num_trials}, p0 = {p0}, p1 = {p1}")
    max_y = max(max(failure_rates), max(noisy_rates), max(exact_rates)) + 0.05
    plt.ylim(0, max(1.0, max_y))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    plt.show()


def exact_failure_probability_no_noise(m: int, mu: float = 0.272) -> float:
    """
        Computes the exact failure probability for the no faulty configuration using the
        binomial distribution formula given in Eq. (25) of Appendix C.1 of the paper.
    """
    T = ceil(mu * m)
    prob = 0.0
    for k in range(T):
        binom = comb(m, k)
        term = binom * (1/3)**k * (2/3)**(m - k)
        prob += term
    return prob


def noisy_failure_length_cond_m0011(m: int, mu: float = 0.272, p0: float = 0.05, p1:float = 0.005) -> float:
    """
        probability that the length condition is not satisfied for the state m = 0011
    """
    T = ceil(mu * m)
    prob = 0.0
    p0011 = 1/3 * (1-p0)**2 * (1-p1)**2 + 1/3*(1-p0)*p0*(1-p1)*p1 + 1/3*p0**2 * p1**2
    for k in range(T):
        binom = comb(m, k)
        term = binom * p0011 ** k * (1 - p0011) ** (m - k)
        prob += term
    return prob

def noisy_failure_length_cond_m1100(m: int, mu: float = 0.272, p0: float = 0.05, p1:float = 0.005) -> float:
    """
        probability that the length condition is not satisfied for the state m = 1100
    """
    T = ceil(mu * m)
    prob = 0.0
    p1100 = 1/3 * (1-p1)**2 * (1-p0)**2 + 1/3*(1-p0)*p0*(1-p1)*p1 + 1/3*p0**2 * p1**2
    for k in range(T):
        binom = comb(m, k)
        term = binom * p1100 ** k * (1 - p1100) ** (m - k)
        prob += term
    return prob

def compute_q_m0011(p0, p1):
    """
        probability that state m = 0011 leaks into a harmful state
    """
    term1 = 1/3 * (1 - p0) ** 2 * (1 - (1 - p1) ** 2)
    term2 = 1/12 * 4 * (1 - p0) * p1 * (1 - p0 * (1-p1))
    term3 = 1/3 * p1 ** 2 * (1 - p0**2)

    return term1 + term2 + term3

def compute_q_m1100(p0, p1):
    """
        probability that state m = 1100 leaks into a harmful state
    """
    term1 = 1/3 * (1 - p1) ** 2 * (1 - (1 - p0) ** 2)
    term2 = 1/12 * 4 * (1 - p1) * p0 * (1 - p1 * (1-p0))
    term3 = 1/3 * p0 ** 2 * (1 - p1**2)

    return term1 + term2 + term3

def leakage_probability_m0011(p0, p1, m, lambda_, mu):
    """
        probability that at least one leakage happens for m=0011 - violating consistency condition
    """
    q0011 = compute_q_m0011(p0, p1)
    return 1 - (1 - q0011) ** m

def leakage_probability_m1100(p0, p1, m, lambda_, mu):
    """
        probability that at least one leakage happens for m=1100 - violating consistency condition
    """
    q1100 = compute_q_m1100(p0, p1)
    return 1 - (1 - q1100) ** m

def analytical_noisy_failure_probability(m, lambda_, mu):
    """
        failure probability of the protocol under noise
        calculated based on the weighted average of selected state being m = 0011 or m = 1100
    """
    p0 = 0.05
    p1 = 0.005

    lT_m0011 = noisy_failure_length_cond_m0011(m, mu=mu, p0=p0, p1=p1)
    pf_noisy_theoretical_m0011 = lT_m0011 + (1 - lT_m0011) * leakage_probability_m0011(p0, p1, m, lambda_, mu)

    lT_m1100 = noisy_failure_length_cond_m1100(m, mu=mu, p0=p0, p1=p1)
    pf_noisy_theoretical_m1100 = lT_m1100 + (1 - lT_m1100) * leakage_probability_m1100(p0, p1, m, lambda_, mu)

    pf_m0011 = 1 / 2 * pf_noisy_theoretical_m0011
    pf_m1100 = 1 / 2 * pf_noisy_theoretical_m1100

    return pf_m0011 + pf_m1100

if __name__ == "__main__":
    simulate_failure_probability()
