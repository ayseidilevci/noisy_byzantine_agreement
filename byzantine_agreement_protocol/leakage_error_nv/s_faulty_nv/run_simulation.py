from os import stat_result

import matplotlib.pyplot as plt
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from application import FaultySender, Receiver0, Receiver1
from math import comb, ceil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

VALID_SINGLET_OUTCOMES = {"0011", "0101", "0110", "1010", "1001", "1100"}

def run_protocol(M, mu=0.272, lambda_=0.94, p=0.01):
    cfg = StackNetworkConfig.from_file("config.yaml")

    programs = {
        "S": FaultySender(M=M),
        "R0": Receiver0("R0", M=M, mu=mu),
        "R1": Receiver1("R1", M=M, mu=mu, lambda_=lambda_),
    }

    result = run(
        config=cfg,
        programs=programs,
        num_times=1,
    )

    sender_result = result[0][0]
    r0_result = result[1][0]
    r1_result = result[2][0]

    sender_measurements = sender_result.get("measurements")
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

    # q = leakage rate
    q = num_leakage / M

    return sender_result.get("inDomain"), r0_result.get("accepted"), r1_result.get("final_decision"), q

def run_single_trial(args):
    M, mu, lambda_ = args
    return run_protocol(M, mu=mu, lambda_=lambda_)

def simulate_failure_probability():
    M_values = list(range(20, 401, 20))
    num_trials = 100
    mu = 0.272
    lambda_ = 0.94
    p0 = 0.05
    p1 = 0.005

    failure_rates = []
    upper_bounds = []
    mc_errors = []
    noisy_rates = []
    avg_qs = []

    for M in M_values:
        print("M: ", M)
        failures = 0
        leakage_qs = []

        args = [(M, mu, lambda_)] * num_trials

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
            results = list(executor.map(run_single_trial, args))

        for inDomain, y0, y1, q in results:
            if not inDomain:
                failures += 1
            elif y0 != y1 and not (None in (y0, y1)):
                failures += 1

            leakage_qs.append(q)

        pf_mc = failures / num_trials
        pf_upper = upper_bound_faulty_sender(M, mu=mu, lambda_=lambda_)

        avg_q = sum(leakage_qs) / len(leakage_qs)
        avg_qs.append(avg_q)
        P = 1 - (1 - avg_q) ** M
        pf_noisy = (1 - P) * pf_upper + P

        noisy_rates.append(pf_noisy)
        failure_rates.append(pf_mc)
        upper_bounds.append(pf_upper)

        se = (pf_mc * (1 - pf_mc) / num_trials) ** 0.5
        mc_errors.append(se)

    plt.figure(figsize=(8, 5))
    plt.errorbar(M_values, failure_rates, yerr=mc_errors, fmt='x', color='red')
    plt.plot(M_values, failure_rates, marker='x', color='red', linestyle='None', label="Monte Carlo")
    plt.plot(M_values, upper_bounds, marker='o', color='green', linestyle='None', label="Upper Bound",
             fillstyle='none')
    plt.plot(M_values, noisy_rates, marker='s', color='blue', linestyle='None', label="Noisy Model", fillstyle='none')
    plt.xlabel("Number of four-qubit singlet states (m)")
    plt.ylabel("Failure probability")
    plt.title(f"Kraus Operator Model Failure Probability (S Faulty), N = {num_trials}, p0 = {p0}, p1 = {p1}")
    max_y = max(max(failure_rates), max(noisy_rates), max(upper_bounds)) + 0.05
    plt.ylim(0, max(1.0, max_y))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def multinomial_coeff(n, k1, k2, k3):
    if k1 + k2 + k3 != n:
        return 0
    return comb(n, k1) * comb(n - k1, k2)

def upper_bound_faulty_sender(
    m: int, mu: float = 0.272, lambda_: float = 0.94, pf_down: float = 0.0
) -> float:
    T = ceil(mu * m)
    Q = T - ceil(lambda_ * T) + 1

    p_known = 0.0
    pf_down = 0.0

    for l3 in range(T, m - T + 1):
        for l1 in range(T - Q, m - Q - l3 + 1):
            l2 = m - l1 - l3
            if l2 < 0 or l1 < 0:
                continue
            coeff = multinomial_coeff(m, l3, l1, l2)
            p_l = coeff * (1 / 3) ** m
            pf_down += p_l * (2 ** -Q)
            p_known += p_l

    return pf_down + (1 - p_known)

if __name__ == "__main__":
    simulate_failure_probability()
