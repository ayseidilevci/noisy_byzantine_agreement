from os import stat_result

import matplotlib.pyplot as plt
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from application import FaultySender, Receiver0, Receiver1
from math import comb, ceil


def run_protocol(M, mu=0.272, lambda_=0.94):
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

    return sender_result.get("inDomain"), r0_result.get("accepted"), r1_result.get("final_decision")


def simulate_failure_probability():
    M_values = list(range(20, 401, 20))
    num_trials = 100
    mu = 0.272
    lambda_ = 0.94

    failure_rates = []
    upper_bounds = []
    mc_errors = []

    for M in M_values:
        print(f"\nRunning for M = {M}")
        failures = 0

        for _ in range(num_trials):
            inDomain, y0, y1 = run_protocol(M, mu=mu, lambda_=lambda_)
            if not inDomain:
                failures += 1
            elif y0 != y1 and not (None in (y0, y1)):
                failures += 1

        pf_mc = failures / num_trials
        pf_upper = upper_bound_faulty_sender(M, mu=mu, lambda_=lambda_)

        failure_rates.append(pf_mc)
        upper_bounds.append(pf_upper)

        se = (pf_mc * (1 - pf_mc) / num_trials) ** 0.5
        mc_errors.append(se)


    plt.figure(figsize=(8, 5))
    plt.errorbar(M_values, failure_rates, yerr=mc_errors, fmt='x', color='red')
    plt.plot(M_values, failure_rates, marker='x', color='red', linestyle='None', label="Monte Carlo")
    plt.plot(M_values, upper_bounds, marker='o', color='green', linestyle='None', label="Upper Bound", fillstyle='none')
    plt.xlabel("Number of four-qubit singlet states (m)")
    plt.ylabel("Failure probability")
    plt.title(f"Failure Probability (S Faulty), N = {num_trials}")
    plt.ylim(0, 1.0)
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