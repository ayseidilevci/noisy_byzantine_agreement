import matplotlib.pyplot as plt
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from application import Sender, Receiver0, Receiver1
from math import comb, ceil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def run_protocol(M, mu=0.272, lambda_=0.94):
    cfg = StackNetworkConfig.from_file("config.yaml")

    programs = {
        "S": Sender(M=M),
        "R0": Receiver0("R0", M=M, mu=mu),
        "R1": Receiver1("R1", M=M, mu=mu, lambda_=lambda_),
    }

    result = run(
        config=cfg,
        programs=programs,
        num_times=1,
    )

    r0_result = result[1][0]
    r1_result = result[2][0]

    return r0_result.get("accepted"), r1_result.get("final_decision")


def run_single_trial(args):
    M, mu, lambda_ = args
    return run_protocol(M, mu=mu, lambda_=lambda_)


def simulate_failure_probability():
    M_values = list(range(20, 401, 20))
    num_trials = 100
    mu = 0.272
    lambda_ = 0.94

    failure_rates = []
    exact_rates = []
    mc_errors = []

    for M in M_values:
        print(f"Running for M = {M}")

        args = [(M, mu, lambda_)] * num_trials

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
            results = list(executor.map(run_single_trial, args))

        failures = sum(1 for y1, y2 in results if y1 is None or y2 is None or y1 != y2)

        pf_mc = failures / num_trials
        pf_exact = exact_failure_probability(M, mu=mu)

        failure_rates.append(pf_mc)
        exact_rates.append(pf_exact)

        se = (pf_mc * (1 - pf_mc) / num_trials) ** 0.5
        mc_errors.append(se)

        print(f"  Monte Carlo failure rate = {pf_mc:.4f}")
        print(f"  Exact failure rate        = {pf_exact:.4f}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(M_values, failure_rates, yerr=mc_errors, fmt='x', color='red')
    plt.plot(M_values, failure_rates, marker='x', color='red', linestyle='None', label="Monte Carlo")
    plt.plot(M_values, exact_rates, marker='o', color='green', linestyle='None', label="Exact", fillstyle='none')
    plt.xlabel("Number of four-qubit singlet states (m)")
    plt.ylabel("Failure probability")
    plt.title(f"Failure Probability (No Faulty), N = {num_trials}")
    max_y = max(max(failure_rates), max(exact_rates)) + 0.05
    plt.ylim(0, max(1.0, max_y))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def exact_failure_probability(m: int, mu: float = 0.272) -> float:
    T = ceil(mu * m)
    prob = 0.0
    for k in range(T):
        binom = comb(m, k)
        term = binom * (1 / 3) ** k * (2 / 3) ** (m - k)
        prob += term
    return prob


if __name__ == "__main__":
    simulate_failure_probability()
