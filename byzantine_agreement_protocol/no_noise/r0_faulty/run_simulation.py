import matplotlib.pyplot as plt
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run
from application import Sender, Receiver0, Receiver1
from math import comb, ceil


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

    sender_result = result[0][0]
    r0_result = result[1][0]
    r1_result = result[2][0]

    return sender_result.get("xS"), r0_result.get("y"), r0_result.get("is_valid"), r1_result.get("final_decision")


def simulate_failure_probability():
    M_values = list(range(20, 401, 10))
    num_trials = 100
    mu = 0.272
    lambda_ = 0.94

    failure_rates = []
    exact_rates = []
    mc_errors = []

    for M in M_values:
        print(f"Running for M = {M}")
        failures = 0

        for _ in range(num_trials):
            xS, y0, is_valid, y1 = run_protocol(M, mu=mu, lambda_=lambda_)

            if not is_valid or xS != y1:
                failures += 1

        pf_mc = failures / num_trials
        pf_upper_bound = upper_bound_failure_probability(M, mu=mu, lambda_=lambda_)

        failure_rates.append(pf_mc)
        exact_rates.append(pf_upper_bound)

        se = (pf_mc * (1 - pf_mc) / num_trials) ** 0.5
        mc_errors.append(se)



    plt.figure(figsize=(8, 5))
    plt.errorbar(M_values, failure_rates, yerr=mc_errors, fmt='x', color='red')
    plt.plot(M_values, failure_rates, marker='x', color='red', linestyle='None', label="Monte Carlo")
    plt.plot(M_values, exact_rates, marker='o', color='green', linestyle='None', label="Exact", fillstyle='none')
    plt.xlabel("Number of four-qubit singlet states (m)")
    plt.ylabel("Failure probability")
    plt.title(f"Failure Probability (R0 Faulty), N = {num_trials}")
    plt.ylim(0, 0.8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def upper_bound_failure_probability(m: int, mu: float = 0.272, lambda_: float = 0.94) -> float:
    T = ceil(mu * m)
    Q = T - ceil(T * lambda_) + 1
    pf_down = 0.0

    for l1 in range(T, m - T + 1):
        for l2 in range(0, T - Q + 1):
            l3 = m - l1 - l2
            if l3 < 0:
                continue
            prob = comb(m, l1) * comb(m - l1, l2)
            prob *= (1/3)**l1 * (1/6)**l2 * (1/2)**l3

            sum_k = 0.0
            for k in range(T - Q + 1 - l2, T - l2 + 1):
                if 0 <= k <= (T - l2):
                    sum_k += comb(T - l2, k) * (2/3)**k * (1/3)**(T - l2 - k)

            pf_down += prob * sum_k

    for l1 in range(T, m - T + 1):
        for l2 in range(T - Q + 1, m - l1 + 1):
            l3 = m - l1 - l2
            if l3 < 0:
                continue
            prob = comb(m, l1) * comb(m - l1, l2)
            prob *= (1/3)**l1 * (1/6)**l2 * (1/2)**l3
            pf_down += prob

    for l1 in range(0, T):
        prob = comb(m, l1) * (1/3)**l1 * (2/3)**(m - l1)
        pf_down += prob

    upper_term = 0.0
    for l1 in range(m - T + 1, m + 1):
        term = comb(m, l1) * (1/3)**l1 * (2/3)**(m - l1)
        upper_term += term

    return pf_down + upper_term

if __name__ == "__main__":
    simulate_failure_probability()
