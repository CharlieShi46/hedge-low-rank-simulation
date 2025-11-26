import numpy as np

# -------------------------------------------------------------
# Exponentially Weighted Average (Hedge) Algorithm
# -------------------------------------------------------------
def EWA(N, T, Y):
    """
    Horizon-adaptive Hedge (Exponentially Weighted Average)
    N: number of experts
    T: time horizon
    Y: loss matrix of shape (N, T)
    Returns: regret (your loss - best expert's loss)
    """
    # Initialize weights uniformly
    weight = np.ones(N) / N
    cum_loss = np.dot(weight, Y[:, 0])

    # Online learning loop
    for t in range(1, T):
        eta = np.sqrt(8 * np.log(N) / t)  # horizon-adaptive learning rate

        # Exponential weight update
        weight *= np.exp(-eta * Y[:, t - 1])
        weight /= np.sum(weight)

        # Add learner's loss for this round
        cum_loss += np.dot(weight, Y[:, t])

    # Best expert cumulative loss
    best_expert_loss = np.min(np.sum(Y, axis=1))
    regret = cum_loss - best_expert_loss

    return regret


# -------------------------------------------------------------
# Placeholder for your low-rank matrix generator
# -------------------------------------------------------------
def algo5(N, r):
    """
    Generate a binary (0/1) loss matrix Y of size (N, N)
    with exact rank r.
    Replace this stub with your actual implementation.

    N: number of experts (= horizon)
    r: desired rank
    """

    # -------- Replace this with your actual algorithm --------
    # For now: use random full-rank matrix as temporary placeholder
    # (Just so the script runs. Replace this with your Algorithm 2.)
    Y = np.random.randint(0, 2, size=(N, N))
    return Y
    # ----------------------------------------------------------


# -------------------------------------------------------------
# Simulation: worst-case regret vs rank
# -------------------------------------------------------------
def simulate_regret(N=20, trials=100000, max_rank=20):
    """
    Run worst-case regret simulation for ranks 1 ... max_rank.
    Returns: numpy array of regrets (length max_rank)
    """
    Regrets = np.zeros(max_rank)

    for r in range(1, max_rank + 1):
        print(f"Simulating rank r = {r} ...")
        max_regret = 0

        for _ in range(trials):
            Y = algo5(N, r)
            Reg = EWA(N, N, Y)
            max_regret = max(max_regret, Reg)

        Regrets[r - 1] = max_regret

    return Regrets


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    N = 20
    trials = 100000
    max_rank = 20

    regrets = simulate_regret(N=N, trials=trials, max_rank=max_rank)
    print("\nWorst-case regret for ranks 1..20:")
    print(regrets)