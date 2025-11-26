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
    Generate a binary (0/1) loss matrix Y of shape (N, N)
    with exact rank r under GF(2) arithmetic.
    
    Steps follow the idea in Algorithm 2 from the project:
      1. Randomly generate independent rows (basis) until rank = r.
      2. Generate dependent rows as GF(2) linear combinations.
      3. Stack into full matrix Y.

    Returns:
        Y : (N, N) binary matrix (dtype=int) with rank r in GF(2)
    """

    # -------- Helper: GF(2) rank over {0,1} --------
    def gf2_rank(matrix):
        A = matrix.copy().astype(int)
        rows, cols = A.shape
        rank = 0
        col = 0

        for r0 in range(rows):
            # find pivot
            while col < cols and A[r0:, col].max() == 0:
                col += 1
            if col == cols:
                break

            # pivot row index
            pivot = r0 + A[r0:, col].argmax()

            # swap pivot row to r0
            if pivot != r0:
                A[[r0, pivot]] = A[[pivot, r0]]

            # eliminate below (GF(2): XOR)
            for rr in range(r0 + 1, rows):
                if A[rr, col] == 1:
                    A[rr] ^= A[r0]   # XOR

            rank += 1
            col += 1

        return rank

    # -------- Step 1: generate r independent basis rows --------
    basis = []
    while len(basis) < r:
        candidate = np.random.randint(0, 2, size=N)
        if len(basis) == 0:
            basis.append(candidate)
            continue

        # check if adding this row increases GF(2) rank
        test = np.vstack(basis + [candidate])
        if gf2_rank(test) > gf2_rank(np.vstack(basis)):
            basis.append(candidate)

    basis = np.vstack(basis)  # shape (r, N)

    # -------- Step 2: generate dependent rows as XOR combinations --------
    rows = [basis[i] for i in range(r)]

    for i in range(N - r):
        # random binary coefficients
        coeff = np.random.randint(0, 2, size=r)
        # avoid all-zero coefficient (would produce zero row)
        while coeff.sum() == 0:
            coeff = np.random.randint(0, 2, size=r)

        # XOR combination of basis rows
        row = np.zeros(N, dtype=int)
        for j in range(r):
            if coeff[j] == 1:
                row ^= basis[j]

        rows.append(row)

    Y = np.vstack(rows)

    # -------- Step 3: shuffle rows (optional but makes simulation unbiased) --------
    np.random.shuffle(Y)

    return Y


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
