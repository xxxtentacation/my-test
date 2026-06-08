import itertools

def dp_f2_weighted_sum(jobs, W):
    n = len(jobs)
    # Sort jobs into groups based on weights (W_1 > W_2 > ... > W_K)
    sorted_weights = sorted(list(set(W)), reverse=True)
    K = len(sorted_weights)
    
    J_groups = {W_k: [] for W_k in sorted_weights}
    for j_tuple in jobs:
        j, a, b, w = j_tuple
        J_groups[w].append((j, a, b))
    
    def johnson_sort(T):
        T1 = sorted([j for j in T if j[1] <= j[2]], key=lambda x: x[1])
        T2 = sorted([j for j in T if j[1] > j[2]], key=lambda x: x[2], reverse=True)
        return T1 + T2
    
    L = [johnson_sort(J_groups[W_k]) for W_k in sorted_weights]
    n_k = [len(L_k) for L_k in L]
    
    # State: (P, A, B, z), where P is a tuple of consumed job counts for K classes
    # Initial state: k=0, P=(0,0,...,0)
    Y = {0: {((0,)*K, 0, 0): 0}} # key: k, value: dict of (P, A, B) -> z
    
    def compute_block(k, P, A, B, z, D):
        T = []
        # In the algorithm, block index starts at 0, building k+1 th block.
        # But indices of groups (classes) are 1..K in docs. 
        # Here m goes from k to K-1 (0-indexed). So the forced one is k.
        for m in range(k, K):
            for i in range(D[m]):
                T.append(L[m][P[m] + i])
        
        sigma = johnson_sort(T)
        
        # Verify block tail constraint
        if not sigma or jobs[sigma[-1][0]][3] != sorted_weights[k]: 
            return None
        
        A_prime, B_prime, z_prime = A, B, z
        for j, a, b in sigma:
            A_prime += a
            B_prime = max(B_prime, A_prime) + b
            z_prime += jobs[j][3] * B_prime
            
        return A_prime, B_prime, z_prime
    
    for k in range(K):
        Y[k+1] = {}
        for (P, A, B), z in Y[k].items():
            if k < K:
                d_k_plus_1 = n_k[k] - P[k]
                
                # generate all possible D vectors
                ranges = []
                for m in range(k, K):
                    if m == k:
                        ranges.append(range(n_k[k] - P[k], n_k[k] - P[k] + 1))
                    else:
                        ranges.append(range(0, n_k[m] - P[m] + 1))
                        
                for d_vals in itertools.product(*ranges):
                    D_full = [0]*k + list(d_vals)
                    result = compute_block(k, P, A, B, z, D_full)
                    if result is None: continue
                    
                    A_prime, B_prime, z_prime = result
                    P_prime = list(P)
                    for m in range(k, K):
                        P_prime[m] += D_full[m]
                    P_prime_tuple = tuple(P_prime)
                    
                    state = (P_prime_tuple, A_prime, B_prime)
                    if state not in Y[k+1] or z_prime < Y[k+1][state]:
                        Y[k+1][state] = z_prime
                        
    # Extract optimal
    # Optimal states should have P = (n_1, n_2, ..., n_K)
    target_P = tuple(n_k)
    min_z = float('inf')
    for (P, A, B), z in Y[K].items():
        if P == target_P:
            min_z = min(min_z, z)
            
    return min_z

if __name__ == "__main__":
    # Test cases for DP algorithm
    # Format: (job_id, a_j, b_j, w_j)
    test_jobs = [
        (0, 2, 3, 10),
        (1, 1, 2, 20),
        (2, 4, 1, 10),
        (3, 3, 4, 20)
    ]
    test_W = [10, 20, 10, 20]
    
    res = dp_f2_weighted_sum(test_jobs, test_W)
    print("DP Result:", res)
    
    # Calculate brute force for validation
    import math
    min_bf_z = float('inf')
    best_bf_perm = None
    for perm in itertools.permutations(test_jobs):
        curr_A = 0
        curr_B = 0
        curr_z = 0
        for j, a, b, w in perm:
            curr_A += a
            curr_B = max(curr_B, curr_A) + b
            curr_z += w * curr_B
        if curr_z < min_bf_z:
            min_bf_z = curr_z
            best_bf_perm = perm
            
    print("Brute force Result:", min_bf_z)
    print("Best permutation:", [j for j, a, b, w in best_bf_perm])
    assert res == min_bf_z, "DP result doesn't match brute force!"

print("Algorithm implementation created")
