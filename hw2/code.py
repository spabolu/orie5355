# ORIE 5355 / INFO 5370 — HW2: end-to-end solution runner
# This script loads the provided data from /mnt/data, computes all requested
# predictions, recommendations, capacity-constrained assignments, and figures.
# It prints key requested values and renders the plots. All charts use matplotlib
# (no seaborn), one per figure, with default colors only.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Helpers ---------------------------------------------------------------

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_ratings_and_factors_from_mnt(type_name='rating_all_zero'):
    """
    Loads user/item latent factors for a given type from /mnt/data.

    type_name ∈ {'rating_all_zero', 'rating_interaction_zero'}
    Files expected (already provided by user):
      /mnt/data/{type_name}_dict_book_factor
      /mnt/data/{type_name}_dict_reader_factor
    """
    base = "/mnt/data"
    book_vectors = load_pickle(os.path.join(base, f"{type_name}_dict_book_factor"))
    user_vectors = load_pickle(os.path.join(base, f"{type_name}_dict_reader_factor"))
    # Convert to numpy arrays in case they're lists
    book_vectors = np.asarray(book_vectors)
    user_vectors = np.asarray(user_vectors)
    return book_vectors, user_vectors

def get_predictions(user_vectors, book_vectors):
    """
    Returns the predicted rating matrix of shape (n_users, n_items),
    using dot-product of latent factors.
    """
    return user_vectors @ book_vectors.T

def get_recommendations_for_each_user(predictions, number_top_items=10):
    """
    For each user (row), return indices of the top-k items by predicted rating.
    Ties broken by numpy's stable argsort; items are 0-indexed.
    Returns: dict {user_index: [top_k_item_indices...]}
    """
    # argsort descending: use negative
    topk = np.argpartition(-predictions, kth=number_top_items-1, axis=1)[:, :number_top_items]
    # Finish sorting the top-k subset for exact order
    out = {}
    for u in range(predictions.shape[0]):
        idxs = topk[u]
        # sort those by predicted score descending
        ordered_local = idxs[np.argsort(-predictions[u, idxs])]
        out[u] = ordered_local.tolist()
    return out

def frequency_histogram_from_recs(recommendations, n_items):
    """
    Given top-k recommendations per user (dict), compute how many times
    each item is recommended, then return a frequency histogram
    mapping {frequency: number_of_items_with_that_frequency}.
    """
    counts = np.zeros(n_items, dtype=int)
    for _, items in recommendations.items():
        for j in items:
            counts[j] += 1
    # Build histogram of frequencies
    unique, freqs = np.unique(counts, return_counts=True)
    hist = dict(zip(unique.tolist(), freqs.tolist()))
    return counts, hist

def plot_histogram_of_item_frequencies(hist_dict, title="Item recommendation frequency histogram"):
    """
    Bar-chart style histogram from {frequency: num_items} dict.
    """
    xs = sorted(hist_dict.keys())
    ys = [hist_dict[x] for x in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Times an item appears in users' top-k lists")
    plt.ylabel("Number of items with that frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Rank utility: precompute full descending rank list per user
def precompute_ranking(predictions):
    """
    Return list (len = n_users) of numpy arrays where arr[u][r] = item id at rank r (0 is best).
    Also return an inverse-rank lookup matrix 'rank_of_item[u, j]' (1-based ranks for convenience).
    """
    n_users, n_items = predictions.shape
    ranked_items = np.argsort(-predictions, axis=1)  # descending
    # Build inverse rank lookup
    rank_of_item = np.empty_like(ranked_items, dtype=int)
    for u in range(n_users):
        # ranked_items[u, r] = item; inverse so rank_of_item[u, item] = r+1 (1-based rank)
        rank_of_item[u, ranked_items[u]] = np.arange(1, n_items+1)
    return ranked_items, rank_of_item

# ---- Load latent factors and demographics ----------------------------------

book_all0, users_all0 = load_ratings_and_factors_from_mnt('rating_all_zero')
book_int0, users_int0 = load_ratings_and_factors_from_mnt('rating_interaction_zero')

print("Shapes:")
print("  rating_all_zero:   items", book_all0.shape, "users", users_all0.shape)
print("  rating_interact0:  items", book_int0.shape, "users", users_int0.shape)

# Load demographics CSV
demographics = pd.read_csv("/mnt/data/user_demographics.csv")

# ---- Problem 1: predictions & recommendations ------------------------------

# (1b) Predictions for both data types
pred_all0 = get_predictions(users_all0, book_all0)        # (1000, 200)
pred_int0 = get_predictions(users_int0, book_int0)        # (1000, 200)

# Print first user's first 10 items for both
np.set_printoptions(precision=3, suppress=True)
print("\nP1b) First user, first 10 predicted ratings (rating_all_missing_zero):")
print(pred_all0[0, :10])
print("\nP1b) First user, first 10 predicted ratings (rating_interaction_zero):")
print(pred_int0[0, :10])

# Scatter of all user-item pairs: x = interaction zero, y = all-missing zero
plt.figure()
plt.scatter(pred_int0.ravel(), pred_all0.ravel(), s=2)
plt.xlabel("Predicted rating (interaction_zero)")
plt.ylabel("Predicted rating (all_missing_zero)")
plt.title("Predicted ratings: interaction_zero vs all_missing_zero")
plt.tight_layout()
plt.show()

# (1c) Top-10 recs per user for both types
recs_all0 = get_recommendations_for_each_user(pred_all0, number_top_items=10)
recs_int0 = get_recommendations_for_each_user(pred_int0, number_top_items=10)

print("\nP1c) Top-10 recommendations for user 0 (rating_all_missing_zero):")
print(recs_all0[0])
print("\nP1c) Top-10 recommendations for user 0 (rating_interaction_zero):")
print(recs_int0[0])

# Frequency histograms for both types
counts_all0, hist_all0 = frequency_histogram_from_recs(recs_all0, n_items=book_all0.shape[0])
counts_int0, hist_int0 = frequency_histogram_from_recs(recs_int0, n_items=book_int0.shape[0])

print("\nP1c) Histogram data (all_missing_zero):", hist_all0)
print("      Max frequency (any item):", counts_all0.max(), " Min:", counts_all0.min())
plot_histogram_of_item_frequencies(hist_all0, title="Item frequency histogram — all_missing_zero")

print("\nP1c) Histogram data (interaction_zero):", hist_int0)
print("      Max frequency (any item):", counts_int0.max(), " Min:", counts_int0.min())
plot_histogram_of_item_frequencies(hist_int0, title="Item frequency histogram — interaction_zero")

# ---- Problem 2: Cold-start for last 100 users (use interaction_zero only) --

existing_user_vectors = users_int0[0:900, :]
existing_user_demographics = demographics.iloc[0:900, :].copy()
new_user_demographics = demographics.iloc[900:, :].copy()

median_wealth = existing_user_demographics["Wealth"].median()
print("\nP2) Median wealth used for split:", median_wealth)

def get_user_vector_for_new_user(new_user_row, existing_demo, existing_vectors, median_split):
    """
    Simple rule: High vs Low by wealth median; use mean vector of the group.
    """
    is_high = new_user_row["Wealth"] > median_split
    mask = (existing_demo["Wealth"] > median_split) if is_high else (existing_demo["Wealth"] <= median_split)
    mean_vec = existing_vectors[mask.values, :].mean(axis=0)
    return mean_vec

# Build mean vectors once
mask_low = existing_user_demographics["Wealth"] <= median_wealth
mask_high = ~mask_low
mean_low = existing_user_vectors[mask_low.values, :].mean(axis=0)
mean_high = existing_user_vectors[mask_high.values, :].mean(axis=0)

# Example check: vector for the "second" new user (index 1 in new_user_demographics)
vec_new_idx1 = get_user_vector_for_new_user(new_user_demographics.iloc[1], existing_user_demographics, existing_user_vectors, median_wealth)
print("\nP2a) Sanity: Predicted vector for new_user_demographics index 1 (rounded to 3dp):")
print(np.round(vec_new_idx1, 3))

# Output the mean vector predicted for the FIRST new user (index 0)
vec_first_new = get_user_vector_for_new_user(new_user_demographics.iloc[0], existing_user_demographics, existing_user_vectors, median_wealth)
print("\nP2a) Predicted vector for new_user_demographics index 0 (rounded to 3dp):")
print(np.round(vec_first_new, 3))

# Build vectors for all 100 new users using the simple demographic rule
pred_new_user_vectors_simple = np.vstack([
    get_user_vector_for_new_user(new_user_demographics.iloc[i], existing_user_demographics, existing_user_vectors, median_wealth)
    for i in range(len(new_user_demographics))
])

# Predictions for 100 new users with demographic model vs full model (true latent vectors)
pred_demo_100x200 = get_predictions(pred_new_user_vectors_simple, book_int0)                 # 100x200
true_user_vectors_last100 = users_int0[900:, :]                                              # 100x10
pred_full_100x200 = get_predictions(true_user_vectors_last100, book_int0)                    # 100x200

# Scatter (20k points): x = demo-based predictions, y = full-model predictions
plt.figure()
plt.scatter(pred_demo_100x200.ravel(), pred_full_100x200.ravel(), s=2)
plt.xlabel("Predicted rating (demographic model)")
plt.ylabel("Predicted rating (full model)")
plt.title("Cold-start: demographic vs full-model predictions (100 users × 200 items)")
plt.tight_layout()
plt.show()

# Example first user-item pair values for reference
print("\nP2a) Example (user0,item0): demo =", round(pred_demo_100x200[0,0], 4), 
      ", full =", round(pred_full_100x200[0,0], 4))

# ---- Problem 2b (Bonus): KNN-based new-user vector -------------------------

from math import isfinite

def _encode_demographics(df, fit_stats=None):
    """
    One-hot encode Age group and Location, keep Wealth numeric.
    Returns X (numpy array) and fit_stats (for standardization of Wealth).
    fit_stats: {'wealth_mean': float, 'wealth_std': float, 'age_cols': [...], 'loc_cols': [...]}
    """
    # Keep order stable
    age_dummies = pd.get_dummies(df["Age group"], prefix="age", drop_first=False)
    loc_dummies = pd.get_dummies(df["Location"], prefix="loc", drop_first=False)
    wealth = df[["Wealth"]].copy()
    if fit_stats is None:
        w_mean = wealth["Wealth"].mean()
        w_std = wealth["Wealth"].std(ddof=0)
        fit_stats = {
            "wealth_mean": float(w_mean),
            "wealth_std": float(w_std if w_std > 0 else 1.0),
            "age_cols": age_dummies.columns.tolist(),
            "loc_cols": loc_dummies.columns.tolist(),
        }
    # Align columns to training fit_stats
    age_dummies = age_dummies.reindex(columns=fit_stats["age_cols"], fill_value=0)
    loc_dummies = loc_dummies.reindex(columns=fit_stats["loc_cols"], fill_value=0)
    # Standardize wealth
    wealth_std = (wealth["Wealth"] - fit_stats["wealth_mean"]) / (fit_stats["wealth_std"] if fit_stats["wealth_std"] != 0 else 1.0)
    X = np.hstack([wealth_std.to_numpy().reshape(-1,1),
                   age_dummies.to_numpy(),
                   loc_dummies.to_numpy()])
    return X, fit_stats

# Fit encoders on existing users
X_existing, fit_stats = _encode_demographics(existing_user_demographics, fit_stats=None)
X_new, _ = _encode_demographics(new_user_demographics, fit_stats=fit_stats)

def get_user_vector_for_new_user_knn(new_user_vec, X_existing, existing_vectors, k=15, eps=1e-6):
    """
    KNN in standardized+one-hot demographic space (Euclidean).
    Weighted average of neighbors' user vectors using inverse-distance weights.
    """
    diffs = X_existing - new_user_vec  # (900, d)
    dists = np.sqrt(np.sum(diffs*diffs, axis=1))  # (900,)
    # Handle exact matches (distance 0): if any, take mean of those
    zero_mask = (dists < eps)
    if np.any(zero_mask):
        return existing_vectors[zero_mask, :].mean(axis=0)
    # Otherwise select k nearest by distance
    nn_idx = np.argpartition(dists, k-1)[:k]
    nn_d = dists[nn_idx]
    weights = 1.0 / (nn_d + eps)
    weights = weights / weights.sum()
    vec = (existing_vectors[nn_idx, :] * weights.reshape(-1,1)).sum(axis=0)
    return vec

# Predict KNN vectors for all 100 new users
pred_new_user_vectors_knn = np.vstack([
    get_user_vector_for_new_user_knn(X_new[i], X_existing, existing_user_vectors, k=15)
    for i in range(X_new.shape[0])
])

# Output the predicted vector for the FIRST new user (index 0) with KNN
print("\nP2b) KNN-predicted vector for new_user_demographics index 0 (rounded to 3dp):")
print(np.round(pred_new_user_vectors_knn[0], 3))

# Scatter vs full model like 2a
pred_knn_100x200 = get_predictions(pred_new_user_vectors_knn, book_int0)

plt.figure()
plt.scatter(pred_knn_100x200.ravel(), pred_full_100x200.ravel(), s=2)
plt.xlabel("Predicted rating (KNN demographic model)")
plt.ylabel("Predicted rating (full model)")
plt.title("Cold-start (KNN): KNN vs full-model predictions (100 users × 200 items)")
plt.tight_layout()
plt.show()

# ---- Problem 3: Capacity constraints (interaction_zero only) ---------------

# Use full 1000x200 predictions and precomputed rankings
pred_full = pred_int0  # alias
n_users, n_items = pred_full.shape
ranked_items, rank_of_item = precompute_ranking(pred_full)

# 3a) Naive sequential: give each user best available item (capacity 5 per item)
capacity = np.full(n_items, 5, dtype=int)
naive_assigned_item = np.full(n_users, -1, dtype=int)
naive_assigned_rank = np.full(n_users, -1, dtype=int)  # 1-based rank

for u in range(n_users):
    # iterate user's ranked items until find one with capacity
    for item in ranked_items[u]:
        if capacity[item] > 0:
            naive_assigned_item[u] = item
            naive_assigned_rank[u] = rank_of_item[u, item]  # 1-based
            capacity[item] -= 1
            break

print("\nP3a) Naive sequential: example ranks for users 0..9:", naive_assigned_rank[:10].tolist())
print("      Last user rank:", int(naive_assigned_rank[-1]))

# Plot line: user index vs rank
plt.figure()
plt.plot(np.arange(n_users), naive_assigned_rank)
plt.xlabel("User index")
plt.ylabel("Rank of assigned item (1=best)")
plt.title("P3a) Naive sequential recommendation under capacity constraints")
plt.tight_layout()
plt.show()

# Histogram of ranks
plt.figure()
plt.hist(naive_assigned_rank, bins=30)
plt.xlabel("Assigned rank (1=best)")
plt.ylabel("Count of users")
plt.title("P3a) Naive sequential — histogram of assigned ranks")
plt.tight_layout()
plt.show()

# 3b) Optimal max-weight matching (Hungarian). We create 5 copies of each item (slots).
# This requires scipy; if not present, we'll skip gracefully.
opt_assigned_rank = None
batched_assigned_rank = None

try:
    from scipy.optimize import linear_sum_assignment

    # Build item-slots: 5 copies of each item => total 1000 slots
    # cost = -pred to maximize total predicted ratings
    item_slots = np.tile(np.arange(n_items), 5)  # length 1000; maps slot -> item id
    cost_matrix = -pred_full[:, item_slots]      # shape (1000, 1000)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind should be [0..999] in order; but we won't assume.
    opt_assigned_item = item_slots[col_ind]
    # compute assigned rank per user
    opt_assigned_rank = np.empty(n_users, dtype=int)
    for ru, u in enumerate(row_ind):
        j = opt_assigned_item[ru]
        opt_assigned_rank[u] = rank_of_item[u, j]

    # Plot optimal
    plt.figure()
    plt.plot(np.arange(n_users), opt_assigned_rank[np.argsort(np.arange(n_users))])
    plt.xlabel("User index")
    plt.ylabel("Rank of assigned item (1=best)")
    plt.title("P3b) Optimal assignment (global Hungarian) — assigned ranks")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(opt_assigned_rank, bins=30)
    plt.xlabel("Assigned rank (1=best)")
    plt.ylabel("Count of users")
    plt.title("P3b) Optimal assignment — histogram of assigned ranks")
    plt.tight_layout()
    plt.show()

    # Batched (100 users per batch), respecting remaining capacities
    remaining_capacity = np.full(n_items, 5, dtype=int)
    batched_assigned_rank = np.full(n_users, -1, dtype=int)

    start = 0
    batch_size = 100
    while start < n_users:
        end = min(start + batch_size, n_users)
        batch_users = np.arange(start, end)
        # Build available slots based on current capacity
        slots = []
        for j in range(n_items):
            if remaining_capacity[j] > 0:
                slots.extend([j] * int(remaining_capacity[j]))
        slots = np.array(slots, dtype=int)
        # cost matrix for this batch
        cost_batch = -pred_full[batch_users][:, slots]  # (b, #slots)
        r_ind, c_ind = linear_sum_assignment(cost_batch)
        # Assign
        for idx_in_batch, slot_idx in zip(batch_users[r_ind], c_ind):
            item = slots[slot_idx]
            batched_assigned_rank[idx_in_batch] = rank_of_item[idx_in_batch, item]
            remaining_capacity[item] -= 1
        start = end

    plt.figure()
    plt.plot(np.arange(n_users), batched_assigned_rank)
    plt.xlabel("User index")
    plt.ylabel("Rank of assigned item (1=best)")
    plt.title("P3b) Batched Hungarian (100-user batches) — assigned ranks")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(batched_assigned_rank, bins=30)
    plt.xlabel("Assigned rank (1=best)")
    plt.ylabel("Count of users")
    plt.title("P3b) Batched Hungarian — histogram of assigned ranks")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\n[Warning] SciPy not available or failed to solve Hungarian assignment.")
    print("Error:", repr(e))
    print("Skipping P3b optimal and batched plots.")

# 3c) Score-function approach (sequential like 3a but using modified scores)
# Normalize predictions per-item to [0, 1] to avoid negative means
pred_norm = pred_full.copy().astype(float)
# min-max per item
mins = pred_norm.min(axis=0)
maxs = pred_norm.max(axis=0)
spans = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
pred_norm = (pred_norm - mins) / spans
# mean rating per item after normalization
mean_item = pred_norm.mean(axis=0)
eps = 1e-6

capacity_sc = np.full(n_items, 5, dtype=int)
score_assigned_item = np.full(n_users, -1, dtype=int)
score_assigned_rank = np.full(n_users, -1, dtype=int)

for u in range(n_users):
    # Compute score_j = (r_ij / rbar_j) * sqrt(Cj), ignoring items with 0 capacity
    valid = capacity_sc > 0
    scores = np.full(n_items, -np.inf)
    # Avoid division by ~0 by adding eps to mean
    tmp = (pred_norm[u, valid] / (mean_item[valid] + eps)) * np.sqrt(capacity_sc[valid])
    scores[valid] = tmp
    best_item = int(np.argmax(scores))
    score_assigned_item[u] = best_item
    score_assigned_rank[u] = rank_of_item[u, best_item]
    capacity_sc[best_item] -= 1

print("\nP3c) Score-function sequential: example ranks for users 0..9:", score_assigned_rank[:10].tolist())
print("      Last user rank:", int(score_assigned_rank[-1]))

plt.figure()
plt.plot(np.arange(n_users), score_assigned_rank)
plt.xlabel("User index")
plt.ylabel("Rank of assigned item (1=best)")
plt.title("P3c) Score-function sequential — assigned ranks")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(score_assigned_rank, bins=30)
plt.xlabel("Assigned rank (1=best)")
plt.ylabel("Count of users")
plt.title("P3c) Score-function sequential — histogram of assigned ranks")
plt.tight_layout()
plt.show()

# Save key arrays in case the user wants to download them
np.save("/mnt/data/pred_all_missing_zero.npy", pred_all0)
np.save("/mnt/data/pred_interaction_zero.npy", pred_int0)
np.save("/mnt/data/recs_all_missing_zero_user0.npy", np.array(recs_all0[0]))
np.save("/mnt/data/recs_interaction_zero_user0.npy", np.array(recs_int0[0]))
np.save("/mnt/data/naive_assigned_rank.npy", naive_assigned_rank)
if 'opt_assigned_rank' in locals() and opt_assigned_rank is not None:
    np.save("/mnt/data/optimal_assigned_rank.npy", opt_assigned_rank)
if 'batched_assigned_rank' in locals() and batched_assigned_rank is not None:
    np.save("/mnt/data/batched_assigned_rank.npy", batched_assigned_rank)
np.save("/mnt/data/score_assigned_rank.npy", score_assigned_rank)

print("\nFiles saved:")
print("  - /mnt/data/pred_all_missing_zero.npy")
print("  - /mnt/data/pred_interaction_zero.npy")
print("  - /mnt/data/recs_all_missing_zero_user0.npy")
print("  - /mnt/data/recs_interaction_zero_user0.npy")
print("  - /mnt/data/naive_assigned_rank.npy")
if 'opt_assigned_rank' in locals() and opt_assigned_rank is not None:
    print("  - /mnt/data/optimal_assigned_rank.npy")
if 'batched_assigned_rank' in locals() and batched_assigned_rank is not None:
    print("  - /mnt/data/batched_assigned_rank.npy")
print("  - /mnt/data/score_assigned_rank.npy")
