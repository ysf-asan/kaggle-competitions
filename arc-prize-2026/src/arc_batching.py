def build_decode_batches(test_id_to_subkeys, n_perm):
    """Batches of four augmentations, ordered so the earliest batches already
    cover diverse transforms.

    `ArcDataset.augment` produces eight dihedral groups (identity/transpose
    crossed with rot90^k), each holding `n_perm` colour permutations, and
    sorting the keys lays those groups out contiguously. A batch pairs two
    groups two rotations apart so it never spends all four slots on
    near-identical views: when a task is cut short by the time budget, what we
    did decode is still spread over the symmetry group. Non-transposed groups
    are scheduled first, and all test outputs get round r before any gets
    round r+1.
    """
    pair_order = [(0, 2), (1, 3), (4, 6), (5, 7)]

    batches = []
    for p in range(0, n_perm, 2):
        for a, b in pair_order:
            for subkeys in test_id_to_subkeys.values():
                batch = (subkeys[a*n_perm + p : a*n_perm + p + 2]
                         + subkeys[b*n_perm + p : b*n_perm + p + 2])
                if batch:
                    batches.append(batch)
    return batches
