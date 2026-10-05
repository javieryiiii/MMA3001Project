import numpy as np
import pytest

import Tree_Project as baseline


def make_synthetic_samples():
    tree_ids = np.asarray([f"tree-{index // 2:02d}" for index in range(40)])
    sample_keys = [
        (tree_id, "Summer", f"sample-{index:02d}")
        for index, tree_id in enumerate(tree_ids)
    ]
    return tree_ids, sample_keys


def normalize_splits(splits):
    return {
        name: (
            frozenset(split["tree_ids"]),
            tuple(split["sample_keys"]),
            tuple(split["indices"].tolist()),
        )
        for name, split in splits.items()
    }


def test_seed_42_split_is_deterministic():
    tree_ids, sample_keys = make_synthetic_samples()

    first = baseline.split_by_tree(tree_ids, sample_keys, seed=42)
    second = baseline.split_by_tree(tree_ids, sample_keys, seed=42)

    assert normalize_splits(first) == normalize_splits(second)
    different_seed = baseline.split_by_tree(tree_ids, sample_keys, seed=43)
    assert normalize_splits(first) != normalize_splits(different_seed)


def test_tree_ids_are_isolated_between_splits():
    tree_ids, sample_keys = make_synthetic_samples()
    splits = baseline.split_by_tree(tree_ids, sample_keys, seed=42)

    baseline.verify_tree_split_isolation(splits)
    train_ids = splits["train"]["tree_ids"]
    validation_ids = splits["validation"]["tree_ids"]
    test_ids = splits["test"]["tree_ids"]
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_tree_split_isolation_rejects_overlap():
    overlapping_splits = {
        "train": {"tree_ids": {"tree-shared"}},
        "validation": {"tree_ids": {"tree-shared"}},
        "test": {"tree_ids": {"tree-test"}},
    }

    with pytest.raises(AssertionError, match="overlap"):
        baseline.verify_tree_split_isolation(overlapping_splits)