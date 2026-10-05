import Tree_Project as baseline


def touch_sample_files(directory, names):
    paths = {}
    for name in names:
        path = directory / name
        path.touch()
        paths[name] = path
    return paths


def test_summer_rgb_gt_pairing_uses_tree_season_and_sample_tag(tmp_path):
    paths = touch_sample_files(
        tmp_path,
        (
            "R01N01_Summer_RGB-12-40-50.png",
            "R01N01_Summer_GT-12-40-50.png",
            "R01N01_Summer_D-12-40-50.png",
            "R01N02_Summer_RGB-12-40-51.png",
            "R01N01_Winter_RGB-11-23-49.png",
            "R01N01_Winter_GT-11-23-49.png",
        ),
    )

    images = baseline.scan_images(tmp_path)
    pairs = baseline.find_summer_rgb_gt_pairs(images)

    expected_key = ("R01N01", "Summer", "12-40-50")
    assert set(pairs) == {expected_key}
    assert pairs[expected_key]["RGB"] == paths["R01N01_Summer_RGB-12-40-50.png"]
    assert pairs[expected_key]["GT"] == paths["R01N01_Summer_GT-12-40-50.png"]
    assert pairs[expected_key]["D"] == paths["R01N01_Summer_D-12-40-50.png"]