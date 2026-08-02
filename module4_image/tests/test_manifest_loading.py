import os
import unittest

import pandas as pd

from models.train import normalize_manifest


class ManifestLoadingTests(unittest.TestCase):
    def test_resolves_image_path_manifest_to_existing_paths(self):
        # data/labels_all.csv is the current merged manifest (post dataset
        # merge -- see data/CLEANING_REPORT.md). It already ships an
        # `image_path` column, so this exercises the pass-through branch.
        manifest_path = os.path.join("data", "labels_all.csv")
        manifest = pd.read_csv(manifest_path)

        resolved = normalize_manifest(manifest, manifest_path)

        self.assertIn("image_path", resolved.columns)
        self.assertTrue(resolved["image_path"].notna().all())
        self.assertTrue(all(os.path.exists(path) for path in resolved["image_path"].tolist()))
        # sanity: normalize_manifest should not silently drop rows when every path is valid
        self.assertEqual(len(resolved), len(manifest))

    def test_resolves_filename_manifest_to_existing_paths(self):
        # Build a small filename-based manifest on the fly (the format
        # generate_forgeries.py / add_new_batch.py produce) to test the
        # resolve_image_path() candidate-search branch.
        real_dir = os.path.join("data", "real")
        sample_files = sorted(os.listdir(real_dir))[:3]
        manifest = pd.DataFrame({
            "filename": sample_files,
            "label": [0] * len(sample_files),
        })
        manifest_path = os.path.join("data", "synthetic_filename_manifest.csv")

        resolved = normalize_manifest(manifest, manifest_path)

        self.assertIn("image_path", resolved.columns)
        self.assertEqual(len(resolved), len(sample_files))
        self.assertTrue(all(os.path.exists(path) for path in resolved["image_path"].tolist()))


if __name__ == "__main__":
    unittest.main()
