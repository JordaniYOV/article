"""Parse all project TOML and check conservative scaffold defaults."""

from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_toml_syntax(self):
        paths = [ROOT / "pyproject.toml"]
        for directory in ("configs", ".codex/agents", "docs/agent_templates"):
            paths.extend((ROOT / directory).glob("*.toml"))
        for path in paths:
            with self.subTest(path=str(path)):
                with path.open("rb") as stream:
                    self.assertTrue(tomllib.load(stream))

    def test_models_disabled_and_unique(self):
        with (ROOT / "configs/models.toml").open("rb") as stream:
            models = tomllib.load(stream)["models"]
        self.assertEqual(len(models), len({model["id"] for model in models}))
        self.assertTrue(all(model["enabled"] is False for model in models))

    def test_datasets_and_experiments_disabled(self):
        with (ROOT / "configs/datasets.toml").open("rb") as stream:
            datasets = tomllib.load(stream)["datasets"]
        self.assertEqual(len(datasets), 5)
        self.assertTrue(all(item["enabled"] is False for item in datasets.values()))
        self.assertEqual(datasets["lucid"]["track"], "lunar_orbital")
        with (ROOT / "configs/experiments.toml").open("rb") as stream:
            experiments = tomllib.load(stream)
        for index in range(1, 6):
            self.assertIs(experiments[f"E{index}"]["enabled"], False)

    def test_local_agent_schema_when_available(self):
        # Codex role settings may legitimately be customized or absent in a clone.
        for installed in (ROOT / ".codex/agents").glob("*.toml"):
            with self.subTest(agent=installed.stem):
                with installed.open("rb") as stream:
                    config = tomllib.load(stream)
                self.assertTrue(config["name"])
                self.assertTrue(config["description"])
                self.assertTrue(config["developer_instructions"])


if __name__ == "__main__":
    unittest.main()
