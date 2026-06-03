"""Tests for workspace and environment discovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nb_nebi_kernels.discovery import (
    discover_environments,
    discover_workspaces,
    env_has_any_kernelspec,
)


class TestDiscoverWorkspaces:
    """Tests for discover_workspaces()."""

    def test_parses_nebi_json_output(self) -> None:
        """Parses JSON from nebi workspace list --json."""
        mock_json = json.dumps([
            {"name": "data-science", "path": "/home/user/data-science", "missing": False},
            {"name": "web-app", "path": "/home/user/web-app", "missing": False},
        ])
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            workspaces = discover_workspaces()

        assert len(workspaces) == 2
        assert workspaces[0].name == "data-science"
        assert workspaces[0].path == "/home/user/data-science"
        assert workspaces[1].name == "web-app"
        assert workspaces[1].path == "/home/user/web-app"

    def test_filters_missing_workspaces(self) -> None:
        """Workspaces with missing=true are excluded."""
        mock_json = json.dumps([
            {"name": "data-science", "path": "/home/user/data-science", "missing": False},
            {"name": "old-project", "path": "/home/user/old-project", "missing": True},
        ])
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            workspaces = discover_workspaces()

        assert len(workspaces) == 1
        assert workspaces[0].name == "data-science"

    def test_returns_empty_when_nebi_not_found(self) -> None:
        """Returns empty list if nebi CLI is not installed."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nebi not found")
            workspaces = discover_workspaces()

        assert workspaces == []

    def test_returns_empty_on_nebi_error(self) -> None:
        """Returns empty list if nebi exits with error."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="some error"
            )
            workspaces = discover_workspaces()

        assert workspaces == []

    def test_returns_empty_when_no_workspaces(self) -> None:
        """Returns empty list when nebi returns empty JSON array."""
        with patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="[]", stderr=""
            )
            workspaces = discover_workspaces()

        assert workspaces == []


class TestDiscoverEnvironments:
    """Tests for discover_environments()."""

    def test_lists_environments_for_workspace(self) -> None:
        """Parses pixi info --json to extract environment names."""
        mock_json = json.dumps({
            "environments_info": [
                {"name": "default", "features": ["default"]},
                {"name": "gpu", "features": ["gpu", "default"]},
            ]
        })
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout=mock_json, stderr=""
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default", "gpu"]

    def test_returns_default_on_error(self) -> None:
        """Falls back to ['default'] if pixi command fails."""
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]

    def test_returns_default_when_pixi_not_found(self) -> None:
        """Falls back to ['default'] if pixi is not installed."""
        with (
            patch("nb_nebi_kernels.discovery.subprocess.run") as mock_run,
            patch("nb_nebi_kernels.discovery._find_manifest", return_value="/mock/pixi.toml"),
        ):
            mock_run.side_effect = FileNotFoundError("pixi not found")
            envs = discover_environments("/home/user/data-science")

        assert envs == ["default"]


class TestEnvHasAnyKernelspec:
    """Tests for env_has_any_kernelspec()."""

    def test_true_when_kernelspec_present(self, tmp_path: Path) -> None:
        """Detects a kernel.json under share/jupyter/kernels/."""
        kernels = tmp_path / ".pixi" / "envs" / "default" / "share" / "jupyter" / "kernels"
        (kernels / "python3").mkdir(parents=True)
        (kernels / "python3" / "kernel.json").write_text("{}")

        assert env_has_any_kernelspec(str(tmp_path), "default") is True

    def test_false_when_env_prefix_missing(self, tmp_path: Path) -> None:
        """Returns False when the env was never installed."""
        assert env_has_any_kernelspec(str(tmp_path), "default") is False

    def test_false_when_kernels_dir_empty(self, tmp_path: Path) -> None:
        """Returns False when share/jupyter/kernels/ has no kernelspecs."""
        kernels = tmp_path / ".pixi" / "envs" / "default" / "share" / "jupyter" / "kernels"
        kernels.mkdir(parents=True)
        assert env_has_any_kernelspec(str(tmp_path), "default") is False

    def test_finds_non_python_kernel(self, tmp_path: Path) -> None:
        """Returns True for any kernelspec, not just python3."""
        kernels = tmp_path / ".pixi" / "envs" / "gpu" / "share" / "jupyter" / "kernels"
        (kernels / "ir").mkdir(parents=True)
        (kernels / "ir" / "kernel.json").write_text("{}")

        assert env_has_any_kernelspec(str(tmp_path), "gpu") is True
