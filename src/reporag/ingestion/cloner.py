"""Repository cloning and source file discovery.

This module clones a Git repository, discovers supported source files,
and returns a manifest describing them.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import TypeAlias

from git import GitCommandError, Repo

logger = logging.getLogger(__name__)

ManifestEntry: TypeAlias = tuple[str, str, int]


class RepoCloner:
    """Clone Git repositories and discover supported source files."""

    DEFAULT_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
    }

    def __init__(self) -> None:
        """Initialize the repository cloner."""
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def clone_and_discover(
        self,
        repo_url: str,
        branch: str = "main",
        shallow: bool = True,
        extensions: dict[str, str] | None = None,
    ) -> list[ManifestEntry]:
        """Clone a repository and discover supported source files.

        Args:
            repo_url: HTTPS Git repository URL.
            branch: Branch to clone.
            shallow: Whether to perform a shallow clone.
            extensions: Optional mapping of file extensions to language names.

        Returns:
            A list of (file_path, language, size_bytes) tuples.

        Raises:
            ValueError: If repository cloning fails.
        """
        supported_extensions = extensions or self.DEFAULT_EXTENSIONS

        self._temp_dir = tempfile.TemporaryDirectory()
        repo_path = self._temp_dir.name

        logger.info("Cloning repository '%s' into '%s'", repo_url, repo_path)

        clone_args: dict[str, object] = {
            "branch": branch,
        }

        if shallow:
            clone_args["depth"] = 1

        try:
            Repo.clone_from(repo_url, repo_path, **clone_args)

            manifest = self._discover_files(
                repo_path,
                supported_extensions,
            )

            logger.info("Discovered %d supported files", len(manifest))
            return manifest

        except GitCommandError as exc:
            logger.exception("Repository cloning failed")
            self.cleanup()
            raise ValueError(f"Failed to clone repository '{repo_url}'") from exc

    def _discover_files(
        self,
        repo_path: str,
        extensions: dict[str, str],
    ) -> list[ManifestEntry]:
        """Discover supported files inside the repository.

        Args:
            repo_path: Local repository path.
            extensions: Supported file extensions.

        Returns:
            A manifest of (file_path, language, size_bytes) tuples.
        """
        manifest: list[ManifestEntry] = []

        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")

            for filename in files:
                extension = Path(filename).suffix.lower()

                if extension not in extensions:
                    continue

                file_path = os.path.join(root, filename)

                try:
                    manifest.append(
                        (
                            file_path,
                            extensions[extension],
                            os.path.getsize(file_path),
                        )
                    )
                except OSError as exc:
                    logger.warning(
                        "Skipping '%s': %s",
                        file_path,
                        exc,
                    )

        return manifest

    def cleanup(self) -> None:
        """Remove the temporary cloned repository."""
        if self._temp_dir is not None:
            logger.info("Cleaning temporary directory")
            self._temp_dir.cleanup()
            self._temp_dir = None
