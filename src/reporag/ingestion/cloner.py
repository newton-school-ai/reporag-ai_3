import logging
import os
import tempfile

from git import GitCommandError, Repo

logger = logging.getLogger(__name__)


class RepoCloner:
    """
    Clones a Git repository to a temporary directory and discovers parseable source files.
    """

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
    }

    def __init__(self):
        self.temp_dir: tempfile.TemporaryDirectory | None = None

    def clone_and_discover(
        self, url: str, branch: str | None = None
    ) -> list[tuple[str, str, int]]:
        """
        Clones the repo into a temporary directory and discovers supported files.
        Returns a manifest containing tuples of (file_path, language, size_bytes).
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        repo_path = self.temp_dir.name

        logger.info(f"Cloning {url} to {repo_path}")
        try:
            kwargs = {"depth": 1}
            if branch:
                kwargs["branch"] = branch

            Repo.clone_from(url, repo_path, **kwargs)
        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            self.cleanup()
            raise ValueError(f"Failed to clone repository {url}: {e}") from e

        manifest = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        # Acceptance Criteria asked for (file_path, language, size_bytes)
                        manifest.append(
                            (file_path, self.SUPPORTED_EXTENSIONS[ext], size)
                        )
                    except OSError as e:
                        logger.warning(f"Failed to get size for {file_path}: {e}")

        return manifest

    def cleanup(self):
        """Cleans up the temporary directory."""
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None
