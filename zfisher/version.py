"""Single source of truth for zFISHer version."""

__version__ = "1.0.0"


def get_full_version():
    """Return version string with git hash appended for dev builds.

    - If running from a clean tagged release: '0.1.0'
    - If running from source with commits: '0.1.0-dev+abc1234'
    - If git is unavailable: '0.1.0'
    """
    import subprocess
    from pathlib import Path

    try:
        repo_dir = Path(__file__).resolve().parent.parent
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Check if current commit is exactly a version tag
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Exact tag match — clean release
        return __version__
    except subprocess.CalledProcessError:
        # No exact tag — dev build
        try:
            return f"{__version__}-dev+{sha}"
        except NameError:
            return __version__
    except Exception:
        return __version__
