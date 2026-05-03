from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_croniter_is_a_base_dependency():
    """OQ-24: croniter must be a runtime dep, not just an extra.

    The gateway imports cron.scheduler unconditionally at startup, and
    cron.jobs.compute_next_run silently returns None for cron-kind
    schedules when croniter is missing — which causes mark_job_run to
    flip ``enabled`` to False after the first fire.  Keeping croniter in
    [cron] *only* meant a fresh install would create the bug, with the
    failure mode being a one-shot disable rather than an ImportError.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert any(dep.startswith("croniter") for dep in deps), (
        "croniter must be in [project].dependencies — see OQ-24"
    )

    # Keep it in the [cron] extra too: removing would be a breaking change
    # for anyone pinning hermes-agent[cron] in their own constraints files.
    cron_extra = data["project"]["optional-dependencies"]["cron"]
    assert any(dep.startswith("croniter") for dep in cron_extra)
