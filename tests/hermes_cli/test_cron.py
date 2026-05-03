"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.cron import cron_command


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronCommandLifecycle:
    def test_pause_resume_run(self, tmp_cron_dir, capsys):
        job = create_job(prompt="Check server status", schedule="every 1h")

        cron_command(Namespace(cron_command="pause", job_id=job["id"]))
        paused = get_job(job["id"])
        assert paused["state"] == "paused"

        cron_command(Namespace(cron_command="resume", job_id=job["id"]))
        resumed = get_job(job["id"])
        assert resumed["state"] == "scheduled"

        cron_command(Namespace(cron_command="run", job_id=job["id"]))
        triggered = get_job(job["id"])
        assert triggered["state"] == "scheduled"

        out = capsys.readouterr().out
        assert "Paused job" in out
        assert "Resumed job" in out
        assert "Triggered job" in out

    def test_edit_can_replace_and_clear_skills(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Combine skill outputs",
            schedule="every 1h",
            skill="blogwatcher",
        )

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule="every 2h",
                prompt="Revised prompt",
                name="Edited Job",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["maps", "blogwatcher"],
                clear_skills=False,
            )
        )
        updated = get_job(job["id"])
        assert updated["skills"] == ["maps", "blogwatcher"]
        assert updated["name"] == "Edited Job"
        assert updated["prompt"] == "Revised prompt"
        assert updated["schedule_display"] == "every 120m"

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule=None,
                prompt=None,
                name=None,
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                clear_skills=True,
            )
        )
        cleared = get_job(job["id"])
        assert cleared["skills"] == []
        assert cleared["skill"] is None

        out = capsys.readouterr().out
        assert "Updated job" in out

    def test_create_with_multiple_skills(self, tmp_cron_dir, capsys):
        cron_command(
            Namespace(
                cron_command="create",
                schedule="every 1h",
                prompt="Use both skills",
                name="Skill combo",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["blogwatcher", "maps"],
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out

        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["skills"] == ["blogwatcher", "maps"]
        assert jobs[0]["name"] == "Skill combo"


class TestCronCommandNameResolution:
    """OQ-26: cron run/pause/resume/remove/edit accept job names, not just IDs."""

    def test_run_resolves_by_name(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Daily brief", schedule="every 24h", name="daily-brief"
        )

        rc = cron_command(Namespace(cron_command="run", job_id="daily-brief"))
        assert rc == 0

        out = capsys.readouterr().out
        assert "Triggered job" in out
        assert job["id"] in out  # canonical ID surfaced in output

    def test_pause_resume_remove_resolve_by_name(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Heartbeat", schedule="every 1h", name="heartbeat"
        )

        cron_command(Namespace(cron_command="pause", job_id="heartbeat"))
        assert get_job(job["id"])["state"] == "paused"

        cron_command(Namespace(cron_command="resume", job_id="heartbeat"))
        assert get_job(job["id"])["state"] == "scheduled"

        cron_command(Namespace(cron_command="remove", job_id="heartbeat"))
        assert get_job(job["id"]) is None

    def test_edit_resolves_by_name(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Edit me", schedule="every 1h", name="edit-target"
        )

        cron_command(
            Namespace(
                cron_command="edit",
                job_id="edit-target",
                schedule=None,
                prompt="New prompt body",
                name=None,
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                clear_skills=False,
            )
        )
        assert get_job(job["id"])["prompt"] == "New prompt body"

    def test_id_still_works(self, tmp_cron_dir, capsys):
        """Regression guard: passing a literal ID must still resolve to itself."""
        job = create_job(prompt="By ID", schedule="every 1h", name="by-id")
        rc = cron_command(Namespace(cron_command="run", job_id=job["id"]))
        assert rc == 0
        assert "Triggered job" in capsys.readouterr().out

    def test_unknown_id_or_name_suggests_close_match(self, tmp_cron_dir, capsys):
        create_job(prompt="Daily brief", schedule="every 24h", name="daily-brief")

        rc = cron_command(Namespace(cron_command="run", job_id="dailybrief"))
        assert rc == 1

        err = capsys.readouterr().out
        assert "No job matches" in err
        assert "daily-brief" in err  # difflib suggestion

    def test_unknown_with_no_close_match_errors_cleanly(self, tmp_cron_dir, capsys):
        create_job(prompt="Daily brief", schedule="every 24h", name="daily-brief")

        rc = cron_command(Namespace(cron_command="run", job_id="zzzzzzzzzz"))
        assert rc == 1
        assert "No job matches" in capsys.readouterr().out

    def test_duplicate_names_force_id_disambiguation(self, tmp_cron_dir, capsys):
        # Two jobs with the same name — name lookup MUST refuse to guess.
        j1 = create_job(prompt="A", schedule="every 1h", name="dupe")
        j2 = create_job(prompt="B", schedule="every 1h", name="dupe")
        assert j1["id"] != j2["id"]

        rc = cron_command(Namespace(cron_command="run", job_id="dupe"))
        assert rc == 1

        msg = capsys.readouterr().out
        assert "Multiple jobs share the name" in msg

        # Both jobs are still operable by ID.
        assert cron_command(Namespace(cron_command="run", job_id=j1["id"])) == 0

    def test_disabled_job_resolves_by_name(self, tmp_cron_dir, capsys):
        """Resume must work for paused/disabled jobs — list_jobs(include_disabled=True)."""
        job = create_job(prompt="Sleeper", schedule="every 1h", name="sleeper")
        cron_command(Namespace(cron_command="pause", job_id=job["id"]))
        # Now paused; resume by name should still resolve.
        rc = cron_command(Namespace(cron_command="resume", job_id="sleeper"))
        assert rc == 0
        assert get_job(job["id"])["state"] == "scheduled"
