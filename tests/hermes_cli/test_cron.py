"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.cron import cron_command, cron_list


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


def _stub_list_jobs(jobs):
    def _inner(include_disabled=False):
        return list(jobs)
    return _inner


def _stub_no_gateway(monkeypatch):
    """cron_list calls find_gateway_pids at the bottom; stub it out so the
    test doesn't depend on whether a gateway happens to be running on the
    machine running pytest."""
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [1])


class TestCronListErrorRendering:
    """OQ-33: cron list must render state=='error' distinctly from [active]."""

    def test_error_state_renders_with_excerpt(self, monkeypatch, capsys):
        job = {
            "id": "abc12345",
            "name": "Recurring with broken croniter",
            "schedule": {"value": "0 9 * * *", "kind": "cron"},
            "schedule_display": "0 9 * * *",
            "next_run_at": "2026-05-04T09:00:00Z",
            "enabled": True,
            "state": "error",
            "last_error": "compute_next_run returned None — schedule could not be advanced",
            "deliver": ["local"],
            "repeat": {},
        }
        monkeypatch.setattr("cron.jobs.list_jobs", _stub_list_jobs([job]))
        _stub_no_gateway(monkeypatch)

        cron_list()
        out = capsys.readouterr().out

        assert "[error]" in out
        # excerpt portion should appear next to the [error] tag
        assert "compute_next_run returned None" in out
        # must NOT regress to [active] for an error-state job
        assert "[active]" not in out

    def test_missing_state_falls_back_to_active(self, monkeypatch, capsys):
        # Legacy jobs persisted without a 'state' field default to scheduled
        # via the cron_list fallback (state = "scheduled" if enabled).  This
        # guards against accidentally breaking pre-OQ-25 jobs.json files.
        job = {
            "id": "legacy01",
            "name": "Pre-OQ-25 job",
            "schedule": {"value": "every 1h", "kind": "interval"},
            "schedule_display": "every 60m",
            "next_run_at": "2026-05-03T13:00:00Z",
            "enabled": True,
            # no "state" key
            "deliver": ["local"],
            "repeat": {},
        }
        monkeypatch.setattr("cron.jobs.list_jobs", _stub_list_jobs([job]))
        _stub_no_gateway(monkeypatch)

        cron_list()
        out = capsys.readouterr().out

        assert "[active]" in out
        assert "[error]" not in out

    def test_long_last_error_is_truncated(self, monkeypatch, capsys):
        long_msg = "x" * 200
        job = {
            "id": "longerr0",
            "name": "Very loud failure",
            "schedule": {"value": "*/5 * * * *", "kind": "cron"},
            "schedule_display": "*/5 * * * *",
            "next_run_at": "2026-05-03T13:05:00Z",
            "enabled": True,
            "state": "error",
            "last_error": long_msg,
            "deliver": ["local"],
            "repeat": {},
        }
        monkeypatch.setattr("cron.jobs.list_jobs", _stub_list_jobs([job]))
        _stub_no_gateway(monkeypatch)

        cron_list()
        out = capsys.readouterr().out

        assert "[error]" in out
        assert "..." in out
        # the full 200-char message must NOT appear verbatim — that would
        # blow up list-view width.  Truncated form is 77 chars + "...".
        assert long_msg not in out
        assert "x" * 77 in out


