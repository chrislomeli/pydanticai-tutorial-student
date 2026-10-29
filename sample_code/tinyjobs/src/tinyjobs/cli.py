"""The `tinyjobs` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tinyjobs import log, storage
from tinyjobs.jobs import Job, JobQueue, UnknownJob


def _print_job(job: Job) -> None:
    line = f"{job.id}\t{job.status.value}\t{job.name}"
    if job.error:
        line += f"\t({job.error})"
    print(line)


def _pretend_work(job: Job) -> None:
    """Stand-in for real work: a job whose name starts with 'fail' fails."""
    if job.name.startswith("fail"):
        raise RuntimeError("asked to fail")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyjobs", description="A small job queue.")
    parser.add_argument(
        "--queue-file",
        help="where the queue is saved (default: $TINYJOBS_QUEUE_FILE or .tinyjobs.json)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="add a job to the queue")
    submit.add_argument("name")

    run = commands.add_parser("run", help="run pending jobs, oldest first")
    run.add_argument("--max", type=int, default=None, help="run at most this many")

    status = commands.add_parser("status", help="show one job")
    status.add_argument("job_id", type=int)

    cancel = commands.add_parser("cancel", help="cancel a job that has not started")
    cancel.add_argument("job_id", type=int)

    commands.add_parser("list", help="show every job")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log.configure()
    path = storage.queue_file() if args.queue_file is None else Path(args.queue_file)
    queue: JobQueue = storage.load(path)

    try:
        if args.command == "submit":
            _print_job(queue.submit(args.name))
        elif args.command == "run":
            ran = 0
            while args.max is None or ran < args.max:
                job = queue.run_next(_pretend_work)
                if job is None:
                    break
                _print_job(job)
                ran += 1
        elif args.command == "status":
            _print_job(queue.get(args.job_id))
        elif args.command == "cancel":
            _print_job(queue.cancel(args.job_id))
        elif args.command == "list":
            for job in queue:
                _print_job(job)
    except UnknownJob as error:
        print(f"no such job: {error.args[0]}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    storage.save(queue, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
