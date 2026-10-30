# tinyjobs

A small job queue with a command-line interface. Jobs are submitted, run in
order, and can be cancelled while they wait. The queue is saved to a JSON
file between commands.

```bash
tinyjobs submit "rebuild index"
tinyjobs run
tinyjobs status 1
tinyjobs list
tinyjobs report
```

`tinyjobs work` is `run` done slowly: each attempt takes a couple of seconds,
and a failed job is tried again.

## Layout

- `src/tinyjobs/jobs.py` — the `Job` record and the `JobQueue`.
- `src/tinyjobs/storage.py` — saving and loading the queue as JSON.
- `src/tinyjobs/retry.py` — trying work again when it fails.
- `src/tinyjobs/worker.py` — a worker that runs pending jobs, slowly, with retries.
- `src/tinyjobs/report.py` — a summary of the queue.
- `src/tinyjobs/cli.py` — the `tinyjobs` command.
- `src/tinyjobs/log.py` — logging setup.
- `docs/` — why it is the way it is: `decisions.md`, `architecture.md`.
- `tests/` — the test suite. Run it with `uv run pytest`.

## Configuration

`TINYJOBS_QUEUE_FILE` sets where the queue is saved (default
`.tinyjobs.json`). `TINYJOBS_WORK_SECONDS` sets how long `work` spends on each
attempt (default 2). Secrets for the hosted runner live in `.env`, which is not
committed.
