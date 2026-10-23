# tinyjobs

A small job queue with a command-line interface. Jobs are submitted, run in
order, and can be cancelled while they wait. The queue is saved to a JSON
file between commands.

```bash
tinyjobs submit "rebuild index"
tinyjobs run
tinyjobs status 1
tinyjobs list
```

## Layout

- `src/tinyjobs/jobs.py` — the `Job` record and the `JobQueue`.
- `src/tinyjobs/storage.py` — saving and loading the queue as JSON.
- `src/tinyjobs/cli.py` — the `tinyjobs` command.
- `src/tinyjobs/log.py` — logging setup.
- `tests/` — the test suite. Run it with `uv run pytest`.

## Configuration

`TINYJOBS_QUEUE_FILE` sets where the queue is saved (default
`.tinyjobs.json`). Secrets for the hosted runner live in `.env`, which is not
committed.
