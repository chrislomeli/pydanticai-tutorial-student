# Architecture

tinyjobs is a package under `src/tinyjobs/`.

| Module | What it holds |
|---|---|
| `jobs` | A job, and the queue that holds jobs and runs them |
| `storage` | Saving the queue to its JSON file, and loading it back |
| `retry` | Trying work again when it fails |
| `worker` | A worker that runs pending jobs, slowly, with retries |
| `report` | A summary of the queue |
| `cli` | The `tinyjobs` command |
| `log` | Where log messages go |

`cli` reads the queue with `storage`, runs it with `jobs` or `worker`, and
writes it back. Nothing else touches the file.
