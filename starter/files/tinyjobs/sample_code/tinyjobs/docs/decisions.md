# Decisions

Why tinyjobs is the way it is. Newest last.

## The queue is a JSON file

The queue is saved as JSON between commands, not in a database. It is small,
it reads by eye, and it needs no dependency. The file is written to a
temporary name and renamed, so a crash never leaves half a file.

## Jobs run one at a time

A job runs alone, oldest first. There is no parallelism, so a job's status
changes in one place and the file never has two writers.

## A failed job is tried three times

When a worker runs a job that fails, it tries again, waiting half a second
and then a second. After the third failure the job is marked failed with the
last error. `tinyjobs run` does not retry; only `tinyjobs work` does.

## Finished jobs do not change

Once a job has succeeded, failed or been cancelled, its status never changes.
