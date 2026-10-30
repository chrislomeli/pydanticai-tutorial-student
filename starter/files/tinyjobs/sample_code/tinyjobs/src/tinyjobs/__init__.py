"""tinyjobs: a small job queue with a command-line interface."""

from tinyjobs.jobs import Job, JobQueue, JobStatus, UnknownJob

__all__ = ["Job", "JobQueue", "JobStatus", "UnknownJob"]
__version__ = "0.4.0"
