"""Background execution of multi-item pipeline runs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from src.db.stores import Job, JobStore
from src.logging.logger import get_logger
from src.services.pipeline import PipelineService

logger = get_logger(__name__)


class JobService:
    """Queues pipeline batches on a single worker thread and tracks progress in ``jobs``.

    One worker matches the pipeline lock: jobs run one after another, and HTTP
    requests return immediately with a job id that clients poll.
    """

    def __init__(self, pipeline: PipelineService, store: JobStore) -> None:
        self.pipeline = pipeline
        self.store = store
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stockpilot-job")
        self._futures: dict[str, Future] = {}

    def submit(self, item_ids: list[str] | None = None) -> Job:
        """Queue a job (default: the demo items).

        Raises:
            KeyError: if any item is unknown (nothing is queued).
        """
        ids = list(dict.fromkeys(item_ids or self.pipeline.repository.item_ids))
        unknown = [i for i in ids if not self.pipeline.repository.has_item(i)]
        if unknown:
            raise KeyError(f"Unknown item_id(s): {', '.join(unknown)}")
        job = self.store.create(ids)
        self._futures[job.job_id] = self._executor.submit(self._execute, job.job_id, ids)
        return job

    def _execute(self, job_id: str, item_ids: list[str]) -> None:
        self.store.mark_running(job_id)
        try:
            for item_id in item_ids:
                try:
                    result = self.pipeline.run_item(item_id)
                    self.store.record_progress(job_id, run_id=result.run_id)
                except Exception as exc:  # noqa: BLE001 - one bad item must not stop the batch
                    logger.exception("job_item_failed", job_id=job_id, item_id=item_id)
                    self.store.record_progress(job_id, error=f"{item_id}: {exc}")
            self.store.finish(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("job_failed", job_id=job_id)
            self.store.finish(job_id, error=str(exc))
        finally:
            self._futures.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        """Current job state."""
        return self.store.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        """Recent jobs."""
        return self.store.list(limit)

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        """Block until a job finishes (tests and scripts)."""
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(job_id)

    def shutdown(self, wait: bool = False) -> None:
        """Stop accepting work; optionally wait for the running job."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
