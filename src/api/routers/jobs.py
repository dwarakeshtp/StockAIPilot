"""Background job endpoints (multi-item pipeline runs)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_services
from src.api.schemas import JobRequest, JobResponse
from src.services import Services

router = APIRouter(prefix="/jobs", tags=["jobs"])
Svc = Annotated[Services, Depends(get_services)]


@router.post("", response_model=JobResponse, status_code=202)
def submit_job(body: JobRequest, services: Svc) -> JobResponse:
    """Queue pipeline runs and return immediately; poll ``GET /jobs/{job_id}``."""
    item_ids = None if body.all_demo else (body.item_ids or None)
    try:
        job = services.jobs.submit(item_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None
    return JobResponse.of(job)


@router.get("", response_model=list[JobResponse])
def list_jobs(services: Svc,
              limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[JobResponse]:
    """Recent jobs, newest first."""
    return [JobResponse.of(j) for j in services.jobs.list(limit)]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, services: Svc) -> JobResponse:
    """Status and progress of a job."""
    job = services.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobResponse.of(job)
