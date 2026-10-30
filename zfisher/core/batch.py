"""Run a batch template's datasets, one after another or several at once.

``run_batch`` runs each dataset of a ``params.BatchConfig`` through the whole
pipeline (``api.run_pipeline``), each into its own output folder with its own
session and session log, and returns one ``DatasetResult`` per dataset, failed
ones included, in the template's order. With ``workers`` above 1 the datasets
run in separate processes (started fresh, ``spawn``, so nothing of the parent,
a GUI included, is shared); each holds its two rounds in memory, so the number
of workers is bounded by memory, not by cores. The CLI and the Batch Process
widget both run batches through here.
"""
import logging
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    name: str
    output_dir: str
    ok: bool
    error: str = ""
    seconds: float = 0.0
    stages_run: list = field(default_factory=list)


def dataset_jobs(config, output_base):
    """``(name, DatasetSpec, output folder, puncta config, pairwise rules, tri rules)``
    for each dataset of ``config``, in its order."""
    from .generate_batch_template import resolve_coloc_for_dataset, resolve_puncta_for_dataset
    jobs = []
    for ds in config.datasets:
        pairwise, tri = resolve_coloc_for_dataset(ds.name, config.coloc_rules)
        jobs.append((ds.name, ds, Path(ds.output_dir or Path(output_base) / ds.name),
                     resolve_puncta_for_dataset(ds.name, config.puncta_rules), pairwise, tri))
    return jobs


def _run_one(job, progress_callback=None, actor=None):
    """Run one dataset; never raises. Top level so worker processes can run it."""
    from . import decisions, pipeline
    from .generate_batch_template import pipeline_kwargs
    name, ds, out, puncta_config, pairwise, tri = job
    if actor:
        decisions.set_process_actor(actor)
    start = time.monotonic()
    try:
        out.mkdir(parents=True, exist_ok=True)
        ran = pipeline.run_full_zfisher_pipeline(output_dir=out, puncta_config=puncta_config,
                                                 pairwise_rules=pairwise, tri_rules=tri,
                                                 progress_callback=progress_callback, **pipeline_kwargs(ds))
        return DatasetResult(name, str(out), True, seconds=time.monotonic() - start, stages_run=list(ran or []))
    except Exception as exc:
        logger.error("Dataset '%s' failed: %s", name, exc, exc_info=True)
        return DatasetResult(name, str(out), False, error=f"{type(exc).__name__}: {exc}",
                             seconds=time.monotonic() - start)
    except BaseException:                       # a worker must report, not vanish
        return DatasetResult(name, str(out), False, error=traceback.format_exc(limit=1),
                             seconds=time.monotonic() - start)


def run_batch(config, output_base, workers=1, on_progress=None, on_result=None):
    """Run every dataset of ``config`` into ``output_base`` (see the module
    docstring). ``on_progress(name, percent, text)`` reports within a dataset
    (only when ``workers`` is 1: progress does not cross processes);
    ``on_result(result)`` is called as each dataset finishes. Returns the
    results in the template's order."""
    from . import decisions
    jobs = dataset_jobs(config, output_base)
    Path(output_base).mkdir(parents=True, exist_ok=True)
    results = {}
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            callback = (lambda p, t, _n=job[0]: on_progress(_n, p, t)) if on_progress else None
            result = _run_one(job, callback)
            results[job[0]] = result
            if on_result:
                on_result(result)
    else:
        context = multiprocessing.get_context("spawn")
        actor = decisions.current_actor()
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
            futures = {pool.submit(_run_one, job, None, actor): job[0] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results[futures[future]] = result
                if on_result:
                    on_result(result)
    return [results[job[0]] for job in jobs]
