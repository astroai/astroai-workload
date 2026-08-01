"""Public AstroAI workload contracts.

Jobs + provenance (no Ray required to import) plus the CANFAR Ray cluster
lifecycle library moved out of the ray-manager container:

* :mod:`astroai_workload.cluster` — create/stop/scale CANFAR worker sessions
* :mod:`astroai_workload.canfar_ops` — CANFAR session API helpers
* :mod:`astroai_workload.manager_client` — drive a ray-manager over HTTP
* :mod:`astroai_workload.dashboard` — Ray Dashboard URL + local reverse proxy
* :mod:`astroai_workload.autoscaler` — Ray-native autoscaler (CANFAR provider)

Install extras: ``astroai-workload[cluster]`` (canfar client) for cluster
lifecycle, ``astroai-workload[jobs]`` (ray) for the Jobs client + autoscaler.
"""

__version__ = "0.2.0"

from .executor import RayExecutor, resolve_jobs_address, run_script
from .facility import ComputeFacility
from .models import (
    DataProductRef,
    ProvenanceManifest,
    ResourceRequest,
    RunSpec,
    RunStatus,
    format_memory,
    parse_memory,
)

__all__ = [
    "ComputeFacility",
    "DataProductRef",
    "ProvenanceManifest",
    "RayExecutor",
    "ResourceRequest",
    "RunSpec",
    "RunStatus",
    "format_memory",
    "parse_memory",
    "resolve_jobs_address",
    "run_script",
    "__version__",
]
