# astroai-workload

`astroai-workload` supplies small, serializable contracts for submitting
astronomical workloads and recording their products and provenance. It does not
define science DAGs, dataframe abstractions, experiment tracking, or cluster
lifecycle management.

The import namespace is `astroai_workload`. Ray support is optional:

```bash
pip install 'astroai-workload[ray]'
```

```python
from astroai_workload import RayExecutor, ResourceRequest, RunSpec

spec = RunSpec(
    run_id="example-001",
    command=("python", "pipeline.py", "--partition", "12"),
    resources=ResourceRequest(cpus=4, memory_bytes=8_000_000_000),
)
run_id = RayExecutor("http://ray-manager:8265").submit(spec)
```

The Ray Jobs adapter schedules driver CPU, GPU, and custom resources. Portable
memory and wall-time requests are recorded in job metadata because Ray Jobs does
not enforce them directly; a facility wrapper may enforce them at allocation
time.

Cluster images and server-side cluster lifecycle remain owned by `containers`.
Science applications remain responsible for stages, partitions, checkpoints,
and scientific acceptance.

The package is available under the MIT License. Third-party dependencies retain
their own licenses and notices.
