# Ray Jobs smoke (no ML deps)

Minimal end-to-end check that `astroai-workload` can submit a Job on a live
**ray-manager** cluster.

## Prerequisites

1. `canfar login`
2. Launch `images.canfar.net/astroai/ray-manager:<tag>` (≥8 GiB)
3. Open the connect URL → create workers
4. Run these commands **inside the manager session**

## Run

```bash
# CLI (preferred — on PATH in current ray-manager images)
astroai-workload run job.py --cpus 1 --memory 1GiB

# or Python helper
python submit.py
```

Expected: status `succeeded` and a log line `astroai-workload smoke ok`.
