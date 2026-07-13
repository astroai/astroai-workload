# astroai-workload

Small Python contracts for submitting jobs to a **CANFAR Ray cluster** and
recording products/provenance. Import namespace: `astroai_workload`.

Ray is a required dependency:

```bash
pip install astroai-workload
# or: uv add astroai-workload
```

## How Ray fits on CANFAR (no host guessing)

The Ray head runs inside a **contributed `ray-manager` session** that **you**
launch on CANFAR (Science Portal or `canfar`). Workers are headless sessions the
manager starts for you. You never type pod IPs or invent names like
`ray-manager:8265`.

| You need | Where it comes from |
|----------|---------------------|
| Manager session URL | `canfar ps` → **connectURL** |
| Ray Dashboard (browser) | `connectURL/dashboard/` (trailing slash) |
| Jobs API for `RayExecutor` | Set automatically on the manager as `ASTROAI_RAY_JOBS_ADDRESS=http://127.0.0.1:8265` |

Cluster lifecycle docs: [astroai-containers/docs/RAY.md](https://github.com/astroai/astroai-containers/blob/main/docs/RAY.md).

### Student loop

1. **Once** (from webterm/vscode on CANFAR): `canfar login`  
   (`canfar auth login` is only a deprecated alias — prefer `canfar login`.)
2. **Laptop / portal:** launch contributed image `images.canfar.net/astroai/ray-manager:<tag>`  
   or run `scripts/ray-launch.sh` from the containers repo.
3. Open the **connectURL** from `canfar ps`.
4. In the browser: confirm auth, run network preflight, **create workers**.
5. Open the stock **Ray Dashboard** at `…/dashboard/`.
6. Submit work:
   - Dashboard → Jobs, or
   - `RayExecutor()` from code running on the manager (address is already set).

## Friendly submit

```python
from astroai_workload import RayExecutor, ResourceRequest, RunSpec

spec = RunSpec(
    run_id="mnist-001",
    command=("python", "train.py", "--epochs", "2"),
    resources=ResourceRequest(cpus=2, memory="4GiB"),
)
# No host:port — uses ASTROAI_RAY_JOBS_ADDRESS on the ray-manager session.
run_id = RayExecutor().submit(spec)
print(RayExecutor().status(run_id))
```

`memory=` accepts human sizes (`4GiB`, `512MiB`, `8GB`) or an int byte count.
`memory_bytes` remains the canonical field in JSON metadata.

Ray Jobs schedules driver CPU/GPU/custom resources. Memory and wall-time are
recorded in job metadata for facility policy; Ray does not hard-enforce them.

## MNIST example

See [`examples/mnist_cnn/`](examples/mnist_cnn/) for a short train + infer path
you can submit after workers are up.

## What this package is not

It does not create clusters, own science DAGs, or replace `canfar` / AstroAI
session images. Session hygiene and storage: [`astroai-lab`](https://github.com/sfabbro/canfar-lab).
Images and Ray manager/worker: [`astroai-containers`](https://github.com/astroai/astroai-containers).

## Benchmark

On an existing two-worker cluster:

```bash
python benchmarks/bench_ray_contract.py --address auto \
  --scratch-root /scratch/astroai-workload-benchmark-<run-id>
```

## License

MIT. Third-party dependencies retain their own licenses.
