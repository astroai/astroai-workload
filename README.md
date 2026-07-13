# astroai-workload

Submit a command as a **Ray job** on CANFAR, then check status, logs, or cancel.
The package also carries small helpers for resources (`memory="4GiB"`) and
provenance records.

## Install

```bash
pip install astroai-workload
```

## Example

```python
from astroai_workload import RayExecutor, ResourceRequest, RunSpec

job = RayExecutor().submit(
    RunSpec(
        run_id="mnist-001",
        command=("python", "train.py", "--epochs", "2"),
        resources=ResourceRequest(cpus=2, memory="4GiB"),
    )
)
print(RayExecutor().status(job))
```

On a **contributed `ray-manager` session** the Jobs address is already configured,
so `RayExecutor()` connects without extra setup.

## On CANFAR

1. `canfar login` (once, from webterm or vscode)
2. Start a **contributed `ray-manager` session** (Science Portal or `canfar create`)
3. Open the session URL from `canfar ps`, create workers, open the Dashboard
4. Run the example on that manager session, or submit the same entrypoint from
   Dashboard → Jobs

## Docs

- [MNIST example](examples/mnist_cnn/) — train, infer, and submit
- [Ray on CANFAR](https://github.com/astroai/astroai-containers/blob/main/docs/RAY.md) — cluster lifecycle
- [astroai-lab](https://github.com/astroai/astroai-lab) — session paths and hygiene

## License

[MIT](https://github.com/astroai/astroai-workload/blob/main/LICENSES/MIT.txt)
