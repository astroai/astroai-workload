# MNIST CNN on CANFAR Ray

Tiny train + infer example for `astroai-workload` after a **ray-manager**
cluster has workers.

## Prerequisites

1. `canfar login` (once, from webterm/vscode).
2. Launch contributed `ray-manager`, open **connectURL** from `canfar ps`.
3. Create workers in the manager UI; open `connectURL/dashboard/`.
4. Run these scripts **on the ray-manager session** (Jobs address is already
   `ASTROAI_RAY_JOBS_ADDRESS=http://127.0.0.1:8265`).

Torch is **not** a package dependency — install it for the job runtime, e.g.:

```bash
pip install torch torchvision
# or pin in a pixi/uv project and set working_directory accordingly
```

## Run

```bash
cd examples/mnist_cnn
python submit.py          # submits train.py via RayExecutor()
python infer.py --ckpt mnist.pt
```

Or submit `train.py` from the Ray Dashboard Jobs UI with the same entrypoint.
