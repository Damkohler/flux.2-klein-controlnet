# FLUX.2 Klein ControlNet

This implementation conditions on a masked source image. RGB VAE inputs use
`[-1,1]`; masks use `>= 0.5` for visible pixels and three binary channels.
Condition tokens retain 128 latent features plus three mask features.

Periodic checkpoints are model directories containing configuration and
safetensors weights. The final directory retains the existing `lastest` name.
Standalone inference requires `--controlnet <checkpoint-directory>`.

Residual routing uses `block_index * head_count // block_count`. When heads do
not outnumber base blocks, every head is consumed. This changes routing semantics
from the former ceiling-interval mapping. Parameter shapes remain unchanged, but
checkpoints trained with the old mapping cannot be assumed to reproduce their
original behavior with this routing or the corrected preprocessing.

## CPU regression tests

Use the existing Python/PyTorch/Diffusers environment on Windows. The suite uses
reduced native transformer classes, synthetic text/images, and a VAE fixture;
temporary synthetic checkpoints are removed. Windows process-memory counters
enforce a conservative 2 GiB limit. No pretrained weights are needed.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -B -u -m unittest discover -s tests -p test_native_contracts.py -v
```

The tests cover conditioning contracts, positional roles, sigma indexing,
residual routing, staged gradients, checkpointed backward, and serialization.
They do not establish pretrained 9B feasibility, convergence, or pose control.
Nonbucket training, batches larger than one, multi-reference alignment, CFG
residual reuse, shortened initialization, and residual scaling remain deferred.
