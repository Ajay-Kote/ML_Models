# GPU Setup (for the RTX 4090 machine)

Having a GPU in the machine is NOT enough by itself -- both `torch` and
`paddlepaddle`'s default pip packages are CPU-only. You need the GPU-specific
builds installed instead. Do this BEFORE running `pip install -r requirements.txt`
(or after -- pip install order doesn't matter, just make sure the GPU versions
are what's actually present at the end).

## 1. Check the driver + CUDA version
```powershell
nvidia-smi
```
Look at the top-right of the output for "CUDA Version: XX.X" -- that's the
max CUDA version your driver supports. (RTX 4090 needs a reasonably recent
driver; if `nvidia-smi` doesn't run at all, update NVIDIA drivers first from
nvidia.com.)

## 2. Install PyTorch with CUDA support
Go to https://pytorch.org/get-started/locally/ , select: Windows / Pip / Python
/ your CUDA version (e.g. CUDA 12.4), and it gives you the exact command.
It looks like:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
(Replace `cu124` with whatever version the site gives you based on your driver.)

**Verify it worked:**
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True RTX 4090` (or similar). If it prints `False`, the CUDA
wheel didn't install correctly -- redo step 2 with the exact command from
the PyTorch site.

## 3. Install PaddlePaddle with GPU support
The regular `paddlepaddle` pip package (what's in `requirements.txt`) is
CPU-only. Uninstall it and install `paddlepaddle-gpu` instead, matching your
CUDA version. Check the official install matrix at:
https://www.paddlepaddle.org.cn/en/install/quick

General pattern:
```powershell
pip uninstall paddlepaddle
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
(the `cu126` part in the URL must match your CUDA version from step 1 --
check the install matrix page for the exact index URL for your CUDA version.)

**Verify it worked:**
```powershell
python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```
Should print `True 1` (or however many GPUs).

## 4. Install everything else
```powershell
pip install -r requirements.txt
```
This will try to (re)install plain `torch`/`paddlepaddle` since they're listed
there too -- if pip says "Requirement already satisfied" for them because your
GPU versions are already present, that's fine, leave it. If it tries to
downgrade/replace your GPU build with the CPU one, remove the `torch>=2.1.0`
and `paddlepaddle==3.3.1` lines from `requirements.txt` before running this
step, since you've already installed the correct GPU versions manually.

## 5. Run training with GPU-optimized settings
Once both checks in steps 2 and 3 print `True`, `train.py` will auto-detect
and use the GPU for both OCR and visual embeddings -- no flag needed. You can
also tune parallelism:
```powershell
python -m models.train --manifest data/labels_all.csv --out_dir models/artifacts --workers 8 --embed_batch_size 64
```
- `--workers 8`: parallel OCR extraction threads (start around your CPU core
  count; more isn't always faster once the GPU/CPU is saturated -- try 4, 8, 16
  and see what's fastest on this machine).
- `--embed_batch_size 64`: how many images the EfficientNet-B0 embedder
  processes per GPU forward pass. An RTX 4090 has plenty of VRAM -- 64 or
  even 128 should be fine for this dataset's small (492) size. If you hit an
  out-of-memory error, lower it.
- `--lgbm_gpu`: optional flag to try training LightGBM itself on GPU. The
  standard pip `lightgbm` wheel does NOT support GPU -- this will print a
  fallback message and use CPU automatically unless LightGBM was specially
  built with GPU support. Not worth chasing for a dataset this size (LightGBM
  on CPU is already fast for ~500 rows) -- the real speedup on this machine
  comes from OCR/embeddings running on GPU, not from LightGBM.

You should see console output confirming what got used:
```
[PaymentOCRPipeline] Using device: gpu:0
[VisualEmbeddingService] Using GPU: NVIDIA GeForce RTX 4090
```
If either line instead says `cpu`, go back and recheck the corresponding
verification command in step 2 or 3.
