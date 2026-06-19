# admm-implementation

this repository implements reconstruction methods from [Learned reconstructions for practical
mask-based lensless imaging](https://arxiv.org/pdf/1908.11502):

- ADMM-100 with fixed hyperparameters
- Le-ADMM-20 with trainable unrolled parameters
- 8M modular Le-ADMM-5 variants: pre+post, pre-only, post-only

the code uses the DigiCam-Mirflickr-MultiMask-10K dataset and supports
inference on a custom folder dataset. trained models can be found on [huggingface](https://huggingface.co/akhasanovv/admm-lensless-checkpoints).

## installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## comet.ml

for logging, use comet:

```bash
export COMET_API_KEY=...
```

## custom dataset

if you want to use custom dataset, it should have this structure:

```text
NameOfTheDirectoryWithData
├── lensless
│   ├── ImageID1.png
│   ├── ImageID2.png
│   .
│   .
│   .
│   └── ImageIDn.png
├── masks
│   ├── ImageID1.npy
│   ├── ImageID2.npy
│   .
│   .
│   .
│   └── ImageIDn.npy
└── lensed # ground truth original image, may not exist
    ├── ImageID1.png
    ├── ImageID2.png
    .
    .
    .
    └── ImageIDn.png
```

## training

default training uses DigiCam train split, evaluates on a small test subset, logs to comet, and saves checkpoints to `saved/<run_name>/`.

ADMM has fixed parameters and does not need training.

to train Le-ADMM-20, run:

```bash
python3 train.py \
  model=le_admm \
  writer=cometml \
  writer.project_name=lensless-admm \
  writer.run_name=le_admm20
```

to train modular Le-ADMM-5, choose your config and run one of the following commands:

```bash
python3 train.py \
  model=pre4_leadmm5_post4 \
  writer=cometml \
  writer.project_name=lensless-admm \
  writer.run_name=pre4_leadmm5_post4
```

```bash
python3 train.py \
  model=pre8_leadmm5 \
  writer=cometml \
  writer.project_name=lensless-admm \
  writer.run_name=pre8_leadmm5
```

```bash
python3 train.py \
  model=leadmm5_post8 \
  writer=cometml \
  writer.project_name=lensless-admm \
  writer.run_name=leadmm5_post8
```

## inference

predictions are saved to:

```text
data/saved/<inferencer.save_path>/<split>/
```

run a trained modular checkpoint:

```bash
python3 inference.py \
  model=pre4_leadmm5_post4 \
  datasets=digicam_eval \
  dataloader=lensless \
  inferencer.from_pretrained=saved/pre4_leadmm5_post4/model_best.pth \
  inferencer.save_path=pre4_leadmm5_post4_test
```

run inference on a custom dataset:

```bash
python3 inference.py \
  model=admm \
  datasets=custom_dir_eval \
  datasets.test.data_dir=/path/to/data_dir \
  dataloader=lensless \
  inferencer.save_path=custom_admm
```

## metrics

if ground truth is available in `lensed/`, calculate metrics for saved
reconstructions:

```bash
python3 calculate_metrics.py \
  --gt-dir /path/to/data_dir/lensed \
  --pred-dir data/saved/custom_admm/test \
  --lpips
```

during inference MSE, PSNR, SSIM, and LPIPS are also computed when the dataset contains `lensed` images and `metrics=reconstruction` is used.

## latency

to measure latency, run:

```bash
python3 benchmark_speed.py \
  model=admm \
  datasets=digicam_eval \
  dataloader=lensless \
  metrics=reconstruction_no_lpips \
  +benchmark.warmup=2 \
  +benchmark.max_batches=20
```

the script prints processed images, total time, milliseconds per image, and images per second.

## checkpoints

if you want to run inference on pretrained model (for example, `checkpoints/le_admm20.pth`), write something like this:

```bash
python3 inference.py \
  model=le_admm \
  datasets=custom_dir_eval \
  datasets.test.data_dir=/path/to/data_dir \
  dataloader=lensless \
  inferencer.from_pretrained=checkpoints/le_admm20.pth \
  inferencer.save_path=le_admm20_custom
```

## demo notebook

see [demo.ipynb](demo.ipynb). it clones the repository in colab, installs dependencies, downloads checkpoints, downloads a custom dataset zip from google drive, runs inference, visualizes samples, and prints metrics when `lensed/` exists.

## credits

based on [Petr Grinberg's template](https://github.com/Blinorot/pytorch_project_template).
