# Docker Guide for KABS Enhance

This guide explains how to build and run the Docker environment for the Dual Re-ID system.

## 1. Build the Image

Run this command in the project root directory:

```bash
docker build -t kabs-enhance .
```

## 2. Run the Container

You need to mount your local directory to save results and access videos.

### For GPU (Recommended)
Requires NVIDIA Container Toolkit.

```bash
docker run -it --gpus all --ipc=host \
    -v "${PWD}:/workspace" \
    kabs-enhance
```

### For CPU (Slow)
If you don't have a GPU:

```bash
docker run -it \
    -v "${PWD}:/workspace" \
    kabs-enhance
```

**Note on Volume Mounting (`-v`):**
- `${PWD}:/workspace`: Maps your current folder to the container's workspace.
- This ensures that results saved in `output/` inside the container appear on your Windows machine.
- On Windows Command Prompt (cmd), replace `${PWD}` with `%cd%`.
- On PowerShell, `${PWD}` works fine.

## 3. Run Experiments

Once inside the container, you can run the comparison scripts:

### Compare Person Models
```bash
python compare_reid_models.py --video data/test_video.mp4 --output output/results_person --test-person
```

### Compare Vehicle Models
```bash
python compare_reid_models.py --video data/test_video.mp4 --output output/results_vehicle --test-vehicle
```

### Compare All Models
```bash
python compare_reid_models.py --video data/test_video.mp4 --output output/results_full --test-all
```

## Troubleshooting

- **Permission Denied**: If you can't write to files, check your folder permissions.
- **CUDA Error**: Ensure you have installed the NVIDIA Container Toolkit and your drivers are up to date.
