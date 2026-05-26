import json
import os
import platform
import subprocess
import sys


def run(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def main():
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": run(["nvcc", "--version"]),
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_built": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            report["torch"]["device"] = {
                "index": device,
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_mib": props.total_memory // (1024 * 1024),
            }
            x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            y = x @ x
            torch.cuda.synchronize()
            report["torch"]["matmul_checksum"] = float(y.float().mean().detach().cpu())
    except Exception as exc:
        report["torch_error"] = repr(exc)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

