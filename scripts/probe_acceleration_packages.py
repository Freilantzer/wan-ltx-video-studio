import importlib
import json

import torch


def probe_module(name):
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main():
    modules = ["triton", "flash_attn", "sageattention", "xformers"]
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "imports": {name: probe_module(name) for name in modules},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

