import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE


def main():
    print(
        json.dumps(
            {
                "wan_module": getattr(wan, "__file__", None),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "flash_attn_2": FLASH_ATTN_2_AVAILABLE,
                "flash_attn_3": FLASH_ATTN_3_AVAILABLE,
                "config_keys": sorted(list(WAN_CONFIGS.keys())),
                "sizes": {key: list(value) for key, value in SIZE_CONFIGS.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

