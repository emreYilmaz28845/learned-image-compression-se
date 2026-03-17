"""ScaleHyperprior with SE block in the synthesis transform.

Defines:
- SEScaleHyperprior: modified decoder with channel attention
- load_pretrained_with_se: transfer pretrained weights to SE model
- load_pretrained_baseline: load vanilla pretrained model for variant B
"""

import torch.nn as nn
from compressai.models import ScaleHyperprior
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from compressai.zoo import bmshj2018_hyperprior

from .se_block import SEBlock

QUALITY_TO_PARAMS = {
    1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192),
    5: (192, 320), 6: (192, 320), 7: (192, 320), 8: (192, 320),
}


class SEScaleHyperprior(ScaleHyperprior):
    """ScaleHyperprior with an SE block after the 2nd transposed conv in g_s.

    Baseline g_s:
        deconv(M,N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,3)

    Modified g_s:
        deconv(M,N) -> IGDN -> deconv(N,N) -> SE(N) -> IGDN -> deconv(N,N) -> IGDN -> deconv(N,3)
    """

    def __init__(self, N, M, reduction=16, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            SEBlock(N, reduction=reduction),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )


def load_pretrained_with_se(quality=3, reduction=16):
    """Load pretrained ScaleHyperprior and transfer weights to SEScaleHyperprior.

    The SE block is randomly initialized; all other weights come from the
    pretrained CompressAI checkpoint.
    """
    N, M = QUALITY_TO_PARAMS[quality]

    pretrained = bmshj2018_hyperprior(quality=quality, pretrained=True)
    se_model = SEScaleHyperprior(N=N, M=M, reduction=reduction)

    # Map baseline g_s indices to SE model g_s indices
    # Baseline: [0]=deconv [1]=GDN [2]=deconv [3]=GDN [4]=deconv [5]=GDN [6]=deconv
    # SE:       [0]=deconv [1]=GDN [2]=deconv [3]=SE  [4]=GDN [5]=deconv [6]=GDN [7]=deconv
    baseline_to_se = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7}

    se_state = se_model.state_dict()
    pretrained_state = pretrained.state_dict()

    new_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("g_s."):
            parts = key.split(".")
            old_idx = int(parts[1])
            if old_idx in baseline_to_se:
                new_idx = baseline_to_se[old_idx]
                new_key = "g_s." + str(new_idx) + "." + ".".join(parts[2:])
                new_state[new_key] = value
        else:
            new_state[key] = value

    se_state.update(new_state)
    se_model.load_state_dict(se_state)
    return se_model


def load_pretrained_baseline(quality=3):
    """Load a pretrained ScaleHyperprior baseline for fine-tuning (variant B)."""
    return bmshj2018_hyperprior(quality=quality, pretrained=True)
