from pathlib import Path
from icecube.dataset import HuggingFaceDatasetV0
from pathlib import Path
from icecube.utils import collate_fn
import torch
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from icecube.models import IceCubeModelEncoderV1, LogCoshLoss, IceCubeModelEncoderV0
from icecube.utils import fit_shuflle, get_score 
from torch import nn

class BASELINE_HF:
    EXP_NAME = "EXP_00"
    FOLDER = Path("RESULTS")
    DATA_CACHE_DIR = Path("data/hf_cashe")
    BATCH_SIZE = 1024 * 2
    NUM_WORKERS = 16
    PRESISTENT_WORKERS = True
    LR = 1e-3
    WD = 1e-5
    WARM_UP_PCT = 0.1
    EPOCHS = 3
    TRN_BATCH_RANGE = (1,600)
    VAL_BATCH_RANGE = (622,627)
    
    TRN_DATASET = HuggingFaceDatasetV0
    VAL_DATASET = HuggingFaceDatasetV0
    COLLAT_FN = collate_fn

    OPT = torch.optim.AdamW
    LOSS_FUNC = nn.MSELoss
    SCHEDULER = get_cosine_schedule_with_warmup
    MODEL_NAME = IceCubeModelEncoderV0
    METRIC = get_score
    DEVICE = "cuda:0"
    FIT_FUNC = fit_shuflle


class BASELINE_HF_V1(BASELINE_HF):
    EXP_NAME = "EXP_01"
    MODEL_NAME = IceCubeModelEncoderV1
    NUM_WORKERS = 24
    BATCH_SIZE = 1024 + 512


class BASELINE_HF_V2(BASELINE_HF_V1):
    EXP_NAME = "EXP_02"
    LOSS_FUNC = LogCoshLoss



