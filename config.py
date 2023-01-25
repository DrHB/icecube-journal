from pathlib import Path
from icecube.dataset import IceCubeCasheDatasetV0
from pathlib import Path
from icecube.dataset import collate_fn
import torch
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch import nn
from icecube.utils import get_score
from icecube.models import IceCubeModelEncoderV1
from icecube.utils import fit


class BASELINE_CONFIG:
    EXP_NAME = "EXP_00"
    FOLDER = Path("RESULTS")
    DATA_CACHE_DIR = Path("data/cache")
    BATCH_SIZE = 1024
    NUM_WORKERS = 16
    PRESISTENT_WORKERS = True
    LR = 1e-3
    WD = 1e-5
    WARM_UP_PCT = 0.1
    EPOCHS = 10
    VAL_BATCH_RANGE = (3,6)
    TRN_BATCH_RANGE = (7,100)
    TRN_DATASET = IceCubeCasheDatasetV0
    VAL_DATASET = IceCubeCasheDatasetV0
    COLLAT_FN = collate_fn

    OPT = torch.optim.AdamW
    LOSS_FUNC = nn.MSELoss
    SCHEDULER = get_cosine_schedule_with_warmup
    MODEL_NAME = IceCubeModelEncoderV1
    METRIC = get_score
    DEVICE = "cuda:0"
    FIT_FUNC = fit
