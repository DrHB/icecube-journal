from pathlib import Path
from icecube.dataset import (
    HuggingFaceDatasetV0,
    HuggingFaceDatasetV1,
    HuggingFaceDatasetV2,
    HuggingFaceDatasetV3,
    HuggingFaceDatasetV4,
    HuggingFaceDatasetGraphV0,
    HuggingFaceDatasetGraphV1
)

from pathlib import Path
from icecube.utils import collate_fn, collate_fn_v1, collate_fn_graphv0
import torch
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from icecube.models import (
    IceCubeModelEncoderV1,
    LogCoshLoss,
    IceCubeModelEncoderV0,
    IceCubeModelEncoderSensorEmbeddinng,
    IceCubeModelEncoderSensorEmbeddinngV1,
    IceCubeModelEncoderSensorEmbeddinngV2,
    IceCubeModelEncoderSensorEmbeddinngV3,
    IceCubeModelEncoderMAT,
    IceCubeModelEncoderMATMasked
)
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
    TRN_BATCH_RANGE = (1, 600)
    VAL_BATCH_RANGE = (622, 627)

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


class BASELINE_EMBED_V0(BASELINE_HF_V2):
    BATCH_SIZE = 1024 + 256
    EXP_NAME = "EXP_03"
    LOSS_FUNC = LogCoshLoss

    COLLAT_FN = collate_fn_v1
    TRN_DATASET = HuggingFaceDatasetV1
    VAL_DATASET = HuggingFaceDatasetV1
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinng


class BASELINE_EMBED_V1(BASELINE_HF_V2):
    EXP_NAME = "EXP_04"

    COLLAT_FN = collate_fn_v1
    TRN_DATASET = HuggingFaceDatasetV2
    VAL_DATASET = HuggingFaceDatasetV2
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV1


class BASELINE_EMBED_V2(BASELINE_HF_V2):
    EXP_NAME = "EXP_05"

    COLLAT_FN = collate_fn_v1
    TRN_DATASET = HuggingFaceDatasetV3
    VAL_DATASET = HuggingFaceDatasetV3
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV2



class MATGRAPH(BASELINE_HF_V2):
    EXP_NAME = "EXP_06"
    LOSS_FUNC = LogCoshLoss
    TRN_DATASET = HuggingFaceDatasetGraphV0
    VAL_DATASET = HuggingFaceDatasetGraphV0
    MODEL_NAME = IceCubeModelEncoderMAT
    COLLAT_FN = collate_fn_graphv0

    TRN_BATCH_RANGE = (1, 100)
    VAL_BATCH_RANGE = (622, 627)
    EPOCHS = 10


class MATGRAPHV2(MATGRAPH):
    EXP_NAME = "EXP_07"
    MODEL_NAME = IceCubeModelEncoderMATMasked

class MATGRAPHV3(MATGRAPHV2):
    EXP_NAME = "EXP_08"
    MODEL_NAME = IceCubeModelEncoderMATMasked
    TRN_DATASET = HuggingFaceDatasetGraphV1
    VAL_DATASET = HuggingFaceDatasetGraphV1


class BASELINE_EMBED_V3(BASELINE_HF_V2):
    EXP_NAME = "EXP_09"

    COLLAT_FN = collate_fn_v1
    TRN_DATASET = HuggingFaceDatasetV3
    VAL_DATASET = HuggingFaceDatasetV3
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV2
    TRN_BATCH_RANGE = (1, 100)
    VAL_BATCH_RANGE = (622, 627)
    EPOCHS = 10

class BASELINE_EMBED_V4(BASELINE_EMBED_V3):
    EXP_NAME = "EXP_10"
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV3

class BASELINE_EMBED_V5(BASELINE_EMBED_V3):
    EXP_NAME = "EXP_11"
    TRN_DATASET = HuggingFaceDatasetV4
    VAL_DATASET = HuggingFaceDatasetV4





