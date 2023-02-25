from pathlib import Path
from icecube.dataset import (
    HuggingFaceDatasetV0,
    HuggingFaceDatasetV1,
    HuggingFaceDatasetV2,
    HuggingFaceDatasetV3,
    HuggingFaceDatasetV4,
    HuggingFaceDatasetV5,
    HuggingFaceDatasetV6,
    HuggingFaceDatasetV7,
    HuggingFaceDatasetV8,
    HuggingFaceDatasetV9,
    HuggingFaceDatasetV10,
    HuggingFaceDatasetV11,
    HuggingFaceDatasetV12,
    HuggingFaceDatasetV13,
    HuggingFaceDatasetGraphV0,
    HuggingFaceDatasetGraphV1,
)

# from icecube.modelsgraph import EncoderWithReconstructionLossV0, CombineLossV0
from pathlib import Path
import torch
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from icecube.models import (
    LogCoshLoss,
    VonMisesFisher2DLossL1Loss,
    EuclideanDistanceLossG,
    VonMisesFisher3DLoss,
    VonMisesFisher3DLossEcludeLoss,
    VonMisesFisher3DLossEcludeLossCosine,
    VonMisesFisher3DLossCosineSimularityLoss,
    IceCubeModelEncoderV0,
    IceCubeModelEncoderV1,
    IceCubeModelEncoderV2,
    IceCubeModelEncoderV1CombinePool,
    EncoderWithDirectionReconstruction,
    EncoderWithDirectionReconstructionV1,
    EncoderWithDirectionReconstructionV2,
    EncoderWithDirectionReconstructionV3,
    EncoderWithDirectionReconstructionV4,
    IceCubeModelEncoderSensorEmbeddinng,
    IceCubeModelEncoderSensorEmbeddinngV1,
    IceCubeModelEncoderSensorEmbeddinngV2,
    IceCubeModelEncoderSensorEmbeddinngV3,
    IceCubeModelEncoderSensorEmbeddinngV4,
    IceCubeModelEncoderMAT,
    IceCubeModelEncoderMATMasked,
)

from icecube.modelsgraph import (
    DynEdgeV0,
    DynEdgeV1,
    EGNNModel,
    EGNNModelV2,
    EGNNModelV3,
    EGNNModelV4,
    EGNNModelV6,
    EGNNModelV7,
    EGNNModelV8,
    EGNNModelV9,
    GraphxTransformerV0,
    GraphxTransformerV1,
    gVonMisesFisher3DLossEcludeLoss,
    gVonMisesFisher3DLoss,
    gVonMisesFisher3DLossCosineSimularityLoss,
)
from icecube.graphdataset import GraphDasetV0, GraphDasetV1, GraphDasetV3

from icecube.utils import (
    fit_shuflle,
    get_score,
    gfit_shuflle,
    get_score_vector,
    get_score_v1,
    collate_fn,
    collate_fn_v1,
    collate_fn_graphv0,
    gget_score_vector,
    gget_score_save,
    eval_save,
    fit_shufllef32,
)
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


class BASELINE_HF_V3(BASELINE_HF_V1):
    EXP_NAME = "EXP_12"
    LOSS_FUNC = LogCoshLoss
    TRN_DATASET = HuggingFaceDatasetV5
    VAL_DATASET = HuggingFaceDatasetV5
    TRN_BATCH_RANGE = (1, 100)
    VAL_BATCH_RANGE = (622, 627)
    EPOCHS = 10


class BASELINE_HF_V4(BASELINE_HF_V3):
    EXP_NAME = "EXP_13"
    TRN_DATASET = HuggingFaceDatasetV7
    VAL_DATASET = HuggingFaceDatasetV7


class BASELINE_HF_V5(BASELINE_HF_V4):
    EXP_NAME = "EXP_14"
    MODEL_NAME = IceCubeModelEncoderV1CombinePool


class BASELINE_HF_V6(BASELINE_HF_V4):
    EXP_NAME = "EXP_15"
    MODEL_NAME = IceCubeModelEncoderV1CombinePool
    MODEL_WTS = "/opt/slh/icecube/RESULTS/EXP_14/EXP_14_9.pth"


class BASELINE_HF_V7(BASELINE_HF_V4):
    EXP_NAME = "EXP_16"
    MODEL_NAME = EncoderWithDirectionReconstruction
    LOSS_FUNC = VonMisesFisher3DLoss
    METRIC = get_score_vector
    TRN_DATASET = HuggingFaceDatasetV8
    VAL_DATASET = HuggingFaceDatasetV8
    MODEL_WTS = False


class BASELINE_HF_V8(BASELINE_HF_V7):
    EXP_NAME = "EXP_17"
    MODEL_WTS = "/opt/slh/icecube/RESULTS/EXP_16/EXP_16_9.pth"


class BASELINE_HF_V8FT(BASELINE_HF_V7):
    EXP_NAME = "EXP_18"
    MODEL_WTS = "/opt/slh/icecube/RESULTS/EXP_17/EXP_17_9.pth"
    TRN_BATCH_RANGE = (1, 600)
    VAL_BATCH_RANGE = (622, 627)
    EPOCHS = 5


class BASELINE_HF_V8FTEVAL(BASELINE_HF_V8FT):
    MODEL_WTS = "/opt/slh/icecube/RESULTS/EXP_18/EXP_18_4.pth"
    METRIC = gget_score_save
    FIT_FUNC = eval_save
    DEVICE = "cuda:1"


class BASELINE_HF_V9(BASELINE_HF_V7):
    EXP_NAME = "EXP_19"
    TRN_DATASET = HuggingFaceDatasetV9
    VAL_DATASET = HuggingFaceDatasetV9
    MODEL_NAME = EncoderWithDirectionReconstructionV1


class BASELINE_HF_V10(BASELINE_HF_V7):
    EXP_NAME = "EXP_20"
    TRN_DATASET = HuggingFaceDatasetV9
    VAL_DATASET = HuggingFaceDatasetV9
    MODEL_NAME = EncoderWithDirectionReconstructionV2
    LOSS_FUNC = VonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 20


class BASELINE_HF_V11(BASELINE_HF_V10):
    EXP_NAME = "EXP_21"
    LOSS_FUNC = VonMisesFisher3DLossEcludeLoss
    NUM_WORKERS = 22


class BASELINE_HF_V12(BASELINE_HF_V10):
    EXP_NAME = "EXP_22"
    LOSS_FUNC = VonMisesFisher3DLossEcludeLossCosine
    NUM_WORKERS = 22


class BASELINE_graph_V0(BASELINE_HF_V10):
    EXP_NAME = "EXP_23"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = DynEdgeV0
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0


class BASELINE_HF_V13(BASELINE_HF_V10):
    EXP_NAME = "EXP_24"
    LOSS_FUNC = VonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EncoderWithDirectionReconstructionV4
    DEVICE = "cuda:0"
    BATCH_SIZE = 1024 + 256
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    
class BASELINE_HF_V14(BASELINE_HF_V13):
    EXP_NAME = "EXP_24_FT"
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_24_FT_1.pth'
    TRN_BATCH_RANGE = [[300, 400], [400, 500], [500, 600], [1, 100], [100, 200], [200, 300], ]
    LR = 1e-4
    LOSS_FUNC = VonMisesFisher3DLossCosineSimularityLoss
    FIT_FUNC = fit_shufllef32
    EPOCHS = 6


    
class BASELINE_HF_V14_FT(BASELINE_HF_V13):
    EXP_NAME = "EXP_24_FT_2"
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_24_FT/EXP_24_FT_5.pth'
    TRN_BATCH_RANGE = [[300, 400], [400, 500], [500, 600], [1, 100], [100, 200], [200, 300], ]
    LR = 1e-4
    LOSS_FUNC = VonMisesFisher3DLossEcludeLossCosine
    FIT_FUNC = fit_shufllef32
    EPOCHS = 6
    TRN_DATASET = HuggingFaceDatasetV10
    VAL_DATASET = HuggingFaceDatasetV10
    BATCH_SIZE = 1024


class BASELINE_graph_V1(BASELINE_HF_V10):
    EXP_NAME = "EXP_25"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = DynEdgeV1
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    
class BASELINE_graph_V1_FT(BASELINE_HF_V10):
    EXP_NAME = "EXP_25_FT"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = DynEdgeV1
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    LR = 1e-4
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_25/EXP_25_9.pth'
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    
class BASELINE_graph_V2(BASELINE_HF_V10):
    EXP_NAME = "EXP_26"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModel
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    
class BASELINE_graph_V3(BASELINE_HF_V10):
    EXP_NAME = "EXP_27"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV2
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    
class BASELINE_graph_V4(BASELINE_HF_V10):
    EXP_NAME = "EXP_28"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV3
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12


class BASELINE_HF_V15(BASELINE_HF_V13):
    EXP_NAME = "EXP_29"
    MODEL_WTS = False
    TRN_BATCH_RANGE = [[300, 400], [400, 500], [500, 600], [1, 100], [100, 200], [200, 300]]
    LR = 1e-3
    LOSS_FUNC = VonMisesFisher3DLoss
    FIT_FUNC = fit_shuflle
    EPOCHS = 10
    TRN_DATASET = HuggingFaceDatasetV11
    VAL_DATASET = HuggingFaceDatasetV11
    BATCH_SIZE = 1024 + 512
    COLLAT_FN = collate_fn_v1
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV4
    
class BASELINE_HF_V15_FT(BASELINE_HF_V15):
    EXP_NAME = "EXP_29_FT"
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_29/EXP_29_5.pth'
    LR = 1e-4
    LOSS_FUNC = VonMisesFisher3DLossEcludeLossCosine
    FIT_FUNC = fit_shufllef32
    EPOCHS = 6
    TRN_DATASET = HuggingFaceDatasetV12
    VAL_DATASET = HuggingFaceDatasetV12
    COLLAT_FN = collate_fn_v1
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV4
    BATCH_SIZE = 768
    
    
class BASELINE_HF_V15_FT_2(BASELINE_HF_V15_FT):
    EXP_NAME = "EXP_29_FT_2"
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_29_FT/EXP_29_FT_5.pth'
    LR = (1e-4 + 1e-3)/2
    LOSS_FUNC = VonMisesFisher3DLossEcludeLossCosine
    FIT_FUNC = fit_shufllef32
    EPOCHS = 6
    TRN_DATASET = HuggingFaceDatasetV13
    VAL_DATASET = HuggingFaceDatasetV13
    COLLAT_FN = collate_fn_v1
    MODEL_NAME = IceCubeModelEncoderSensorEmbeddinngV4
    BATCH_SIZE = 512
    
class BASELINE_graph_V5(BASELINE_HF_V10):
    EXP_NAME = "EXP_30"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV4
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    
class BASELINE_graph_V6(BASELINE_graph_V5):
    EXP_NAME = "EXP_31"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV6
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    DEVICE = 'cuda:1'
    BATCH_SIZE = 1024
    
class BASELINE_graph_V7(BASELINE_HF_V10):
    EXP_NAME = "EXP_32"
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV7
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV1
    VAL_DATASET = GraphDasetV1
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    BATCH_SIZE = 1024 
    
    
class BASELINE_graph_V8(BASELINE_graph_V5):
    EXP_NAME = "EXP_33"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = GraphxTransformerV0
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV3
    VAL_DATASET = GraphDasetV3
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    DEVICE = 'cuda:1'
    BATCH_SIZE = 1024
    
class BASELINE_graph_V8_FT(BASELINE_graph_V8):
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_33/EXP_33_11.pth'
    EXP_NAME = "EXP_33_FT"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 6
    DEVICE = 'cuda:1'
    BATCH_SIZE = 756
    LR = 1e-4
    
    
class BASELINE_graph_V9(BASELINE_graph_V5):
    EXP_NAME = "EXP_34"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV8
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    DEVICE = 'cuda:0'
    
class BASELINE_graph_V9_FT(BASELINE_graph_V9):
    EXP_NAME = "EXP_34_FT"
    MODEL_WTS = '/opt/slh/icecube/RESULTS/EXP_34/EXP_34_11.pth'
    LR = 1e-4
    EPOCHS = 6
    LOSS_FUNC = gVonMisesFisher3DLossCosineSimularityLoss
    
    
class BASELINE_graph_V10(BASELINE_graph_V5):
    EXP_NAME = "EXP_35"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = EGNNModelV9
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV0
    VAL_DATASET = GraphDasetV0
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    DEVICE = 'cuda:0'
    
class BASELINE_graph_V11(BASELINE_graph_V5):
    EXP_NAME = "EXP_36"
    LOSS_FUNC = gVonMisesFisher3DLoss
    NUM_WORKERS = 22
    MODEL_NAME = GraphxTransformerV1
    FIT_FUNC = gfit_shuflle
    METRIC = gget_score_vector
    TRN_DATASET = GraphDasetV3
    VAL_DATASET = GraphDasetV3
    TRN_BATCH_RANGE = [[1, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600]]
    EPOCHS = 12
    DEVICE = 'cuda:1'
    BATCH_SIZE = 1024
    
    
    
    
    
    

# class BASELINE_HF_V11(BASELINE_HF_V4):
#    EXP_NAME = "EXP_21"
#    MODEL_NAME = EncoderWithDirectionReconstructionV3
#    LOSS_FUNC = VonMisesFisher2DLossL1Loss
#    METRIC = get_score_v1
#    TRN_BATCH_RANGE = (1, 100)
#    VAL_BATCH_RANGE = (622, 627)
#    MODEL_WTS = False
# 2
