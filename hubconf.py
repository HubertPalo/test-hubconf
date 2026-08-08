from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder
from minerva.models.nets.time_series.cnns import CNN_PF_Backbone
from minerva.models.nets.time_series.resnet import _ResNet1D, ResNetSEBlock
from minerva.models.nets.time_series.imu_transformer import _IMUTransformerEncoder
from minerva.models.nets.tnc import RnnEncoder, TSEncoder
from pathlib import Path
import yaml
import torch


def _get_harscnnencoder_backbone(dim=2304):
    model = HARSCnnEncoder(dim=dim, input_channel=6, inner_conv_output_dim=128*10)
    return model

def _get_cnnpff_backbone():
    model = CNN_PF_Backbone(include_middle=True, flatten=True)
    return model

def _get_resnetse5_backbone():
    model = _ResNet1D(input_shape=[6, 60], avg_pooling=True, residual_block_cls=ResNetSEBlock)
    return model

def _get_transformer_backbone():
    model = _IMUTransformerEncoder(
        input_shape=[6, 60],
        transformer_dim=64,
        encode_position=True,
        nhead=8,
        dim_feedforward=128,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        num_encoder_layers=6
    )
    return model

def _get_rnn_backbone():
    model = RnnEncoder(
        hidden_size=100,
        in_channel=6,
        encoding_size=320,
        cell_type='GRU',
        num_layers=1,
        dropout=0,
        bidirectional=True,
        permute=True
    )
    return model

def _get_ts2vec_backbone():
    model = TSEncoder(
        input_dims=6,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        permute=True
    )
    return model

def _get_backbone(model_name: str):
    models = {
        "harscnnencoder": _get_harscnnencoder_backbone,
        "cnnpff": _get_cnnpff_backbone,
        "resnetse5": _get_resnetse5_backbone,
        "transformer": _get_transformer_backbone,
        "rnn": _get_rnn_backbone,
        "ts2vec": _get_ts2vec_backbone,

    }
    return models.get(model_name)()

def _get_model(model_name: str, url: str):
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
    print(f"Creating model: {model_name}, Link: {url}, weights: {weights}")
    return weights

def _get_function_that_creates_custom_model(model_name, link):
    def custom_function():
        return _get_model(model_name, link)
    return custom_function



HUBCONF_DIR = Path(__file__).resolve().parent
LINKS_FILE = HUBCONF_DIR / "links.yaml"

available_models = yaml.safe_load(open(LINKS_FILE, "r"))
for model_name, link in available_models.items():
    globals()[model_name] = _get_function_that_creates_custom_model(model_name, link)