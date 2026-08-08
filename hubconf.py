from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder


def _create_model(pretrained=False):
    return HARSCnnEncoder(
        dim=125,
        input_channel=6,
        inner_conv_output_dim=128 * 10
    )


MODEL_NAME = "harsccencoder"

globals()[MODEL_NAME] = _create_model
