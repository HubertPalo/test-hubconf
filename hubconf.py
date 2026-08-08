from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder


_AVAILABLE_MODELS = {
    "harsccencoder": HARSCnnEncoder,
    "HARSCnnEncoder": HARSCnnEncoder,
}


def __getattr__(name):
    if name not in _AVAILABLE_MODELS:
        raise AttributeError(
            f"Modelo '{name}' no encontrado. "
            f"Modelos disponibles: {list(_AVAILABLE_MODELS.keys())}"
        )

    def model(pretrained=False):
        return _AVAILABLE_MODELS[name](
            dim=125,
            input_channel=6,
            inner_conv_output_dim=128 * 10
        )

    return model
