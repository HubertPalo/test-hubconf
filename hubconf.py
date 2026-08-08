from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder
import yaml

def _create_model(model_name: str, link: str):
    print(f"Creating model: {model_name}, Link: {link}")
    return "asd"
# def _create_model(pretrained=False):
#     return HARSCnnEncoder(
#         dim=125,
#         input_channel=6,
#         inner_conv_output_dim=128 * 10
#     )

available_models = yaml.safe_load(open("links.yaml", "r"))
for model_name, link in available_models.items():
    globals()[model_name] = _create_model

# MODEL_NAME = "harsccencoder"

# globals()[MODEL_NAME] = _create_model
