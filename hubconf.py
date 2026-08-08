from minerva.models.nets.lfr_har_architectures import HARSCnnEncoder
import yaml

def _get_model(model_name: str, link: str):
    print(f"Creating model: {model_name}, Link: {link}")
    return "asd"

def _get_function_that_creates_custom_model(model_name, link):
    def custom_function():
        return _get_model(model_name, link)
    return custom_function

available_models = yaml.safe_load(open("links.yaml", "r"))
for model_name, link in available_models.items():
    globals()[model_name] = _get_function_that_creates_custom_model(model_name, link)
