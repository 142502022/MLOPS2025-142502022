import json
import toml
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import itertools

def load_config(f_path):
    if f_path.endswith('.json'):
        with open(f_path,'r') as f:
            return json.load(f)
    elif f_path.endswith('.toml'):
        return toml.load(f_path)
    else:
        raise ValueError("!!!UnKnown file type!!!")

def get_model(model_name):
    if model_name not in models.__dict__:
        raise ValueError(f"'{model_name}' is not found in torchvision")
    model = models.__dict__[model_name](pretrained=True)
    model.eval()
    return model

def get_image_transforms():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])

def perform_inference(model,img_path,transforms):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def generate_grid_combinations(params):
    keys = params.keys()
    values = params.values()
    combinations = list(itertools.product(*values))
    param_list = []
    for comb in combinations:
        param_list.append(dict(zip(keys,comb)))

    return param_list


if __name__ == "__main__":
    pipeline_config = load_config("pipeline_config.json")
    model_params = load_config("model_params.toml")
    grid_search_config = load_config("grid_search.json")
    data_path = pipeline_config["data_source"]["path"]
    model_architectures = pipeline_config["model_architectures"]
    image_transforms = get_image_transforms()
    grid_combination = generate_grid_combinations(grid_search_config["parameters"])

    for model_name in model_architectures:
        print(f"\n Performing inference with {model_name}")
        model = get_model(model_name)
        params = model_params.get(model_name, {})
        print(f"Model parameters loaded: {params}")

        for filename in os.listdir((data_path)):
            if filename.endswith(pipeline_config["data_source"]["file_format"]):
                img_path = os.path.join(data_path,filename)
                print(f"Inferring on img: '{filename}'")

                output = perform_inference(model,img_path,image_transforms)

                print(f"Output tensor shape:{output.shape}")

    for model_name in model_architectures:
        print(f"Model:- {model_name}")

        for i,comb in enumerate(grid_combination):
            print(f"Testing combination {i+1}:{comb}")
            print("[Simulating training with these parameters]")
