import torch
import argparse
import json
from RAG_pipeline_flux_PuLID import RAG_FluxPipeline

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_pulid_models()
pipe.load_pretrain()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    return parser.parse_args()

args = parse_arguments()

if args.idx is not None:
    
    json_file = 'data/PuLID_Gallery.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    item = data[args.idx]


    prompt = item["prompt"]
    HB_replace = item["HB_replace"]
    HB_prompt_list =  item["HB_prompt_list"]
    HB_m_offset_list = item["HB_m_offset_list"]
    HB_n_offset_list = item["HB_n_offset_list"]
    HB_m_scale_list = item["HB_m_scale_list"]
    HB_n_scale_list = item["HB_n_scale_list"]
    SR_delta = item["SR_delta"]
    SR_hw_split_ratio = item["SR_hw_split_ratio"]
    SR_prompt = item["SR_prompt"]
    height = item["height"]
    width = item["width"]
    seed = item["seed"]

    id_images_path = item["id_images_path"] 
    id_weights = item["id_weights"]


else:
    prompt = "A man is holding a sign that says RAG-Diffusion, and another man is holding a sign that says PuLID."
    HB_replace = 2
    HB_prompt_list =  [
        "A man is holding a sign that says RAG-Diffusion",
        "another man is holding a sign that says PuLID."
    ]
    HB_m_offset_list =  [
            0.05,
            0.55
        ]
    HB_n_offset_list =  [
            0.2,
            0.2
        ]
    HB_m_scale_list =  [
            0.4,
            0.4
        ]
    HB_n_scale_list = [
            0.6,
            0.6
        ]
    SR_delta = 0.1
    SR_hw_split_ratio = "0.5, 0.5"
    SR_prompt = "A man is holding a sign that says RAG-Diffusion BREAK another man is holding a sign that says PuLID."
    height, width = 1024, 1024
    seed = 2272

    id_images_path = ["./data/Skeleton.jpg", "./data/Lecun.jpg"]
    id_weights = [1.0, 1.0]


image = pipe(
    SR_delta=SR_delta,
    SR_hw_split_ratio=SR_hw_split_ratio,
    SR_prompt=SR_prompt,
    HB_prompt_list=HB_prompt_list,
    HB_m_offset_list=HB_m_offset_list,
    HB_n_offset_list=HB_n_offset_list,
    HB_m_scale_list=HB_m_scale_list,
    HB_n_scale_list=HB_n_scale_list,
    HB_replace=HB_replace,
    seed=seed,
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=20,
    guidance_scale=3.5,
    id_images_path=id_images_path,
    id_weights=id_weights
).images[0]

filename = "./RAG_with_PuLID.png"
image.save(filename)
print(f"Image saved as {filename}")
