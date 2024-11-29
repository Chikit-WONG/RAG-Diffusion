from RAG_pipeline_flux import RAG_FluxPipeline
from diffusers import FluxPriorReduxPipeline
from diffusers.utils import load_image
import torch
import argparse
import json

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    return parser.parse_args()

args = parse_arguments()

if args.idx is not None:

    json_file = 'data/Redux_Gallery.json'
    with open(json_file, 'r') as f:
        data = json.load(f)  

    item = data[args.idx]

    prompt = item["prompt"]
    HB_replace = item["HB_replace"]
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
    Redux_list = [pipe_prior_redux(load_image(Redux)) for Redux in item["Redux_list"]]
    del pipe_prior_redux
    torch.cuda.empty_cache()

else:
    prompt = "A man is holding a sign that says RAG-Diffusion, and another man is holding a sign that says flux-redux."
    HB_replace = 8
    HB_m_offset_list = [
            0.05,
            0.55
        ]
    HB_n_offset_list = [
            0.2,
            0.2
        ]
    HB_m_scale_list = [
            0.4,
            0.4
        ]
    HB_n_scale_list = [
            0.4,
            0.4
        ]
    SR_delta = 0.2
    SR_hw_split_ratio = "0.5,0.5"
    SR_prompt = "A man is holding a sign that says RAG-Diffusion BREAK another man is holding a sign that says flux-redux."
    height = 1024
    width = 1024
    seed = 2272
    Redux_list = [
            "data/Redux/Lecun.jpg",
            "data/Redux/Hinton.jpg"
        ]
    Redux_list = [pipe_prior_redux(load_image(Redux)) for Redux in Redux_list]
    del pipe_prior_redux
    torch.cuda.empty_cache()


image = pipe(
    SR_delta = SR_delta,
    SR_hw_split_ratio = SR_hw_split_ratio,
    SR_prompt = SR_prompt,
    HB_m_offset_list = HB_m_offset_list,
    HB_n_offset_list = HB_n_offset_list,
    HB_m_scale_list = HB_m_scale_list,
    HB_n_scale_list = HB_n_scale_list,
    Redux_list = Redux_list,
    HB_replace = HB_replace,
    seed = seed,
    prompt = prompt, height=height, width=width, num_inference_steps=20, guidance_scale=3.5
    )
image.images[0].save("RAG_with_Redux.png")
