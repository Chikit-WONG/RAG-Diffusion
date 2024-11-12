import torch
import argparse
import json
from RAG_pipeline_flux import RAG_FluxPipeline

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    return parser.parse_args()

args = parse_arguments()
if args.idx is not None:
    
    json_file = './data/RAG_Gallery.json'
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


else:
    prompt = "a balloon on the bottom of a dog"
    HB_replace = 2
    HB_prompt_list =  [
            "Balloon",
            "Dog"
        ]
    HB_m_offset_list =  [
            0.1,
            0.1
        ]
    HB_n_offset_list =  [
            0.55,
            0.05
        ]
    HB_m_scale_list =  [
            0.8,
            0.8
        ]
    HB_n_scale_list = [
            0.4,
            0.45
        ]
    SR_delta = 1.0
    SR_hw_split_ratio = "0.5; 0.5"
    SR_prompt = "A playful dog, perhaps a golden retriever, with its ears perked up, sitting on the balloon, giving an enthusiastic demeanor. BREAK A colorful balloon floating gently, its string dangling gracefully, just beneath the dog."
    height, width = 1024, 1024
    seed = 1234


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
).images[0]

filename = "RAG.png"
image.save(filename)
print(f"Image saved as {filename}")


