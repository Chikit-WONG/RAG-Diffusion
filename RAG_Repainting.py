from RAG_pipeline_flux import RAG_FluxPipeline
import argparse
import torch
from PIL import Image
import json

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    return parser.parse_args()

args = parse_arguments()

if args.idx is not None:

    # We provide repainting cases in various scenarios
    json_file = 'data/Repainting_Gallery.json'
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

    Repainting_prompt = item["Repainting_prompt"]
    Repainting_SR_prompt = item["Repainting_SR_prompt"]
    Repainting_HB_prompt = item["Repainting_HB_prompt"]
    Repainting_mask = Image.open(item["Repainting_mask"]).convert("L")
    Repainting_HB_replace = item["Repainting_HB_replace"]
    Repainting_seed = item["Repainting_seed"]
    Repainting_single = item["Repainting_single"]

else:
    prompt = "A vase and an apple."
    HB_replace = 2
    HB_prompt_list =  [
        "Vase",
        "Apple"
    ]
    HB_m_offset_list = [
        0.05,
        0.65
    ]
    HB_n_offset_list = [
        0.1,
        0.25
    ]
    HB_m_scale_list = [
        0.5,
        0.3
    ]
    HB_n_scale_list = [
        0.8,
        0.5
    ]
    SR_delta = 0.5
    SR_hw_split_ratio = "0.6, 0.4"
    SR_prompt = "A beautifully crafted vase, its elegant curves and floral embellishments standing prominently on the left side. Its delicate design echoes a sense of timeless artistry. BREAK On the right, a shiny apple with vibrant red skin, enticing with its perfectly smooth surface and hints of green around the stem."
    height = 1024
    width = 1024
    seed = 1202

    Repainting_prompt = "A vase and a Rubik's Cube."
    Repainting_SR_prompt = "A beautifully crafted vase, its elegant curves and floral embellishments standing prominently on the left side. Its delicate design echoes a sense of timeless artistry. BREAK On the right, a vibrant Rubik's Cube, with its distinct colorful squares, sitting next to the vase, adding a playful and dynamic contrast to the still life composition."
    Repainting_HB_prompt = "Rubik's Cube"
    Repainting_mask = Image.open("data/Repainting_mask/mask_0.png").convert("L") 
    Repainting_HB_replace = 3
    Repainting_seed = 100
    Repainting_single = 0


image, Repainting_image_output = pipe(
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
        Repainting_mask=Repainting_mask,
        Repainting_prompt=Repainting_prompt,
        Repainting_SR_prompt=Repainting_SR_prompt,
        Repainting_HB_prompt=Repainting_HB_prompt,
        Repainting_HB_replace=Repainting_HB_replace,
        Repainting_seed=Repainting_seed,
        Repainting_single=Repainting_single,
        prompt=prompt, 
        height=height, 
        width=width, 
        num_inference_steps=20, 
        guidance_scale=3.5
        )

image.images[0].save("RAG_Original.png")
Repainting_image_output.images[0].save("RAG_Repainting.png")

print(f"Repainting Done.")