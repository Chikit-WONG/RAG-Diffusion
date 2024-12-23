import json
import argparse
from PIL import Image
import torch
from RAG_pipeline_flux_IP_Adapter import RAG_FluxPipeline
from IP_Adapter import IPAdapter, resize_img

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

image_encoder_path = "google/siglip-so400m-patch14-384"
ipadapter_path = "./ip-adapter.bin" # you need to download this file at "https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/tree/main"
ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, help="Loading parameters in json")
    return parser.parse_args()

args = parse_arguments()

if args.idx is not None:

    json_file = 'data/IP_Adapter_Gallery.json'
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
    prompt = "On the left a man with fire and on the right is a strong man with a sword."
    HB_replace = 2
    HB_prompt_list =  [
        "a man with fire",
        "a strong man with a sword"
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
    SR_prompt = "a man with fire BREAK a strong man with a sword"
    height, width = 1024, 1024
    seed = 2438

    id_images_path = ["./data/Skeleton.jpg", "./data/strong.png"]
    id_weights = [0.8, 0.8]

id_images = []
for id_image in id_images_path:
    image = Image.open(id_image).convert("RGB")
    image = resize_img(image)
    id_images.append(image)

images = ip_model.generate(
    prompt=prompt,
    width=width,
    height=height,
    seed=seed,
    num_inference_steps=20,
    guidance_scale=3.5,
    
    pil_image=id_images,
    scale=id_weights,

    SR_delta=SR_delta,
    SR_hw_split_ratio=SR_hw_split_ratio,
    SR_prompt=SR_prompt,
    HB_prompt_list=HB_prompt_list,
    HB_m_offset_list=HB_m_offset_list,
    HB_n_offset_list=HB_n_offset_list,
    HB_m_scale_list=HB_m_scale_list,
    HB_n_scale_list=HB_n_scale_list,
    HB_replace=HB_replace
)

images[0].save(f"./RAG_with_IPAdapter.png")