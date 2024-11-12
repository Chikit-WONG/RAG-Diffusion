import json
import torch
from RAG_pipeline_flux import RAG_FluxPipeline
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', type=str, required=True)
    parser.add_argument('--idx', type=int, required=True)
    return parser.parse_args()

args = parse_arguments()
json_file = './data/LoRA_Gallery.json'
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

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

if args.lora == "8steps":
    pipe.load_lora_weights('ByteDance/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors')
    pipe.fuse_lora(lora_scale=0.125)
    num_inference_steps = 8
elif args.lora == "MiaoKa-Yarn-World":
    pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-MiaoKa-Yarn-World', weight_name='FLUX-dev-lora-MiaoKa-Yarn-World.safetensors')
    pipe.fuse_lora(lora_scale=1.0)
    num_inference_steps = 20
elif args.lora == "Black-Myth-Wukong":
    pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-collections', weight_name='FLUX-dev-lora-Black_Myth_Wukong_hyperrealism_v1.safetensors')
    pipe.fuse_lora(lora_scale=0.7)
    num_inference_steps = 20

pipe = pipe.to("cuda")

image = pipe(
    prompt=prompt,
    HB_replace=HB_replace,
    HB_prompt_list=HB_prompt_list,
    HB_m_offset_list=HB_m_offset_list,
    HB_n_offset_list=HB_n_offset_list,
    HB_m_scale_list=HB_m_scale_list,
    HB_n_scale_list=HB_n_scale_list,
    SR_delta=SR_delta,
    SR_hw_split_ratio=SR_hw_split_ratio,
    SR_prompt=SR_prompt,
    seed=seed,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=3.5,
).images[0]

filename = f"LoRA_{args.idx}.png"
image.save(filename)