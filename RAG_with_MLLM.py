import torch
from RAG_pipeline_flux import RAG_FluxPipeline
from RAG_MLLM import local_llm_cpu, local_llm_gpu, GPT4

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "A small elephant on the left and a huge rabbit on the right."

# para_dict = GPT4(prompt,key='')
para_dict = local_llm_gpu(prompt, model_path='../Qwen2.5-VL/models/Qwen2.5-VL-7B-Instruct')
print('-------------------------------------')
print('Model out put:')
print(type(para_dict))
print()
print(para_dict)
print('-------------------------------------')

HB_replace = 2
HB_prompt_list =  para_dict["HB_prompt_list"]
HB_m_offset_list = eval(para_dict["HB_m_offset_list"])
HB_n_offset_list = eval(para_dict["HB_n_offset_list"])
HB_m_scale_list = eval(para_dict["HB_m_scale_list"])
HB_n_scale_list = eval(para_dict["HB_n_scale_list"])
SR_delta = 1.0
SR_hw_split_ratio = para_dict["SR_hw_split_ratio"]
SR_prompt = para_dict["SR_prompt"]
height = 1024
width = 1024
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

from datetime import datetime

# 获取时间戳（例如：2025-08-28_16-55-30）
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 拼接文件名
filename = f"./results/RAG_{timestamp}.png"

# 保存图片
image.save(filename)
print(f"Image saved as {filename}")