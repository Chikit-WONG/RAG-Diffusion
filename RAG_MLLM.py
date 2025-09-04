import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers  import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def GPT4(prompt,key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open('template.txt', 'r',encoding="utf-8") as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting."
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    # print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    response_txt = response.text
    # print(response_txt)
    obj=response.json()
    # print(obj)
    text=obj['choices'][0]['message']['content']
    # print(text)
    # print()
    return get_params_dict(text)

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: (.*?)(?=\n|\Z)", response)
    if split_ratio_match:
        SR_hw_split_ratio = split_ratio_match.group(1)
        # print("Final split ratio:", final_split_ratio)
    else:
        SR_hw_split_ratio="NULL"
        # print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        SR_prompt = prompt_match.group(1).strip()
        # print("Regional Prompt:", regional_prompt)
    else:
        SR_prompt="NULL"
        # print("Regional Prompt not found.")

    HB_prompt_list_match = re.search(r"HB_prompt_list: (.*?)(?=\n|\Z)", response)
    if HB_prompt_list_match:
        HB_prompt_list = HB_prompt_list_match.group(1).strip()
        # print("sub_prompt_list:", sub_prompt_list)
    else:
        HB_prompt_list="NULL"
        # print("sub_prompt_list not found.")

    HB_m_offset_list_match = re.search(r"HB_m_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_m_offset_list_match:
        HB_m_offset_list = HB_m_offset_list_match.group(1).strip()
        # print("x_offset_list:", x_offset_list)
    else:
        HB_m_offset_list="NULL"
        # print("x_offset_list not found.")
    
    HB_n_offset_list_match = re.search(r"HB_n_offset_list: (.*?)(?=\n|\Z)", response)
    if HB_n_offset_list_match:
        HB_n_offset_list = HB_n_offset_list_match.group(1).strip()
        # print("y_offset_list:", y_offset_list)
    else:
        HB_n_offset_list="NULL"
        # print("y_offset_list not found.")

    HB_m_scale_list_match = re.search(r"HB_m_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_m_scale_list_match:
        HB_m_scale_list = HB_m_scale_list_match.group(1).strip()
        # print("x_scale_list:", x_scale_list)
    else:
        HB_m_scale_list="NULL"
        # print("x_scale_list not found.")

    HB_n_scale_list_match = re.search(r"HB_n_scale_list: (.*?)(?=\n|\Z)", response)
    if HB_n_scale_list_match:
        HB_n_scale_list = HB_n_scale_list_match.group(1).strip()
        # print("y_scale_list:", y_scale_list)
    else:
        HB_n_scale_list="NULL"
        # print("y_scale_list not found.")

    image_region_dict = {'SR_hw_split_ratio': SR_hw_split_ratio, 'SR_prompt': SR_prompt, 'HB_prompt_list': HB_prompt_list, 'HB_m_offset_list': HB_m_offset_list, 'HB_n_offset_list': HB_n_offset_list, 'HB_m_scale_list': HB_m_scale_list, 'HB_n_scale_list': HB_n_scale_list}
    return image_region_dict

# def local_llm(prompt,model_path=None):
#     if model_path==None:
#         model_id = "Llama-2-13b-chat-hf" 
#     else:
#         model_id=model_path
#     print('Using model:',model_id)
#     tokenizer = LlamaTokenizer.from_pretrained(model_id)
#     model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
#     with open('./data/RAG_template.txt', 'r') as f:
#         template=f.readlines()
#     user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
#     textprompt= f"{' '.join(template)} \n {user_textprompt}"
#     model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
#     model.eval()
#     with torch.no_grad():
#         print('waiting for LLM response')
#         res = model.generate(**model_input, max_new_tokens=4096)[0]
#         output=tokenizer.decode(res, skip_special_tokens=True)
#         output = output.replace(textprompt,'')
#     return get_params_dict(output)
#     # return output

# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import torch, json, os

# # def local_llm(prompt, model_path=None):
# #     model_id = model_path or "Llama-2-13b-chat-hf"
# #     print("Using model:", model_id)

# #     # 若固定用 LLaMA，可换回 LlamaTokenizer/LlamaForCausalLM
# #     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_id,
# #         device_map="auto",
# #         torch_dtype=torch.float16,  # 或 "auto"/bfloat16
# #         trust_remote_code=True
# #     ).eval()

# #     with open("./data/RAG_template.txt", "r", encoding="utf-8") as f:
# #         template = " ".join(f.readlines()).strip()

# #     user_text = f"Caption: {prompt}\nOnly return a valid JSON for parameters.\n"
# #     textprompt = f"{template}\n{user_text}"

# #     inputs = tokenizer(textprompt, return_tensors="pt").to(model.device)
# #     max_ctx = getattr(model.config, "max_position_embeddings", 4096)
# #     max_new = max(64, min(1024, max_ctx - inputs["input_ids"].shape[-1] - 16))

# #     with torch.no_grad():
# #         print("waiting for LLM response")
# #         out = model.generate(
# #             **inputs,
# #             max_new_tokens=max_new,
# #             temperature=0.2,
# #             top_p=0.9,
# #             repetition_penalty=1.05,
# #             eos_token_id=tokenizer.eos_token_id,
# #             pad_token_id=tokenizer.eos_token_id,
# #         )
# #         seq = out[0]
# #         gen_ids = seq[inputs["input_ids"].shape[-1]:]
# #         text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# #     # 建议在 get_params_dict 之前先尝试 json 解析
# #     try:
# #         data = json.loads(text)
# #     except Exception:
# #         data = get_params_dict(text)  # 与现有逻辑兼容

# #     return data

# # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# # from qwen_vl_utils import process_vision_info
# # import torch
# # import json

# # def local_llm(prompt, model_path=None, image_path=None):
# #     model_id = model_path or "Qwen2.5-VL-7B-Instruct"  # 使用 Qwen2.5-VL 模型路径，默认路径为 Qwen2.5-VL-7B-Instruct
# #     print("Using model:", model_id)

# #     # 加载 Qwen2.5-VL 模型和处理器
# #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #         model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
# #     ).eval()

# #     processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# #     # 处理用户输入的文本
# #     text = f"Caption: {prompt}\nOnly return a valid JSON for parameters.\n"

# #     # 如果有图像输入，处理图像
# #     if image_path:
# #         image_inputs, video_inputs = process_vision_info([{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}])
# #     else:
# #         image_inputs, video_inputs = None, None

# #     # 将文本和图像输入转换为模型的输入格式
# #     inputs = processor(
# #         text=[text],
# #         images=image_inputs,
# #         videos=video_inputs,
# #         padding=True,
# #         return_tensors="pt",
# #     )

# #     inputs = inputs.to("cuda")

# #     # 推理：生成输出
# #     with torch.no_grad():
# #         print("Waiting for LLM response...")
# #         generated_ids = model.generate(**inputs, max_new_tokens=128)

# #     # 提取生成的文本（去除输入的部分）
# #     generated_ids_trimmed = [
# #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# #     ]
# #     output_text = processor.batch_decode(
# #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# #     )
# #     print("Generated text:", output_text)

# #     # 将生成的文本尝试解析为 JSON，若失败，则返回原始输出
# #     try:
# #         data = json.loads(output_text[0])  # 假设是一个 JSON 格式的字符串
# #     except Exception:
# #         data = output_text[0]  # 如果无法解析为 JSON，直接返回生成的文本
        
# #     # print(type(data))
# #     # print()
# #     # print(data)

# #     return get_params_dict(data)


# from transformers  import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

def local_llm_cpu(prompt,  model_path=None):
    
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu"
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # print(processor)  # 检查是否包含 video/image processor 等模块
    
    with open('./data/RAG_template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {user_textprompt}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textprompt},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(output_text)
    
    return get_params_dict(output_text[0])

def local_llm(prompt,  model_path=None):
    
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # print(processor)  # 检查是否包含 video/image processor 等模块
    
    with open('./data/RAG_template.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {user_textprompt}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textprompt},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # 输入长度
    input_len = inputs["input_ids"].shape[-1]

    # 模型的最大上下文窗口
    max_ctx = getattr(model.config, "max_position_embeddings", 4096)  # Qwen2.5-VL 7B 是 4k

    # 计算还能生成的 token 数
    max_new = max_ctx - input_len

    # 避免负数或太小
    max_new = max(1, max_new)


    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(output_text)
    
    return get_params_dict(output_text[0])


# import requests
# import json
# import os
# from transformers import AutoTokenizer
# import transformers
# import re
# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer


# def GPT4(prompt,key):
#     url = "https://api.openai.com/v1/chat/completions"
#     api_key = key
#     with open('template.txt', 'r',encoding="utf-8") as f:
#         template=f.readlines()
#     user_textprompt=f"Caption:{prompt} \n Let's think step by step, please reply in plain text and do not use any bold or bullet-point Markdown formatting."
    
#     textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
#     payload = json.dumps({
#     "model": "gpt-4o",
#     "messages": [
#         {
#             "role": "user",
#             "content": textprompt
#         }
#     ]
#     })
#     headers = {
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {api_key}',
#     'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
#     'Content-Type': 'application/json'
#     }
#     # print('waiting for GPT-4 response')
#     response = requests.request("POST", url, headers=headers, data=payload)
#     response_txt = response.text
#     # print(response_txt)
#     obj=response.json()
#     # print(obj)
#     text=obj['choices'][0]['message']['content']
#     # print(text)
#     # print()
#     return get_params_dict(text)

# def get_params_dict(output_text):
#     response = output_text
#     # Find Final split ratio
#     split_ratio_match = re.search(r"Final split ratio: (.*?)(?=\n|\Z)", response)
#     if split_ratio_match:
#         SR_hw_split_ratio = split_ratio_match.group(1)
#         # print("Final split ratio:", final_split_ratio)
#     else:
#         SR_hw_split_ratio="NULL"
#         # print("Final split ratio not found.")
#     # Find Regioanl Prompt
#     prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
#     if prompt_match:
#         SR_prompt = prompt_match.group(1).strip()
#         # print("Regional Prompt:", regional_prompt)
#     else:
#         SR_prompt="NULL"
#         # print("Regional Prompt not found.")

#     HB_prompt_list_match = re.search(r"HB_prompt_list: (.*?)(?=\n|\Z)", response)
#     if HB_prompt_list_match:
#         HB_prompt_list = HB_prompt_list_match.group(1).strip()
#         # print("sub_prompt_list:", sub_prompt_list)
#     else:
#         HB_prompt_list="NULL"
#         # print("sub_prompt_list not found.")

#     HB_m_offset_list_match = re.search(r"HB_m_offset_list: (.*?)(?=\n|\Z)", response)
#     if HB_m_offset_list_match:
#         HB_m_offset_list = HB_m_offset_list_match.group(1).strip()
#         # print("x_offset_list:", x_offset_list)
#     else:
#         HB_m_offset_list="NULL"
#         # print("x_offset_list not found.")
    
#     HB_n_offset_list_match = re.search(r"HB_n_offset_list: (.*?)(?=\n|\Z)", response)
#     if HB_n_offset_list_match:
#         HB_n_offset_list = HB_n_offset_list_match.group(1).strip()
#         # print("y_offset_list:", y_offset_list)
#     else:
#         HB_n_offset_list="NULL"
#         # print("y_offset_list not found.")

#     HB_m_scale_list_match = re.search(r"HB_m_scale_list: (.*?)(?=\n|\Z)", response)
#     if HB_m_scale_list_match:
#         HB_m_scale_list = HB_m_scale_list_match.group(1).strip()
#         # print("x_scale_list:", x_scale_list)
#     else:
#         HB_m_scale_list="NULL"
#         # print("x_scale_list not found.")

#     HB_n_scale_list_match = re.search(r"HB_n_scale_list: (.*?)(?=\n|\Z)", response)
#     if HB_n_scale_list_match:
#         HB_n_scale_list = HB_n_scale_list_match.group(1).strip()
#         # print("y_scale_list:", y_scale_list)
#     else:
#         HB_n_scale_list="NULL"
#         # print("y_scale_list not found.")

#     image_region_dict = {'SR_hw_split_ratio': SR_hw_split_ratio, 'SR_prompt': SR_prompt, 'HB_prompt_list': HB_prompt_list, 'HB_m_offset_list': HB_m_offset_list, 'HB_n_offset_list': HB_n_offset_list, 'HB_m_scale_list': HB_m_scale_list, 'HB_n_scale_list': HB_n_scale_list}
#     return image_region_dict

# def local_llm(prompt, model_path=None):
#     if model_path is None:
#         model_id = "Llama-2-13b-chat-hf" 
#     else:
#         model_id = model_path
    
#     print('Using model:', model_id)
    
#     try:
#         # Try to load with safetensors format first
#         tokenizer = LlamaTokenizer.from_pretrained(model_id)
#         model = LlamaForCausalLM.from_pretrained(
#             model_id, 
#             load_in_8bit=False, 
#             device_map='auto', 
#             torch_dtype=torch.float16,
#             use_safetensors=True  # Force use of safetensors
#         )
#     except Exception as e1:
#         print(f"Failed to load with safetensors: {e1}")
#         try:
#             # Fallback: try without specifying safetensors
#             print("Trying alternative loading method...")
#             tokenizer = LlamaTokenizer.from_pretrained(model_id)
#             model = LlamaForCausalLM.from_pretrained(
#                 model_id, 
#                 load_in_8bit=False, 
#                 device_map='auto', 
#                 torch_dtype=torch.float16,
#                 trust_remote_code=True  # In case remote code is needed
#             )
#         except Exception as e2:
#             print(f"Alternative loading also failed: {e2}")
#             # Try with different approach
#             print("Trying with auto tokenizer and model...")
#             try:
#                 tokenizer = AutoTokenizer.from_pretrained(model_id)
#                 model = LlamaForCausalLM.from_pretrained(
#                     model_id,
#                     device_map='auto',
#                     torch_dtype=torch.float16,
#                     low_cpu_mem_usage=True
#                 )
#             except Exception as e3:
#                 print(f"All loading methods failed. Last error: {e3}")
#                 raise e3
    
#     # Load template
#     template_path = './data/RAG_template.txt'
#     if not os.path.exists(template_path):
#         # Fallback to current directory if data folder doesn't exist
#         template_path = 'RAG_template.txt'
    
#     try:
#         with open(template_path, 'r') as f:
#             template = f.readlines()
#     except FileNotFoundError:
#         print(f"Template file not found at {template_path}. Using empty template.")
#         template = []
    
#     user_textprompt = f"Caption:{prompt} \n Let's think step by step:"
#     textprompt = f"{' '.join(template)} \n {user_textprompt}"
    
#     model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
#     model.eval()
    
#     with torch.no_grad():
#         print('waiting for LLM response')
#         res = model.generate(**model_input, max_new_tokens=4096)[0]
#         output = tokenizer.decode(res, skip_special_tokens=True)
#         output = output.replace(textprompt, '')
    
#     return get_params_dict(output)