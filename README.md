<div align="center">

<h1> Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement </h1>

[**Zhennan Chen**](#)<sup>1*</sup> · [**Yajie Li**](#)<sup>1*</sup> · [**Haofan Wang**](https://haofanwang.github.io/)<sup>2,3</sup> · [**Zhibo Chen**](#)<sup>3</sup> · [**Zhengkai Jiang**](https://jiangzhengkai.github.io/)<sup>4</sup> · [**Jun Li**](https://sites.google.com/view/junlineu/)<sup>1</sup> · [**Qian Wang**](#)<sup>5</sup> · [**Jian Yang**](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup> ·[**Ying Tai**](https://tyshiwo.github.io/)<sup>1✉</sup>

<sup>1</sup>Nanjing University · <sup>2</sup>InstantX Team · <sup>3</sup>Liblib AI · <sup>4</sup>HKUST · <sup>5</sup>China Mobile

<a href='https://arxiv.org/abs/2411.06558'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
</div>

<table class="center">
  <tr>
    <td width=100% style="border: none"><img src="assets/pictures/teaser.jpg" style="width:100%"></td>
  </tr>
</table>

We present **RAG**, a **R**egional-**A**ware text-to-image **G**eneration method conditioned on regional descriptions for precise layout composition. Regional prompting, or compositional generation, which enables fine-grained spatial control, has gained increasing attention for its practicality in real-world applications. However, previous methods either introduce additional trainable modules, thus only applicable to specific models, or manipulate on score maps within cross-attention layers using attention masks, resulting in limited control strength when the number of regions increases. To handle these limitations, we decouple the multi-region generation into two sub-tasks, the construction of individual region (**Regional Hard Binding**) that ensures the regional prompt is properly executed, and the overall detail refinement (**Regional Soft Refinement**) over regions that dismiss the visual boundaries and enhance adjacent interactions. Furthermore, RAG novelly makes repainting feasible, where users can modify specific unsatisfied regions in the last generation while keeping all other regions unchanged, without relying on additional inpainting models. Our approach is tuning-free and applicable to other frameworks as an enhancement to the prompt following property. Quantitative and qualitative experiments demonstrate that RAG achieves superior performance over attribute binding and object relationship than previous tuning-free methods. 

## News ##
- **2024.11.12**: 🚀 Our code and technical report are released.


## Text-to-Image Generation
### 1. Set Environment
```bash
conda create -n RAG python==3.9
conda activate RAG
pip install xformers==0.0.28.post1 diffusers peft torchvision==0.19.1 opencv-python==4.10.0.84 sentencepiece==0.2.0 protobuf==5.28.1
```
### 2. Quick Start
```python
import torch
from RAG_pipeline_flux import RAG_FluxPipeline

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


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
```
- `HB_replace` (`int`):
  The times of hard binding. More times can make the position control more precise, but may lead to obvious boundaries.
- `HB_prompt_list` (`List[str]`):
  Fundamental descriptions for each individual region or object.
- `HB_m_offset_list`, `HB_n_offset_list`, `HB_m_scale_list`, `HB_n_scale_list`(`List[float]`):
  Corresponding to the coordinates of each fundamental prompt in HB_prompt_list.
- `SR_delta` (`float`):
  The fusion strength of image latent and regional-aware local latent. This is a flexible parameter, you can try 0.25, 0.5, 0.75, 1.0.
- `SR_prompt` (`str`):
  Highly descriptive sub-prompts for each individual region or object. Each sub-prompt is separated by *BREAK*.
- `SR_hw_split_ratio` (`str`):
  The global region divisions correspond to each highly descriptive sub-prompt in SR_prompt.

<details open>
<summary>The following shows several schematic diagrams of `HB_m_offset_list`, `HB_n_offset_list`, `HB_m_scale_list`, `HB_n_scale_list`, `SR_hw_split_ratio`.</summary> 
<table class="center">
  <tr>
    <td width=25% style="border: none"><img src="assets/pictures/region1.jpg" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/pictures/region2.jpg" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/pictures/region3.jpg" style="width:100%"></td>

  <!-- <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">HB_m_offset_list=[<font color="rgb(0,176,240)">1/9</font>, *, *, *, *], HB_n_offset_list=[<font color="rgb(0,176,240)">1/12</font>, *, *, *, *], HB_m_scale_list=[<font color="rgb(0,176,240)">2/4</font>, *, *, *, *], HB_n_scale_list[<font color="rgb(0,176,240)">1/8</font>, *, *, *, *], 
    SR_hw_split_ratio=[<font color="rgb(84,130,53)">1/3</font>, <font color="rgb(127,96,0)">2/4</font>, <font color="rgb(127,96,0)">1/4</font>, <font color="rgb(127,96,0)">1/4</font>; <font color="rgb(84,130,53)">2/3</font>, <font color="rgb(217,83,153)">2/3</font>, <font color="rgb(217,83,153)">1/3</font>] </td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">SR_hw_split_ratio=[<font color="rgb(127,96,0)">1/6</font>, <font color="rgb(127,96,0)">2/6</font>, <font color="rgb(127,96,0)">3/6</font>]</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">SR_hw_split_ratio=[<font color="rgb(127,96,0)">1/2</font>, <font color="rgb(127,96,0)">1/2</font>]</td>

  </tr> -->
</table>
</details>

### 3. RAG with MLLM
```python
import torch
from RAG_pipeline_flux import RAG_FluxPipeline
from RAG_MLLM import local_llm, GPT4

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "A small elephant on the left and a huge rabbit on the right."

para_dict = GPT4(prompt,key='')
print(para_dict)

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

filename = "RAG.png"
image.save(filename)
print(f"Image saved as {filename}")
```


## Gallery

### 1. Image Inference

<details open>
<summary>Examples</summary> 
<table class="center">
  <tr>
    <td width=100% style="border: none"><img src="assets/pictures/celebrity.png" style="width:100%"></td>
  </tr>
  <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">On the left, Einstein is painting the Mona Lisa; in the center, Elon Reeve Musk is participating in the U.S. presidential election; on the right, Trump is hosting a Tesla product launch.</td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=0</pre>
    </td>
    </td>
  </td>
  <tr>
    <td width=100% style="border: none"><img src="assets/pictures/animal.png" style="width:100%"></td>
  </tr>
  <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">On the left, a penguin wearing sunglasses is sunbathing in the desert; in the center, a tiger wearing a scarf is standing on a glacier; on the right, a panda in a windbreaker is walking through the forest.</td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=1</pre>
    </td>
    </td>
  </tr>
</table>
<table class="center">
  <tr>
    <td width=25% style="border: none"><img src="assets/pictures/cup.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/pictures/dog_balloon.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/pictures/apple_glass.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="assets/pictures/tree.png" style="width:100%"></td>
  </tr>
  <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">Seven ceramic mugs in different colors are placed on a wooden table, with numbers from 1 to 7 written on the cups, and a bunch of white roses on the left.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A balloon on the bottom of a dog.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A cylindrical glass, obscuring the right half of the apple behind it.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word"> From left to right, Pink blossoming trees, Green sycamore trees, Golden maples and Snow-blanketed pines.</td>
  </tr>
  <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=2</pre>
    </td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=3</pre>
    </td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=4</pre>
    </td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG.py --idx=5</pre>
    </td>
  </tr>
</table>
</details>
   

### 2. Image Repainting
<details open>
<summary>Example 1</summary>
<table class="center">
    <tr>
        <td style="border: none"><img src="assets/pictures/shirt.png"></td>
        <td style="border: none"><img src="assets/pictures/shirt_noise.png"></td>
        <td style="border: none"><img src="assets/pictures/shirt_repainting.png"></td>
    </tr>
    <tr>
        <td width="25%" style="border: none; text-align: center; word-wrap: break-word" colspan="3">Text prompt: "A brown curly hair African girl in blue shirt printed with a bird."

Repainting prompt: "A brown curly hair African girl in pink shirt printed with a bird."</td>
    </tr>
</table>
</details>

<details open>
<summary>Example 2</summary>
<table class="center">
    <tr>
        <td style="border: none"><img src="assets/pictures/anime.png"></td>
        <td style="border: none"><img src="assets/pictures/anime_noise.png"></td>
        <td style="border: none"><img src="assets/pictures/anime_repainting.png"></td>
    </tr>
    <tr>
        <td width="25%" style="border: none; text-align: center; word-wrap: break-word" colspan="3">Text prompt: "A man on the left, a woman on the right."

Repainting prompt: "A man on the left, an anime woman on the right."</td>
    </tr>
</table>
</details>

### 3. RAG With LoRA
```python
import torch
from RAG_pipeline_flux import RAG_FluxPipeline

pipe = RAG_FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

# 8steps
# pipe.load_lora_weights('ByteDance/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors')
# pipe.fuse_lora(lora_scale=0.125)

# MiaoKa-Yarn-World
# pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-MiaoKa-Yarn-World', weight_name='FLUX-dev-lora-MiaoKa-Yarn-World.safetensors')
# pipe.fuse_lora(lora_scale=1.0)

# Black-Myth-Wukong
pipe.load_lora_weights('Shakker-Labs/FLUX.1-dev-LoRA-collections', weight_name='FLUX-dev-lora-Black_Myth_Wukong_hyperrealism_v1.safetensors')
pipe.fuse_lora(lora_scale=0.7)

pipe = pipe.to("cuda")

prompt = "A mountain on the left, a crouching man in the middle, and an ancient architecture on the right."
HB_replace = 3
HB_prompt_list = [
        "Mountain",
        "Crouching man",
        "Ancient architecture"
    ]
HB_m_offset_list = [
        0.02,
        0.35,
        0.68
    ]
HB_n_offset_list = [
        0.1,
        0.1,
        0.0
    ]
HB_m_scale_list = [
        0.29,
        0.3,
        0.29
    ]
HB_n_scale_list = [
        0.8,
        0.8,
        1.0
    ]
SR_delta = 0.0
SR_hw_split_ratio = "0.33, 0.34, 0.33"
SR_prompt = "A mountain towering on the left, its peaks reaching into the sky, the steep slopes inviting exploration and wonder. BREAK In the middle, a crouching man is focused, his posture suggesting thoughtfulness or a momentary pause in action. BREAK On the right, an ancient architecture, its stone walls and archways revealing stories of the past, stands firmly, offering a glimpse into historical grandeur."
height = 1024
width = 1024
seed = 1236

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
    num_inference_steps=20,
    guidance_scale=3.5,
).images[0]

filename = "RAG_with_LoRA.png"
image.save(filename)
```

<details open>
<summary><a href="https://huggingface.co/ByteDance/Hyper-SD" target="_blank">Hyper-Flux</a></summary> 
<table class="center">
    <tr style="line-height: 0">
        <td style="border: none; text-align: center">8-steps LoRA</td>
        <td style="border: none; text-align: center">Flux</td>
        <td style="border: none; text-align: center">8-steps LoRA</td>
        <td style="border: none; text-align: center">Flux</td>
    </tr>
    <tr>
        <td width="25%" style="border: none"><img src="assets/pictures/LoRA_5_8.png" style="width:100%"></td>
        <td width="25%" style="border: none"><img src="assets/pictures/LoRA_7_20.png" style="width:100%"></td>
        <td width="25%" style="border: none"><img src="assets/pictures/LoRA_6_8.png" style="width:100%"></td>
        <td width="25%" style="border: none"><img src="assets/pictures/LoRA_8_20.png" style="width:100%"></td>
    </tr>
    <tr>
        <td width="25%" style="border: none; text-align: center; word-wrap: break-word" colspan="2">From left to right: a red cake, an orange cake, a yellow cake, and a green cake.</td>
        <td width="25%" style="border: none; text-align: center; word-wrap: break-word" colspan="2">The spring, summer, autumn, and winter of Hokkaido.</td>
    </tr>
    <tr>
      <td width="50%" style="border: none; text-align: center; word-wrap: break-word" colspan="2">
        <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=8steps --idx=0</pre>
      </td>
      <td width="50%" style="border: none; text-align: center; word-wrap: break-word" colspan="2">
        <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=8steps --idx=1</pre>
      </td>
    </tr>
</table>
</details> 

<details open>
<summary><a href="https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-collections" target="_blank">FLUX.1-dev-LoRA-collections</a></summary> 
<table class="center">
  <tr>
    <td width=50% style="border: none"><img src="assets/pictures/LoRA_1.png" style="width:100%"></td>
    <td width=50% style="border: none"><img src="assets/pictures/LoRA_2.png" style="width:100%"></td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">A man on the left is holding a bag and a man on the right is holding a book.</td>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">A mountain on the left, a crouching man in the middle, and an ancient architecture on the right.</td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=Black-Myth-Wukong --idx=2</pre>
    </td>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=Black-Myth-Wukong --idx=3</pre>
    </td>
  </tr>
</table>
</details>

<details open>
<summary><a href="https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-MiaoKa-Yarn-World" target="_blank">FLUX.1-dev-LoRA-MiaoKa-Yarn-World</a></summary> 
<table class="center">
  <tr>
    <td width=50% style="border: none"><img src="assets/pictures/LoRA_3.png" style="width:100%"></td>
    <td width=50% style="border: none"><img src="assets/pictures/LoRA_4.png" style="width:100%"></td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">A two-tier cabinet: the top shelf has two pears made of wool, and the bottom shelf has three apples made of wool.</td>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">On the left is a forest made of wool, and on the right is a volcano made of wool.</td>
  </tr>
  <tr>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=MiaoKa-Yarn-World --idx=4</pre>
    </td>
    <td width="50%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python RAG_with_LoRA.py --lora=MiaoKa-Yarn-World --idx=5</pre>
    </td>
  </tr>
</table>
</details>


# 👏 Acknowledgment

 Our work is sponsored by [HuggingFace](https://huggingface.co) and [fal.ai](https://fal.ai), and it built on [diffusers](https://github.com/huggingface/diffusers), [Flux.1-dev](https://github.com/black-forest-labs/flux), [RPG](https://github.com/YangLing0818/RPG-DiffusionMaster?tab=readme-ov-file).



# 📖BibTeX
```
@article{chen2024region,
  title={Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement},
  author={Chen, Zhennan and Li, Yajie and Wang, Haofan and Chen, Zhibo and Jiang, Zhengkai and Li, Jun and Wang, Qian and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2411.06558},
  year={2024}
}
```
