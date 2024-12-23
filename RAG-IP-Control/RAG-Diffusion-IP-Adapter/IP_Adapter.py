from attention_processor_IP_Adapter import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
from typing import List
import numpy as np
import torch


def resize_img(input_image, max_side=1280, min_side=1024, size=None, pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)

    return input_image

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)
        
        return image_proj_model
    
    def set_ip_adapter(self):
        transformer = self.pipe.transformer
        ip_attn_procs = {} # 19+38=57
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = transformer.attn_processors[name]
    
        transformer.set_attn_processor(ip_attn_procs)
    
    def load_ip_adapter(self):
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_images=None, clip_image_embeds=None):
        image_embeds = None
        if pil_images is not None:
            image_embeds = []
            for pil_image in pil_images:
                if isinstance(pil_image, Image.Image):
                    pil_image = [pil_image]
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)

                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
                image_embeds.append(image_prompt_embeds)
        
        return image_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, IPAFluxAttnProcessor2_0):
                attn_processor.scale_list = scale

    def get_mask(self, HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, height, width):
        HB_m_offset_list = [int(HB_m_offset * width // 16) for HB_m_offset in HB_m_offset_list]
        HB_n_offset_list = [int(HB_n_offset * height // 16) for HB_n_offset in HB_n_offset_list]
        HB_m_scale_list = [int(HB_m_scale * width // 16) for HB_m_scale in HB_m_scale_list]
        HB_n_scale_list = [int(HB_n_scale * height // 16) for HB_n_scale in HB_n_scale_list]
        height, width = height // 16, width // 16

        masks = []
        for HB_m_offset, HB_n_offset, HB_m_scale, HB_n_scale in zip(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list):
            mask = torch.zeros([height, width], dtype=torch.bool)
            mask[HB_n_offset:HB_n_offset+HB_n_scale, HB_m_offset:HB_m_offset+HB_m_scale] = True
            mask = mask[None, :, :, None].view(1, -1, 1).to(self.device)
            masks.append(mask)

        return masks
                
    
    def generate(
        self,
        SR_delta: float,
        SR_hw_split_ratio: str,
        SR_prompt: str,
        HB_m_offset_list: List[float],
        HB_n_offset_list: List[float],
        HB_m_scale_list: List[float],
        HB_n_scale_list: List[float],
        HB_replace: int,
        HB_prompt_list: List[str]=None,

        width=1024,
        height=1024,
        num_inference_steps=20,
        guidance_scale=3.5,

        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        scale=None,
        seed=None,
        **kwargs,
    ):
        self.set_scale(scale)

        image_emb_list = self.get_image_embeds(pil_images=pil_image, clip_image_embeds=clip_image_embeds)
        image_mask_list = self.get_mask(HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list, height, width)
        
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        images = self.pipe(
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
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            image_emb_list=image_emb_list,
            image_mask_list=image_mask_list,
            **kwargs,
        ).images

        return images

