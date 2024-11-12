import math
import torch
import torchvision.transforms.functional as F
TOKENS = 75
    
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn" in name and "transformer_blocks" in name  and "single_transformer_blocks" not in name and module.__class__.__name__ == "Attention":
            module.forward = FluxTransformerBlock_hook_forward(self, module)           
        elif "attn" in name and "single_transformer_blocks" in name and module.__class__.__name__ == "Attention":
            module.forward = FluxSingleTransformerBlock_hook_forward(self, module) 

def FluxSingleTransformerBlock_hook_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None, SR_encoder_hidden_states_list=None, SR_norm_encoder_hidden_states_list=None, SR_hidden_states_list=None, SR_norm_hidden_states_list=None):
        flux_hidden_states=module.processor(module, hidden_states=hidden_states, image_rotary_emb=image_rotary_emb)

        height = self.h 
        width = self.w
        x_t = hidden_states.size()[1]-512
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w

        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)
        contexts_list = SR_norm_hidden_states_list

        def single_matsepcalc(x, contexts_list, image_rotary_emb):
            h_states = []
            x_t = x.size()[1]-512
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            latent_out = latent_w
            latent_in = latent_h
            i = 0
            sumout = 0
            SR_all_out_list=[]

            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    context = contexts_list[i]
                    i = i + 1 + dcell.breaks
                    SR_all_out = module.processor(module, hidden_states=context, image_rotary_emb=image_rotary_emb)
                    out = SR_all_out[:, 512 :, ...]
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2])
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)

                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out
                    out = out[:, int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end), :]

                    v_states.append(out)
                    SR_all_out_list.append(SR_all_out)

                output_x = torch.cat(v_states,dim = 2) 
                h_states.append(output_x)

            output_x = torch.cat(h_states,dim = 1) 
            output_x = output_x.reshape(x.size()[0], x.size()[1]-512, x.size()[2]) 
            new_SR_all_out_list = []

            for SR_all_out in SR_all_out_list:
                SR_all_out[:, 512 :, ...] = output_x
                new_SR_all_out_list.append(SR_all_out)
            x[:, 512 :, ...] = output_x * self.SR_delta + x[:, 512 :, ...] * (1-self.SR_delta)

            return x, new_SR_all_out_list
        
        return single_matsepcalc(flux_hidden_states, contexts_list, image_rotary_emb)
    
    return forward

def FluxTransformerBlock_hook_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None, SR_encoder_hidden_states_list=None, SR_norm_encoder_hidden_states_list=None, SR_hidden_states_list=None, SR_norm_hidden_states_list=None):
        flux_hidden_states, flux_encoder_hidden_states = module.processor(module, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, image_rotary_emb=image_rotary_emb)
        
        height = self.h 
        width = self.w
        x_t = hidden_states.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w

        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)

        contexts_list = SR_norm_encoder_hidden_states_list

        def matsepcalc(x, contexts_list, image_rotary_emb):
            h_states = []
            x_t = x.size()[1]
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            latent_out = latent_w
            latent_in = latent_h
            i = 0
            sumout = 0
            SR_context_attn_output_list = []

            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    context = contexts_list[i]
                    i = i + 1 + dcell.breaks
                    out,SR_context_attn_output = module.processor(module, hidden_states=x, encoder_hidden_states=context, image_rotary_emb=image_rotary_emb)
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2]) 
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)

                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out

                    out = out[:, int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end), :]
                    v_states.append(out)
                    SR_context_attn_output_list.append(SR_context_attn_output)

                output_x = torch.cat(v_states,dim = 2) 
                h_states.append(output_x)

            output_x = torch.cat(h_states,dim = 1) 
            output_x = output_x.reshape(x.size()[0],x.size()[1],x.size()[2]) 

            return output_x * self.SR_delta + flux_hidden_states * (1-self.SR_delta), flux_encoder_hidden_states, SR_context_attn_output_list

        return matsepcalc(hidden_states, contexts_list, image_rotary_emb)

    return forward

def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x


def init_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn" in name and "transformer_blocks" in name  and "single_transformer_blocks" not in name and module.__class__.__name__ == "Attention":
            module.forward = FluxTransformerBlock_init_forward(self, module)           
        elif "attn" in name and "single_transformer_blocks" in name and module.__class__.__name__ == "Attention":
            module.forward = FluxSingleTransformerBlock_init_forward(self, module) 

def FluxSingleTransformerBlock_init_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None,RPG_encoder_hidden_states_list=None,RPG_norm_encoder_hidden_states_list=None,RPG_hidden_states_list=None,RPG_norm_hidden_states_list=None):
        return module.processor(module, hidden_states=hidden_states, image_rotary_emb=image_rotary_emb)
    return forward

def FluxTransformerBlock_init_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None,RPG_encoder_hidden_states_list=None,RPG_norm_encoder_hidden_states_list=None,RPG_hidden_states_list=None,RPG_norm_hidden_states_list=None):
        return module.processor(module, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, image_rotary_emb=image_rotary_emb)
    return forward