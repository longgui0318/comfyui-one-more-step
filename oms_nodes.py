import os
import copy
import torch
import folder_paths

import comfy.model_patcher
import comfy.ldm.models.autoencoder
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.sampler_helpers
from .unet_2d_condition_woct import UNet2DConditionWoCTModel

from diffusers import StableDiffusionPipeline,UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from comfy import model_management

  
class LoadMoreStepModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                 "model_base": (["sd15_21","sdxl"], {"default": "sd15_21","display": "select"}),
                 "oms_unet": (folder_paths.get_filename_list("unet"), ),
                 }
                }
    RETURN_TYPES = ("OMS_MODEL",)
    RETURN_NAMES = ("OMS_MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "loaders"

    def load_unet(self, model_base, oms_unet):
        unet_path = folder_paths.get_full_path("unet", oms_unet)
        workspace_path = os.path.join(os.path.dirname(__file__))
        if model_base == "sd15_21":
            unet_type = "oms_15_21"
        elif model_base == "sdxl":
            unet_type = "oms_xl"
        else:
            raise ValueError("Unsupported base models.") 
        config_dict = UNet2DConditionWoCTModel.load_config(os.path.join(workspace_path, f"{unet_type}/config.json"))
        unet = UNet2DConditionWoCTModel.from_config(config_dict)
        state_dict = comfy.utils.load_torch_file(unet_path,device=torch.device("cpu"))
        unet.load_state_dict(state_dict)
        return (unet,)


class CalculateMoreStepLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"source_model": ("MODEL",),
                 "oms_model":("OMS_MODEL",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                 "oms_positive": ("CONDITIONING",),
                 "oms_negative": ("CONDITIONING",),
                 "latents": ("LATENT", ),
                 }
                }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)

    FUNCTION = "calculate_more"

    CATEGORY = "latent"

    def calculate_more(self,source_model, oms_model,seed,steps,cfg,sampler_name,scheduler,oms_positive,oms_negative, latents,):
        latent_image = latents["samples"]
        if torch.count_nonzero(latent_image) > 0: #如何传入潜变量不为空，进行标准输入缩放
            latent_image = source_model.inner_model.process_latent_in(latent_image)
        else:#若传入潜变量为空，则说明需要初始化一个空高斯分布的噪点图
            batch_inds = latents["batch_index"] if "batch_index" in latents else None
            latent_image = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        positive_cond = oms_positive[0][0]
        negative_cond = oms_negative[0][0]
        oms_emb = torch.cat([positive_cond,negative_cond],dim=0)
        latent_input_oms = torch.cat([latent_image] * 2)
        v_pred_oms = oms_model(latent_input_oms,oms_emb)['sample']
        
        
        sigmas = self._calculate_sigmas(steps,source_model.model.model_sampling,scheduler,sampler_name)
        sigmas = sigmas.to(source_model.load_device)
                ## Perform OMS
        # alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        # alpha_prod_t_prev =  alphas_cumprod[int(timesteps[0].item())] 
        
        
        # latents = self.oms_step(v_pred_oms, latents, cfg, generator, alpha_prod_t_prev)
        return (latents,)
    
    def _calculate_sigmas(self,steps,model_sampling,scheduler,sampler_name):
        sigmas = None

        discard_penultimate_sigma = False
        if sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = comfy.samplers.calculate_sigmas(model_sampling,scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
        

    
    def oms_step(self, predict_v, latents, oms_guidance_scale, generator, alpha_prod_t_prev):
        pred_uncond, pred_text = predict_v.chunk(2)
        predict_v = pred_uncond + oms_guidance_scale * (pred_text - pred_uncond)
        # so fking dirty but keep it for now
        alpha_prod_t = torch.zeros_like(alpha_prod_t_prev)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * predict_v
        # pred_original_sample = - predict_v
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        pred_prev_sample = pred_prev_sample
        # TODO unit variance but seem dont need it

        device = latents.device
        variance_noise = randn_tensor(
            latents.shape, generator=generator, device=device, dtype=latents.dtype
        )
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t 
        variance = torch.clamp(variance, min=1e-20) * variance_noise

        latents = pred_prev_sample + variance
        return latents



NODE_CLASS_MAPPINGS = {
    "Load More Step Model": LoadMoreStepModel,
    "Calculate More Step Latent": CalculateMoreStepLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load More Step Model": "Load More Step Model",
    "Calculate More Step Latent": "Calculate More Step Latent",
}
