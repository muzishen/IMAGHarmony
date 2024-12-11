import os
from typing import List
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, style_ckpt=None,
                 num_tokens=4, target_blocks=["down_blocks.2.attentions.1"], inference=False):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.style_ckpt = style_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks
        print(inference)
        self.inference = inference

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter(inference=inference)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()
        if self.style_ckpt is not None:
            self.load_style_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, inference=False):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    if 'up_blocks.0.attentions.1' in name:
                        attn_procs[name] = IPAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            scale=1.0,
                            num_tokens=self.num_tokens,
                            inference=inference,
                            up=True
                        ).to(self.device, dtype=torch.float16)
                    else:
                        attn_procs[name] = IPAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            scale=1.0,
                            lay_out_scale=1.0,
                            num_tokens=self.num_tokens,
                            inference=inference
                        ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        if self.inference:
            print('load ipa params!!!')
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    def load_style_adapter(self):
        if os.path.splitext(self.style_ckpt)[-1] == ".safetensors":
            state_dict = {"style_adapter": {}}
            with safe_open(self.style_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # if 'to_k_s' in key or 'to_v_s' in key or 'to_k_ip' in key or 'to_v_ip' in key:
                    if 'to_k_s' in key or 'to_v_s' in key:
                        state_dict["style_adapter"][key.replace("style_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.style_ckpt, map_location="cpu")
        style_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        style_layers.load_state_dict(state_dict["style_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None,extra_embeds=None,number_class_crossattention=None,uncond_extra_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:
            
            clip_image_embeds = clip_image_embeds - content_prompt_embeds
        
        
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        
        if extra_embeds is not None:
            number_class_crossattention=number_class_crossattention.to(self.device,torch.float16)
            extra_embeds=extra_embeds.to(self.device,torch.float16)
            output=number_class_crossattention(extra_embeds,clip_image_embeds)
            image_prompt_embeds=torch.cat([image_prompt_embeds,image_prompt_embeds+output],dim=1)
        
        
        
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        uncond_image_prompt_embeds=torch.cat([uncond_image_prompt_embeds,uncond_image_prompt_embeds],dim=1)
        # uncond_clip_image_embeds=torch.zeros_like(clip_image_embeds)
        # if uncond_extra_embeds is not None:
        #     uncond_extra_embeds=uncond_extra_embeds.to(self.device,torch.float16)
        #     uncond_output=number_class_crossattention(uncond_extra_embeds,uncond_clip_image_embeds)
        #     uncond_image_prompt_embeds=torch.cat([uncond_image_prompt_embeds,uncond_image_prompt_embeds+uncond_output],dim=1)
            
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            neg_content_emb=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        #extra text
        


        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            neg_content_emb=None,
            neg_content_prompt=None,
            neg_content_scale=1.0,
            inference=True,
            extra_text=None,
            number_class_crossattention=None,
            ** kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_,  # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_,  # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb
        else:
            pooled_prompt_embeds_ = None
        extra_prompt_embeds=None
        
        #extra_text
        # if extra_text is not None:
        #     with torch.inference_mode():
        #         text_extra_input_ids_1 = self.pipe.tokenizer(
        #         extra_text,
        #         max_length=self.pipe.tokenizer.model_max_length,
        #         padding="max_length",
        #         truncation=True,
        #         return_tensors="pt"
        #     ).input_ids
        #         text_extra_input_ids_2 = self.pipe.tokenizer_2(
        #         extra_text,
        #         max_length=self.pipe.tokenizer_2.model_max_length,
        #         padding="max_length",
        #         truncation=True,
        #         return_tensors="pt"
        #     ).input_ids
                
        #         encoder_extra_output = self.pipe.text_encoder(text_extra_input_ids_1.to(self.device),
        #                                           output_hidden_states=True)
        #         text_extra_embeds = encoder_extra_output.hidden_states[-2]
        #         print(text_extra_embeds)
        #         encoder_extra_output_2=self.pipe.text_encoder_2(text_extra_input_ids_2.to(self.device),
        #                                               output_hidden_states=True)
        #         text_extra_embeds_2 = encoder_extra_output_2.hidden_states[-2].to(self.device)
        #         text_extra_embeds=torch.concat([text_extra_embeds,text_extra_embeds_2],dim=-1)
        
        if extra_text is not None:
            with torch.inference_mode():
                (
                extra_prompt_embeds,
                extra_negative_prompt_embeds,
                extra_pooled_prompt_embeds,
                extra_negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                extra_text,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )     
                
                
       
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image,
                                                                                content_prompt_embeds=pooled_prompt_embeds_,extra_embeds=extra_prompt_embeds,number_class_crossattention=number_class_crossattention,uncond_extra_embeds=extra_negative_prompt_embeds)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # print(negative_prompt_embeds.shape)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        
        
        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
