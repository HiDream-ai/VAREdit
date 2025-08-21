"""
Image inference module for VAREdit model.
Supports 2B and 8B model variants for image editing with text instructions.
"""
import argparse
import logging
from typing import Tuple, Any, Optional
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
import PIL.Image as PImage
from tools.run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer,
    gen_one_img, h_div_w_templates, dynamic_resolution_h_w
)
import time
import torch

def transform(pil_img, target_image_size):
    # currently only support square image.
    width, height = pil_img.size
    max_dim = max(width, height)
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(pil_img, (0, 0))
    def crop_op(image):
        image = image.resize((max_dim, max_dim), resample=PImage.LANCZOS)
        crop_image = image.crop((0, 0, width, height))
        return crop_image
    padded_image = padded_image.resize((target_image_size, target_image_size), resample=PImage.LANCZOS)
    im = to_tensor(np.array(padded_image))
    return im.add(im).add_(-1), crop_op

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    '2B': {
        'vae_filename': 'infinity_vae_d32reg.pth',
        'vae_type': 32,
        'model_type': 'infinity_2b',
        'apply_spatial_patchify': 0,
    },
    '8B': {
        'vae_filename': 'infinity_vae_d56_f8_14_patchify.pth',
        'vae_type': 14,
        'model_type': 'infinity_8b',
        'apply_spatial_patchify': 1,
    }
}

# Common model arguments
COMMON_ARGS = {
    'cfg_insertion_layer': 0,
    'add_lvl_embeding_only_first_block': 1,
    'use_bit_label': 1,
    'rope2d_each_sa_layer': 1,
    'rope2d_normalized_by_hw': 2,
    'use_scale_schedule_embedding': 0,
    'sampling_per_bits': 1,
    'text_channels': 2048,
    'h_div_w_template': 1.000,
    'use_flex_attn': 0,
    'cache_dir': '/dev/shm',
    'checkpoint_type': 'torch',
    'bf16': 1,
    'enable_model_cache': 0,
}


def load_model(pretrain_root: str, model_path: str, model_size: str, image_size: int) -> Tuple[Any, ...]:
    """
    Load the model and its components.
    
    Args:
        pretrain_root: Root directory for pretrained models
        model_path: Path to the specific model checkpoint
        
    Returns:
        Tuple of (args, model, vae, tokenizer, text_encoder)
        
    Raises:
        ValueError: If unsupported model size is specified
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size}. Choose '2B' or '8B'.")
    
    config = MODEL_CONFIGS[model_size]
    
    # Build arguments
    args_dict = {
        **COMMON_ARGS,
        **config,
        'model_path': model_path,
        'vae_path': f"{pretrain_root}/{config['vae_filename']}",
        'text_encoder_ckpt': f"{pretrain_root}/flan-t5-xl"
    }
    args = argparse.Namespace(**args_dict)
    if image_size == 512:
        args.pn = "0.25M"
    elif image_size == 1024:
        args.pn = "1M"
    else:
        raise ValueError(f"Unsupported image size: {image_size}. Choose 512 or 1024.")
    logger.info(f"Loading {model_size} model from {model_path}")
    
    # Load components
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args)
    
    logger.info("Model loaded successfully")
    return args, model, vae, text_tokenizer, text_encoder, image_size


def generate_image(
    model_components: Tuple[Any, ...],
    src_img_path: str,
    instruction: str,
    cfg: float = 4.0,
    tau: float = 0.5,
    seed: Optional[int] = -1,
) -> None:
    """
    Generate edited image based on source image and text instruction.
    
    Args:
        model_components: Tuple of (args, model, vae, tokenizer, text_encoder)
        src_img_path: Path to source image
        instruction: Text instruction for editing
        cfg: Classifier-free guidance scale
        tau: Temperature parameter
    """
    args, model, vae, tokenizer, text_encoder, image_size = model_components
    
    # Set default image size
    assert image_size in [512, 1024], f"Invalid image size: {image_size}, expected 512 or 1024"
    if image_size == 512:
        pn = "0.25M"
    elif image_size == 1024:
        pn = "1M"
    
    # Load and preprocess source image
    try:
        with Image.open(src_img_path) as src_img:
            src_img = src_img.convert('RGB')
            src_img_tensor, crop_op = transform(src_img, image_size)
    except Exception as e:
        logger.error(f"Failed to load source image: {e}")
        raise
    
    # Set up generation parameters
    aspect_ratio = 1.0  # h:w ratio
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - aspect_ratio))]
    scale_schedule = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[h_div_w_template][pn]['scales']]
    
    logger.info(f"Generating image with instruction: '{instruction}'")
    
    # Generate image
    if seed == -1:
        seed = np.random.randint(0, 1000000)
    torch.cuda.empty_cache()
    start_time = time.time()
    generated_image = gen_one_img(
        model, vae, tokenizer, text_encoder,
        instruction, src_img_tensor,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=0,
        apply_spatial_patchify=args.apply_spatial_patchify,
    )
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f"Max memory: {max_memory:.2f} GB")
    generated_image_np = generated_image.cpu().numpy()
    if generated_image_np.shape[2] == 3:
        generated_image_np = generated_image_np[..., ::-1]
    result_image = Image.fromarray(generated_image_np.astype(np.uint8))
    result_image = crop_op(result_image)
    return result_image

def main():
    """Main execution function with example usage."""
    try:
        # Load model
        model_components = load_model(
            "HiDream-ai/VAREdit",
            "HiDream-ai/VAREdit/8B-1024.pth",
            "8B",
            1024
        )
        
        # Generate image
        generate_image(
            model_components,
            "assets/test.jpg",
            "Add glasses to this girl and change hair color to red",
            cfg=3.0,
            tau=1.0,
            seed=42
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()