"""
Gradio app for VAREdit image editing model.
Provides web interface for editing images with text instructions.
"""
import gradio as gr
import os
import tempfile
from PIL import Image
import logging

from infer import load_model, generate_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAREditApp:
    def __init__(self):
        self.model_components = None
        self.current_model_size = None
        self.current_image_size = None
        
    def load_model_if_needed(self, model_size: str, image_size: int):
        """Load model if not already loaded or if configuration changed."""
        if (self.model_components is None or 
            self.current_model_size != model_size or 
            self.current_image_size != image_size):
            
            logger.info(f"Loading {model_size} model for {image_size}px images...")
            
            # Default paths - you may need to adjust these
            pretrain_root = "HiDream-ai/VAREdit"
            if model_size == "2B":
                assert image_size == 512, "2B model only supports 512px images"
                model_path = "HiDream-ai/VAREdit/2B-512.pth"  # Update this path
            elif model_size == "8B":
                if image_size == 512:
                    model_path = "HiDream-ai/VAREdit/8B-512.pth"
                elif image_size == 1024:
                    model_path = "HiDream-ai/VAREdit/8B-1024.pth"
                else:
                    raise ValueError(f"Unsupported image size: {image_size}, expected 512 or 1024")
            
            self.model_components = load_model(pretrain_root, model_path, model_size, image_size)
            self.current_model_size = model_size
            self.current_image_size = image_size
            logger.info("Model loaded successfully")
    
    def edit_image(
        self, 
        input_image: Image.Image, 
        instruction: str,
        image_size: int = 1024,
        cfg: float = 4.0,
        tau: float = 0.5,
        seed: int = -1
    ) -> Image.Image:
        """Edit image based on text instruction."""
        if input_image is None:
            raise gr.Error("Please upload an image")
        
        if not instruction.strip():
            raise gr.Error("Please provide an editing instruction")
        
        try:
            # Load model if needed
            self.load_model_if_needed(model_size = "8B", image_size = image_size)
            
            # Save input image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                input_image.save(tmp_file.name, 'JPEG')
                temp_path = tmp_file.name
            
            try:
                # Generate edited image
                result_image = generate_image(
                    self.model_components,
                    temp_path,
                    instruction,
                    cfg=cfg,
                    tau=tau,
                    seed=seed if seed != -1 else None
                )
                
                return result_image
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise gr.Error(f"Failed to edit image: {str(e)}")

# Initialize app
app = VAREditApp()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="VAREdit Image Editor") as demo:
        gr.Markdown("# VAREdit Image Editor")
        gr.Markdown("Edit images using natural language instructions with the VAREdit model.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=400
                )
                
                instruction = gr.Textbox(
                    label="Editing Instruction",
                    placeholder="e.g., 'Remove glasses from this person', 'Change the sky to sunset', 'Add a hat'",
                    lines=2
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    image_size = gr.Radio(
                        choices=[512, 1024],
                        value=1024,
                        label="Image Size"
                    )
                    
                    cfg = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="CFG Scale (Guidance Strength)"
                    )
                    
                    tau = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.1,
                        step=0.01,
                        label="Temperature (Tau)"
                    )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)",
                        precision=0
                    )
                
                edit_btn = gr.Button("Edit Image", variant="primary", size="lg")
                
            with gr.Column():
                output_image = gr.Image(
                    label="Edited Image",
                    height=400
                )
        
        # Example images and instructions
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["assets/test_3.jpg", "change shirt to a black-and-white striped Breton top, add a red beret, set the background to an artist's loft with a window view of the Eiffel Tower"],
                ["assets/test.jpg", "Add glasses to this girl and change hair color to red"],
                ["assets/test_1.jpg", "replace all the bullets with shimmering, multi-colored butterflies."],
                ["assets/test_4.jpg", "Set the scene against a dark, blurred-out server room, make all text and arrows glow with a vibrant cyan light"],
            ],
            inputs=[input_image, instruction],
            outputs=output_image,
            fn=lambda img, inst: app.edit_image(img, inst),
            cache_examples=False
        )
        
        # Set up event handler
        edit_btn.click(
            fn=app.edit_image,
            inputs=[input_image, instruction, image_size, cfg, tau, seed],
            outputs=output_image
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )