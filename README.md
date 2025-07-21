# LoRA Model Testing Script

This repository contains a script to test a fine-tuned LoRA model for multimodal (vision-language) tasks.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have the LoRA model files in the `lora_model` directory:
   - adapter_config.json
   - adapter_model.safetensors
   - special_tokens_map.json
   - tokenizer.json
   - tokenizer_config.json

## Using the Script

1. Update the `image_path` variable in `test.py` to point to your own image file:
   ```python
   image_path = "your_image.jpg"  # Change this to your image path
   ```

2. Run the script:
   ```
   python test.py
   ```

3. Customize the instruction in the script as needed for different types of image analysis:
   ```python
   instruction = "You are an expert radiographer. Describe accurately what you see in this image."
   ```

## Hardware Requirements

- For optimal performance, a CUDA-compatible GPU is recommended
- If no GPU is available, the script will fall back to CPU mode (much slower)

## Troubleshooting

- If you encounter CUDA out-of-memory errors, you may need to adjust the model loading parameters
- If you have issues with the model loading, ensure the base model is accessible 