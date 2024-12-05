pip install diffusers
pip install transformers
pip install accelerate
conda install pillow
python -c "import torch, diffusers, transformers; print('Environment setup successful!')"


# Create and activate environment
conda create --name stable-diffusion python=3.8
conda activate stable-diffusion

# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install additional dependencies
pip install diffusers transformers accelerate
conda install pillow

# Verify installation
python -c "import torch, diffusers, transformers; print('Environment setup successful!')"


proxychains pip install diffusers transformers tqdm pillow numpy

 pip install diffusers
 proxychains conda create -n pytorch python=3.10
 proxychains pip3 install torch torchvision torchaudio