"""
AUTHOR -  @shyamgupta196 
# CHECK README FILE TO KNOW MORE!!!! 
#this executes faster with help of cacher
transition-image example [made it in columns]done!
edges available [this will be added from the image segmentation model file]done!
"""

import streamlit as st
from gan import Generator, fakes, get_noise
import numpy as np
import torch

# from torchvision.utils import make_grid

st.markdown(
    """
    <style>
    .reportview-container {
        background: black
    }
   .sidebar .sidebar-content {
        background: gray
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center; color: #aa00ff;font-size:160%'>Get your GAN number's Image </h1>",
    unsafe_allow_html=True,
)

st.image("MNIST_DCGAN_Progression.png")
gen = Generator(64)

gen.load_state_dict(torch.load('model_scripted.pth', map_location='cpu'))
gen.eval()
noise = get_noise(128, 64, 'cpu')
fake2 = fakes(gen, noise)
fake2 = fake2.cpu().detach().numpy()
st.image(fake2, 'Here is your generated image')
