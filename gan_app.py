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
st.markdown("<h2 style='text-align: center; color: #aa00ff;font-size:40%'>Refresh To Get New Image generated from the model</h2>",unsafe_allow_html=True)

gen = Generator(64)

gen.load_state_dict(torch.load('model_scripted.pth', map_location='cpu'))
gen.eval()
noise = get_noise(128, 64, 'cpu')
fake2 = fakes(gen, noise)
fake2 = fake2.cpu().detach().numpy()
st.image(fake2,caption='Generated Image',use_column_width='always')

st.image("MNIST_DCGAN_Progression.png",caption='Results when model was trained', use_column_width='auto')

st.markdown("<a href='https://github.com/shyamgupta196/GAN-s-are-fun'>Check out the project</a>",unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #ffffff;font-size:160%'> 😎 Documentation 😎 </h1>",
    unsafe_allow_html=True,
)

st.write("Here I present a Generative Adverarial Network, which can generate numbers from noise.\n \n A GAN, is a fight between two networks to win, A Generator and A Discriminator. A generator tries to fool the model by trying to generate images which looks exactly same as original image, whereas a Discriminator tries to calculate the loss, between the image GENERATED image and the ORIGINAL one. Our model WINS only if the Generator succeeds in FOOLING the Discriminator and the loss of discrimator is reduced, which will mean that the discriminator cannot distinguish between an original and a generated image. \n\n Let's understand in brief, how GAN's Work")

st.markdown(
    "<h1 style='text-align: center; color: #00000;font-size:160%'> ⚡ The Generator ⚡ </h1>",
    unsafe_allow_html=True,
)

st.write('''It takes in random noise as input and with the help of Transpose Convolutions it learn to generate an image from noise. I tried to generate 3 channel RGB images but the model could not
         converge the losses, so i trained on 1 channel grayscale images. I made a 4 (3 hidden + 1 output) layer model, and the last layer of the model has a different activation function than the rest of the layers.
         preferably **tanh()**. 
         ''')


st.markdown(
    "<h1 style='text-align: center; color: #00000;font-size:160%'> 🕵️ The Discriminator 🕵️ </h1>",
    unsafe_allow_html=True,
)

st.write('''
The Discriminator is a detective model that takes an imput of original image and compares against the generated image from the generator, basically it stacks a few conv layers
 to extract the information and calculate the losses from it.  We use Leaky ReLU activation function and not use any AF on the last layer.\n\n The Loss function used in network is BCEwithLogitLoss, which makes sense because it applies sigmoid AF automatically while calculating the loss.''')


st.markdown("<a href='https://github.com/soumith/ganhacks'> Use this link to learn to improve gans more! </a>",unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center; color: #00000;font-size:160%'> 😄Thank's for reading😄 </h1>",
    unsafe_allow_html=True,
)
