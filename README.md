
# GAN's for fun

# 😎 Documentation 😎
Here I present a Generative Adverarial Network, which can generate numbers from noise.

A GAN, is a fight between two networks to win, A Generator and A Discriminator. A generator tries to fool the model by trying to generate images which looks exactly same as original image, whereas a Discriminator tries to calculate the loss, between the image GENERATED image and the ORIGINAL one. Our model WINS only if the Generator succeeds in FOOLING the Discriminator and the loss of discrimator is reduced, which will mean that the discriminator cannot distinguish between an original and a generated image.

Let's understand in brief, how GAN's Work

#⚡ The Generator ⚡
It takes in random noise as input and with the help of Transpose Convolutions it learn to generate an image from noise. I tried to generate 3 channel RGB images but the model could not converge the losses, so i trained on 1 channel grayscale images. I made a 4 (3 hidden + 1 output) layer model, and the last layer of the model has a different activation function than the rest of the layers. preferably tanh().

# 🕵️ The Discriminator 🕵️
The Discriminator is a detective model that takes an imput of original image and compares against the generated image from the generator, basically it stacks a few conv layers to extract the information and calculate the losses from it. We use Leaky ReLU activation function and not use any AF on the last layer.

The Loss function used in network is BCEwithLogitLoss, which makes sense because it applies sigmoid AF automatically while calculating the loss.

Use this link to learn to improve gans more!

# 😄Thank's for reading😄
Checkout [www.sankhyikii.com](https://www.sankhyikii.com/portfolio) for more projects realted to data science 
