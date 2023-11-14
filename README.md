# Multi2Human
This is the code of Multi2Human: Controllable Human Image Generation with Multimodal Controls(Coming soon！)
This repository provides the official PyTorch implementation for the following paper:
**Multi2Human: Controllable Human Image Generation with Multimodal Controls**</br>

![front_page_sample](assets/sampleteaser.png)

### Abstract
>   *Generating high-quality and diverse human images presents a substantial difficulty within the field of computer vision, especially in developing controllable generative models that can utilize input from various modalities. Such models could enable innovative applications like digital human, fashion design, and content creation. In this study, we introduce Multi2Human, a two-stage image synthesis framework for controllable human image generation with multimodal controls. In the first stage, a novel WAvelet-vqVaE (WAVE) architecture is designed to embed human images using a learnable codebook. The WAVE model enhances the conventional Vector Quantized Variational Autoencoder (VQVAE) by integrating wavelets throughout the encoder, thereby enhancing the quality of image reconstruction and synthesis. In the second stage, a new Multimodal Conditioned Diffusion Model (MCDM) is de- signed to estimate the underlying prior distribution within the discrete latent space using a discrete diffusion process, thus allowing for human image generation condi- tioned on multimodal controls. Quantitative and qualitative analysis demonstrates that the proposed method has the ability to create high-quality, lifelike full-body human images while satisfying the specified multimodal controls.*

![front_page_sample](assets/framework.png)

## Setup

**Clone this repo:**
```bash
git clone https://github.com/gxl-groups/Multi2Human.git
cd Multi2Human
```

**Dependencies:**

```bash
conda create --name multi2Human --file requirements.yml
conda activate multi2Human
```


