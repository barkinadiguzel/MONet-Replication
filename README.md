# 🧩 MONet-Replication – Unsupervised Scene Decomposition

This repository is a **clean PyTorch reimplementation** of  
**MONet: Unsupervised Scene Decomposition and Representation (Burgess et al., 2019)**.

The goal is to turn the original paper’s **architecture, math, and block diagram** into a readable and modular codebase.

- Recursive **attention-based object discovery** 🪞  
- **Component-wise VAEs** for object modeling 🧬  
- Full **ELBO objective** for generative training 🧠  

**Paper reference:** [Unsupervised Scene Decomposition and Representation](https://arxiv.org/abs/1901.11390) 📄

---

## 🌠 Overview – How MONet Works

MONet decomposes a scene into objects **one by one** using recursive attention.  
Each object is modeled with its own VAE and the final image is composed from all parts.
```text
Input Image x (B, 3, H, W)
        ⬇️
CNN Encoder (feature maps)
        ⬇️
Attention Net αψ(x, scope)
        ⬇️
Recurrent Attention
  - Generates masks m_k
  - Updates scope
        ⬇️
Component-wise VAE (one per mask m_k)
  - Encoder: qφ(z_k | x, m_k)
  - Decoder: pθ(x | z_k)
        ⬇️
Mask Decoder pθ(c | {z_k})
  - Predicts masks from latent slots
        ⬇️
Compositor
  - Soft-masked summation: x̂ = Σ_k m_k * x_k
        ⬇️
Output:
  - x̂       ← Reconstructed image
  - masks   ← Attention masks
  - z_slots ← Latent vectors
  - mus, logvars ← Latent stats

```
---

## 🧮 Core Math

### Recursive Attention
```math
m_k = s_k · σ(α_ψ(x, s_k))  
s_{k+1} = s_k · (1 − m_k)
```

### Component-wise VAE
```math
q(z_k | x, m_k) = N(μ_k, σ_k²)  
p(z_k) = N(0, I)
```

### Scene Reconstruction
```math
x̂ = Σ_k m_k · x_k
```

### ELBO Objective
```math
L = reconstruction + β · KL(z) + γ · KL(masks)
```

---

## 🧠 What This Model Does

- Decomposes scenes into **K object slots**  
- Learns **unsupervised object masks**  
- Trains with a **full probabilistic generative model**  
- Produces object-level latent representations  

This is MONet exactly as described in the paper — just turned into PyTorch.

---

## 📦 Repository Structure

```bash
MONet-Replication/
├── src/
│   ├── encoder/
│   │   ├── cnn_encoder.py         # Image → feature map
│   │   └── mask_encoder.py        # (Image, Mask) → latent posterior params
│   │
│   ├── attention/
│   │   ├── attention_net.py       # αψ(x, scope) → attention logits
│   │   ├── scope_update.py        # Recursive scope logic
│   │   ├── mask_generator.py      # mk generation step-by-step
│   │   └── recurrent_attention.py # Full MONet attention loop
│   │
│   ├── vae/
│   │   ├── encoder.py             # qφ(z_k | x, m_k)
│   │   ├── decoder.py             # pθ(x | z_k)
│   │   ├── mask_decoder.py        # pθ(c | {z_k})
│   │   └── component_vae.py       # One masked VAE forward
│   │
│   ├── decoder/
│   │   └── compositor.py          # Σ_k m_k * x_k
│   │
│   ├── model/
│   │   └── monet.py               # Full MONet forward pipeline
│   │
│   ├── loss/
│   │ 	└── monet_loss.py   		 # Full MONet ELBO
│   │
│   └── config.py                   # slots, latent_dim, image_size
│
│
├── requirements.txt
└── README.md
```
---


## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
