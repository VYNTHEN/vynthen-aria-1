<!-- ====================== HERO ====================== -->

<h1 align="center">🚀 Vynthen Aria 1</h1>

<p align="center">
<b>The Next-Gen 3B Text Generation, Coding & Website Generation AI</b><br>
Built by <b>Vynthen AI</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Model-3B-blueviolet?style=for-the-badge">
<img src="https://img.shields.io/badge/Base-Qwen2.5--3B-Instruct-black?style=for-the-badge">
<img src="https://img.shields.io/badge/License-Apache--2.0-green?style=for-the-badge">
</p>

<p align="center">
<a href="https://huggingface.co/Vynthen/Vynthen-Aria-1">
<img src="https://img.shields.io/badge/🚀%20Run%20Vynthen%20Aria%201%20Now-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black">
</a>
</p>

<p align="center">
⭐ Generate full apps, websites, and complex code instantly
</p>

---

<!-- ====================== ABOUT ====================== -->

## 🧠 About Vynthen Aria 1

**Vynthen Aria 1** is a **3 Billion parameter AI developer model** created by **Vynthen AI**.

It is designed to act like a **mini lovable coding assistant** that can:

- 💻 Write complex production-ready code  
- 🌐 Generate full websites from prompts  
- 🧠 Explain solutions step-by-step  
- ⚡ Build apps, bots, and tools  
- 🤖 Answer like **"I am Vynthen AI"**  

Fine-tuned on **10,000 high-quality template-based examples**.

Built on **Qwen2.5-3B-Instruct** for strong reasoning + coding ability.

---

<!-- ====================== FEATURES ====================== -->

## ✨ Key Features

### 💻 Advanced Coding
- Python, JS, React, C++, HTML, CSS
- Full projects with folder structure
- Clean production-ready code

### 🌐 Website Generator
- Landing pages  
- Dashboards  
- Portfolios  
- E-commerce UIs  

Just prompt like:

> "Build a modern fintech dashboard with login page"

### 🧠 Smart Reasoning
- Step-by-step explanations  
- Debugging help  
- Architecture planning  

### ⚡ Fast & Lightweight
- Only **3B parameters**
- Runs on consumer GPUs
- Optimized with LoRA fine-tuning

---

<!-- ====================== SPECS ====================== -->

## 📊 Model Specifications

| Feature | Details |
|---------|---------|
| Model Name | **Vynthen Aria 1** |
| Creator | **Vynthen AI** |
| Parameters | **3 Billion** |
| Dataset | 10k Template-Based Examples |
| Tasks | Coding, Websites, Chat |
| Training Method | LoRA Fine-Tuning |
| License | Apache-2.0 |

---

<!-- ====================== QUICK START ====================== -->

## ▶️ Quick Start

### Install

```bash
pip install transformers accelerate torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Vynthen/Vynthen-Aria-1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Create a responsive coffee shop website with menu and booking form"

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=600)

print(tokenizer.decode(output[0], skip_special_tokens=True))

<!-- ====================== DEMOS ====================== -->
🌐 Example Prompts
Build a full SaaS landing page with pricing, testimonials, and login.

Create a Discord bot with moderation commands and database.

Make a React dashboard with charts and authentication.

Generate a modern portfolio with animations and dark mode.
