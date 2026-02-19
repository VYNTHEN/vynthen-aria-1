<!-- ====================== HERO ====================== -->

# 🚀 Vynthen Aria 1

<p align="center">
  <img src="https://img.shields.io/badge/Vynthen-AI-blueviolet?style=for-the-badge">
</p>

<p align="center">
  <b>Vynthen Aria 1 — 3B Coding & Website Generation AI</b><br>
  Built by <b>Vynthen AI</b>
</p>

---

<p align="center">
  <a href="https://huggingface.co/Vynthen/Vynthen-Aria-1">
    <img src="https://img.shields.io/badge/🚀%20Try%20the%20Model-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black" />
  </a>
</p>

<p align="center">
  ⭐ Run the model instantly on Hugging Face
</p>

---

## 🧠 About

**Vynthen Aria 1** is a **3B parameter AI model** created by **Vynthen AI**.

Designed for:

- 💻 Writing complex code  
- 🌐 Generating full websites from prompts  
- 🧠 Step-by-step reasoning  
- 🤖 Acting as a coding assistant  

Fine-tuned on **10,000 template-based examples** to create a mini-lovable developer AI.

Built on top of **Qwen2.5-3B-Instruct**.

---

## 📊 Model Details

| Feature | Info |
|---------|------|
| Model Name | Vynthen Aria 1 |
| Creator | Vynthen AI |
| Parameters | 3 Billion |
| Tasks | Coding, Website Generation, Chat |
| License | Apache-2.0 |

---

## ▶️ Quick Start

```bash
pip install transformers accelerate torch


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
