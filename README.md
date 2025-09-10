
This repository contains a Jupyter notebook demonstrating how to fine-tune Gemma models (specifically Gemma 2B) using Low Rank Adaptation (LoRA) in Keras with KerasNLP.
# Overview
Gemma is a family of lightweight open models from Google. This notebook shows:
Pre-training and fine-tuning concepts for LLMs.
Using LoRA to efficiently fine-tune large models by reducing trainable parameters.
Fine-tuning on a subset of the Dolly 15k dataset (prompt-response pairs).
Inference examples before and after fine-tuning.

# Key techniques:
FaceNet-inspired embeddings (though not directly used; the notebook focuses on text generation).
LoRA rank configuration (default: 4).
Optimization with AdamW and sparse categorical crossentropy.

The fine-tuned model improves response quality for instruction-following tasks.
# Requirements

Python 
Keras
KerasNLP
Access to Gemma models via Kaggle (requires Kaggle API key).
GPU runtime recommended (e.g., T4 GPU on Colab for fine-tuning).
