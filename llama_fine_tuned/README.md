---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
---

# Mental Health Support Chatbot

A fine-tuned language model specialized in providing empathetic and supportive mental health conversations.

## Model Details

### Model Description

This model is a fine-tuned version of Llama-3.2-3B-Instruct, specifically trained to provide supportive mental health conversations. It has been trained on therapeutic dialogue data to better understand and respond to mental health concerns with empathy and professional guidance.

- **Developed by:** Nada
- **Model type:** Causal Language Model
- **Language(s) (NLP):** English
- **License:** Same as base model
- **Finetuned from model:** meta-llama/Llama-3.2-3B-Instruct



### Direct Use

This model is designed to be used as a mental health support chatbot, providing:
- Empathetic responses to mental health concerns
- Supportive conversation
- Professional therapeutic guidance
- Crisis intervention support

### Downstream Use

The model can be integrated into:
- Mental health support applications
- Therapeutic conversation platforms
- Crisis intervention systems
- Mental health education tools

### Out-of-Scope Use

This model should NOT be used for:
- Emergency medical advice
- Professional therapy replacement
- Legal advice
- Harmful or manipulative purposes

## Bias, Risks, and Limitations

### Limitations

- The model is not a replacement for professional mental health care
- May not handle all crisis situations appropriately
- Limited to English language interactions
- May have biases present in the training data

### Recommendations

Users should:
- Always provide clear disclaimers that this is an AI assistant
- Include emergency contact information
- Have human oversight for critical situations
- Monitor conversations for potential risks
- Provide clear instructions for seeking professional help

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nada013/mental-health-chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example usage
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

### Training Data

The model was fine-tuned on:
- MentalChat16K dataset
- Therapeutic conversation data
- Mental health support dialogues

### Training Procedure

#### Training Hyperparameters

- **Training regime:** Mixed precision training
- **Learning rate:** 2e-5
- **Batch size:** 1
- **Gradient accumulation steps:** 8
- **Epochs:** 3
- **Warmup ratio:** 0.03

#### Speeds, Sizes, Times

- **Base model:** Llama-3.2-3B-Instruct
- **Fine-tuning time:** [22 hours]
- **Model size:** ~3.2B parameters

## Evaluation

### Testing Data

The model was evaluated on:
- Mental health conversation datasets
- Therapeutic dialogue benchmarks
- User interaction testing

### Metrics

- Response quality
- Empathy level
- Professional tone
- Safety measures
- Crisis detection

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Transformer-based language model
- **Objective:** Generate supportive and empathetic responses to mental health concerns
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
- **LoRA parameters:**
  - r: 8
  - alpha: 32
  - dropout: 0.1
  - target modules: q_proj, k_proj, v_proj, o_proj

### Compute Infrastructure

- **Hardware:** [gpu: nividia rtx 3070Ti 8GB VRAM]
- **Training time:** [22 hours]
- **Framework:** PyTorch
- **Libraries:** Transformers, PEFT

## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{mental-health-chatbot-2024,
  author = {Nada},
  title = {Mental Health Support Chatbot},
  year = {2024},
  publisher = {HuggingFace},
  journal = {HuggingFace Hub},
  howpublished = {\url{https://huggingface.co/nada013/mental-health-chatbot}}
}
```

## Contact
- Email: [nadak2982@gmail.com]
