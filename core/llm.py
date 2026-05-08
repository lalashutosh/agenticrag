# core/llm.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# =========================
# 1. Global handles
# =========================

_tokenizer = None
_model = None
_pipe = None


# =========================
# 2. Model loading logic
# =========================

def load_model():
    """
    Switch between full precision and quantized model based on available VRAM.
    """
    import torch

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram_gb >= 16:
            return load_full_precision()
        else:
            return load_quantized()
    else:
        return load_full_precision()


def load_full_precision():
    global _tokenizer, _model, _pipe

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer
    )

    return _pipe


def load_quantized():
    global _tokenizer, _model, _pipe

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer
    )

    return _pipe


# =========================
# 3. Ensure model is loaded once
# =========================

if _pipe is None:
    load_model()


# =========================
# 4. Core invoke API (UNCHANGED CONTRACT)
# =========================

def invoke(messages, max_new_tokens=1024, json_mode=False):
    """
    LangGraph-compatible LLM call.
    """

    global _pipe, _tokenizer

    # Convert chat format → Qwen chat template
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = _pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # IMPORTANT: deterministic RAG core
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        return_full_text=False,
        generation_config=None    # suppress max_length warning
    )

    text = outputs[0]["generated_text"]

    # Optional JSON enforcement hook (used by agents)
    if json_mode:
        return text.strip()

    return text.strip()