from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def invoke(messages):
    prompt = ""

    for m in messages:
        # Handle LangChain message objects
        if hasattr(m, "content"):
            role = type(m).__name__.replace("Message", "").lower()
            content = m.content

        # Handle dict format
        elif isinstance(m, dict):
            role = m["role"]
            content = m["content"]

        else:
            continue

        prompt += f"{role.upper()}: {content}\n"

    prompt += "ASSISTANT:"

    output = pipe(prompt, max_new_tokens=512, do_sample=False)

    return output[0]["generated_text"].split("ASSISTANT:")[-1].strip()