import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Salesforce/xgen-7b-8k-inst"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).cuda()
if torch.cuda.is_available():
    model = model.to("cuda:0")
print("model device:", model.device)


def greet(all_input):
    header = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )

    prompt = "### Human: "

    inputs = tokenizer(header + prompt + all_input, return_tensors="pt")

    sample = model.generate(input_ids=inputs['input_ids'].to(model.device), 
                            attention_mask=inputs['attention_mask'].to(model.device),
                            do_sample=True, max_new_tokens=2048, top_k=100, eos_token_id=50256)
    output = tokenizer.decode(sample[0])
    prefix = "### Assistant:"
    result  = output[output.strip().index(prefix)+len(prefix):]
    return result

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    
demo.launch(server_port=8688, share=True, show_api=True) 

# CUDA_VISIBLE_DEVICES=7 gradio slow_agent/run_gradio.py