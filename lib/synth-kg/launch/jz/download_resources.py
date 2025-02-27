from transformers import AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained("meta-llama/llama-2-7b-hf")
AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-hf")

AutoModelForCausalLM.from_pretrained("xz97/AlpaCare-llama2-13b")
AutoTokenizer.from_pretrained("xz97/AlpaCare-llama2-13b")
