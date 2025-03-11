from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained("meta-llama/llama-2-7b-hf")
AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-hf")

AutoModelForCausalLM.from_pretrained("xz97/AlpaCare-llama2-13b")
AutoTokenizer.from_pretrained("xz97/AlpaCare-llama2-13b")

SentenceTransformer("FremyCompany/BioLORD-2023")
AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")
