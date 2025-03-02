from transformers import AutoTokenizer, LlamaForCausalLM
import sys
path=sys.argv[1]
model = LlamaForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
print('inputs:', inputs)
# generate_ids = model.generate(inputs.input_ids, max_length=30)
