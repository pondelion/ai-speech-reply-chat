import os

import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

text = 'こんにちは'

tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
# torch.save(tokenizer.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_medium_tokenizer.pth'))
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_medium_model.pth'))

input = tokenizer.encode(text, return_tensors='pt')

model.eval().to('cuda')
with torch.no_grad():
    output = model.generate(input.to('cuda'), do_sample=True, max_length=100, num_return_sequences=1)
    print(tokenizer.batch_decode(output))
