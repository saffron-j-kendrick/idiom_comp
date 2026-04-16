import os
import torch
from transformers import pipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

access_token = os.environ.get('HF_TOKEN_LLAMA')
if access_token is None:
    raise ValueError("HF_TOKEN_LLAMA is not set")
# llama_32 = "meta-llama/Llama-3.2-1B-Instruct"

llama_3_8b = "meta-llama/Meta-Llama-3-8B-Instruct"

prompt = [
    {"role": "system", "content": "You are a linguist who understands idiomatic phrases with a verb noun construction, such as 'kick the bucket'. You know that there is a potential idiomatic interpretation of the phrase and a literal one. When you receive a phrase, first tell me what the idiom phrase means and then you always create three idiomatic sentences and three literal sentences which use the phrase, however for the literal sentences you replace the noun within the construction with another noun so that the preceding part of the sentence stays exactly the same and only the noun changes. The replacement noun must work for all three pairs of sentences. These sentences should place the verb noun construction towards the end of a clause and then use a ',' and then 'it was a '. The goal is to leave each sentence unfinished with 'it was a' so that there are possible continuation words which you will also predict. You also provide potential next word continuations which could be nouns or adjectives that satisfy the idiom sentence and potential next word continuations that satisfy the literal sentence. Here is an example of the output I expect, using the phrase 'spill the beans'. This example shows how I want the sentences to be set out, ending each sentence unfinished and providing the next possible words to each sentence. idiom1: 'spill the beans'. prompt_idiom1: 'The suspect was nervous as he spilled the beans, it was a big'. prompt_literal1: 'The suspect was nervous as he spilled the drink, it was a big'. idiom1_answers = ['surprise', 'secret', 'mystery']. literal1_answers = ['mistake', 'problem', 'shock']. prompt_idiom2: 'The suspect hesitated during questioning before he spilled the beans, it was a big'. prompt_literal2: 'The suspect hesitated during questioning before he spilled the drink, it was a big'. idiom2_answers = ['question', 'moment', 'step']. literal2_answers = ['mistake', 'problem', 'shock']. prompt_idiom3: 'The suspect felt a lot of pressure from the staring officer and he spilled the beans, it was a big'. prompt_literal3: 'The suspect felt a lot of pressure from the staring officer and he spilled the drink, it was a big'. idiom3_answers = ['surprise', 'story' 'moment']. literal3_answers = ['mistake', 'problem', 'deal'.]. So for each prompt_idiom, use the provided phrase i.e. 'spill the beans', and then for each prompt_literal, replace the noun part with another noun i.e. 'spill the drink'. Each pair of prompt_idiom[x] and prompt_literal[x] should have the same starting context. The idiom_answers and literal_answers should be different from each other.  "},
    {"role": "user", "content": "Your next task is to create three idiomatic sentences and three literal sentences which use the phrase 'lift a finger', tell me the meaning of the phrase and provide the next word continuations for each sentence in the same format as the example. So you will replace 'finger' with 'hand' for literal sentences, and construct three pairs of sentences in total. I want five possible answers for each sentence, these answers must be unique so within each set, no repeats. Thank you. Can you do the same with 'hit the road' and replace 'road' with 'brakes', and 'scratch the surface' and replace 'surface' with record', and 'bite the dust' and replace 'dust' with 'pickkle'. Thanks."},
]

generator = pipeline(model=llama_3_8b, token = access_token, device=device, torch_dtype=torch.bfloat16)
generation = generator(
    prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=10000
)

# print(f"Generation: {generation[0]['generated_text']}")


# save the generation to a file
with open('generation.txt', 'w') as f:
    f.write(generation[0]['generated_text'])
print(f"Generation saved to generation.txt")
