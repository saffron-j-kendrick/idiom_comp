import os
import torch
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLMModel, RobertaConfig, BertConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM



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
# get assistant message text from chat-style output
assistant_text = generation[0]["generated_text"][-1]["content"]
with open("generation.txt", "w") as f:
  f.write(assistant_text)




# dev_model_configs = {'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", token = access_token), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token = access_token), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token = access_token), "mistralai/Mistral-7B-v0.1")}

# def load_model(name, all_hidden_states=True):
#     configuration_class, model_class, tokeniser_class, weights = dev_model_configs[name]
#     model, tokeniser = load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states)
#     return model, tokeniser

# def load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states=True):
#     config = configuration_class.from_pretrained(weights, output_hidden_states=all_hidden_states)
#     model = model_class.from_pretrained(weights, config=config)
#     model = model.to(device)
        
#     tokeniser = tokeniser_class.from_pretrained(weights)
    
#     return model, tokeniser





# models = dev_model_configs.keys()
# print(models)
# for model_name in models:
#     model, tokeniser = load_model(model_name)

#     if tokeniser.pad_token is None:
#             if tokeniser.eos_token:
#                 tokeniser.pad_token = tokeniser.eos_token
#             else:
#                 tokeniser.add_special_tokens({'pad_token': '<pad>'})


#     print(model)


