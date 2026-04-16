import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLMModel, RobertaConfig, BertConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

access_token = os.environ.get('HF_TOKEN_LLAMA')

dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B')}

# dev_model_configs = {'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), "mistralai/Mistral-7B-v0.1")}

# dev_model_configs = {"tiiuae/Falcon3-7B-Base": (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), "tiiuae/Falcon3-7B-Base")    }

# dev_model_configs = {"Qwen/Qwen2.5-7B" : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), "Qwen/Qwen2.5-7B")}

# dev_model_configs = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" : (AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")}

# dev_model_configs = {'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2')}

# dev_model_configs = {'bert-base-uncased': (AutoConfig.from_pretrained("bert-base-uncased"), AutoModelForMaskedLM.from_pretrained("bert-base-uncased"), AutoTokenizer.from_pretrained("bert-base-uncased"), "bert-base-uncased")}

# dev_model_configs = {"google/multiberts-seed_3": (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForCausalLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), "google/multiberts-seed_3")}


# dev_model_configs = {"FacebookAI/roberta-base" : (AutoConfig.from_pretrained("FacebookAI/roberta-base"), AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base"), AutoTokenizer.from_pretrained("FacebookAI/roberta-base"), "FacebookAI/roberta-base")}

# dev_model_configs = {"tohoku-nlp/bert-base-japanese" : (AutoConfig.from_pretrained("tohoku-nlp/bert-base-japanese"), AutoModelForMaskedLM.from_pretrained("tohoku-nlp/bert-base-japanese"), AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese"), "tohoku-nlp/bert-base-japanese")}

        #                      'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), "mistralai/Mistral-7B-v0.1"),
#                     'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = token) , 'meta-llama/Llama-3.2-3B'),
#                     'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2'),
#                      "tiiuae/Falcon3-7B-Base": (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), "tiiuae/Falcon3-7B-Base") }

# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" : (AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
#                      'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2'),
#                      "Qwen/Qwen2.5-7B" : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), "Qwen/Qwen2.5-7B"),
#                      'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), "mistralai/Mistral-7B-v0.1"),
#                      'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = token) , 'meta-llama/Llama-3.2-3B')



# dev_model_configs = { 'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token = token_mistral), "mistralai/Mistral-7B-v0.1"),
#                     'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = token) , 'meta-llama/Llama-3.2-3B'),
#                     'Qwen/Qwen2.5-7B' : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), "Qwen/Qwen2.5-7B"),
#                      "tiiuae/Falcon3-7B-Base": (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), "tiiuae/Falcon3-7B-Base") }


def load_model(name, all_hidden_states=True):
    configuration_class, model_class, tokeniser_class, weights = dev_model_configs[name]
    model, tokeniser = load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states)
    return model, tokeniser

def load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states=True):
    config = configuration_class.from_pretrained(weights, output_hidden_states=all_hidden_states)
    model = model_class.from_pretrained(weights, config=config)
        
    tokeniser = tokeniser_class.from_pretrained(weights)
    
    return model, tokeniser


def load_roberta():
    return load_model('roberta-base')
