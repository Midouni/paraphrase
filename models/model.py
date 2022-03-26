import os
import tensorflow
from transformers import pipeline, AutoTokenizer,GPT2LMHeadModel,AutoConfig
from simpletransformers.t5 import T5Model

TOKENIZERS_PARALLELISM=False

#---------------------------------------GPT2 fine tuned model------------------------------------------#
#Load GPT2 Fine tuned model
gpt2_pipeline_path = os.path.join(os.path.dirname(__file__),'gpt-generator')
test = GPT2LMHeadModel.from_pretrained(gpt2_pipeline_path)
gpt2_paraphraser_generator = pipeline('text-generation',gpt2_pipeline_path,local_files_only=True)


#Function to generate paraphrases with GPT2 fine tuned model
def gpt2_paraphraser(input_sentence):
    generated = gpt2_paraphraser_generator('<s>'+input_sentence+'</s>>>>><p>')
    return generated[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]

#---------------------------------------T5 fine tuned model---------------------------------------#
#Load T5 Fine tuned model & prepare model arguments
t5_model_path = os.path.join(os.path.dirname(__file__),'t5-generator')
args = {

"overwrite_output_dir": True,
"max_seq_length": 256,
"max_length": 50,
"top_k": 50,
"top_p": 0.95,
"num_return_sequences": 2,
"use_cuda" : False,
"fp16" : False
}
t5_paraphraser_generator = T5Model("t5",t5_model_path,args=args,use_cuda = False)
prefix = "paraphrase"

#Function to generate paraphrases with T5 fine tuned model
def t5_paraphraser(input_sentence):
    generated = t5_paraphraser_generator.predict([prefix+input_sentence])[0]
    return generated[1]