import tensorflow
from transformers import pipeline, AutoTokenizer,GPT2LMHeadModel,AutoConfig
#from simpletransformers.t5 import T5Model


#--------------------------GPT2 fine tuned model---------------------------#
#import gpt fine-tuned model 

gpt2_pipeline_path = "/Users/mohamed/Desktop/Paraphrase/models/gpt-generator"
gpt2_paraphraser_generator = pipeline('text-generation',gpt2_pipeline_path,local_files_only=True)

#generate paraphrase using gpt2
def gpt2_paraphraser(input_sentence):
    p = gpt2_paraphraser_generator('<s>'+input_sentence+'</s>>>>><p>')
    return p[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]

#--------------------------T5 fine tuned model---------------------------#

#import t5 fine-tuned model & args
# t5_model_path = "/Users/mohamed/Desktop/Paraphrase/models/t5-generator"
# args = {
# "overwrite_output_dir": True,
# "max_seq_length": 256,
# "max_length": 50,
# "top_k": 50,
# "top_p": 0.95,
# "num_return_sequences": 3,
# "use_cuda":False
# }

# t5_paraphraser_generator = T5Model("t5",t5_model_path,args=args)
# prefix = "paraphrase"

#generate paraphrase using t5
# def t5_paraphraser(input_sentence):
#     preds = t5_paraphraser_generator.predict([prefix+input_sentence])[0]
#     print(preds)
#     return preds[1]