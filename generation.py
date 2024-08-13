import os
import logging
import torch
import streamlit as st
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

@st.cache_resource
def load_model_gen():
    try:
        with st.status("Loading the model.."):
             if os.environ.get('remote_model_path'):
                model_path = os.environ.get('remote_model_path')
             else:
                model_path = 'magnolia-psychometrics/bandura-v1' 
             time.sleep(1)
             st.write("Model found!")
             time.sleep(1)
             time.sleep(1)
             if 'generator' not in st.session_state:
                 st.session_state.generator = pipeline(task='text-generation', model=model_path, tokenizer=model_path)
             st.write("Model loaded!")
    except Exception as e:
        logging.error(f'Error while loading models/tokenizer: {e}')
        st.error('Error while loading models/tokenizer.')  

#def load_model():
#    
#    keys = ['generator']
#    
#    if any(st.session_state.get(key) is None for key in keys):
#        
#        with st.spinner('Loading the model...'):
#            try:
#                if os.environ.get('remote_model_path'):
#                    model_path = os.environ.get('remote_model_path')
#                else:
#                    model_path = 'gpt2'
#
#                st.session_state.generator = pipeline(task='text-generation', model=model_path, tokenizer=model_path)
#
#                logging.info('Loaded models and tokenizer!')
#
#            except Exception as e:
#                logging.error(f'Error while loading models/tokenizer: {e}')

def generate_items(constructs, prefix='', **kwargs):

    with st.spinner(f'Generating item(s) for `{constructs}`...'):
        construct_sep = '#'
        item_sep = '@'

        constructs = constructs if isinstance(constructs, list) else [constructs]
        encoded_constructs = construct_sep + construct_sep.join([x.lower() for x in constructs])
        encoded_prompt = f'{encoded_constructs}{item_sep}{prefix}' 

        outputs = st.session_state.generator(encoded_prompt, **kwargs)    
        truncate_str = f'{encoded_constructs}{item_sep}'
        
        item_stems = []
        for output in outputs:
            item_stems.append(output['generated_text'].replace(truncate_str, ''))
        
        return item_stems
        
def get_next_tokens(prefix, breadth=5):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Encode the prefix
    inputs = tokenizer(prefix, return_tensors='pt')

    # Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # Only consider the last token for next token predictions
    last_token_logits = logits[:, -1, :]

    # Get the indices of the top 'breadth' possible next tokens
    top_tokens = torch.topk(last_token_logits, breadth, dim=1).indices.tolist()[0]

    # Decode the token IDs to tokens
    next_tokens = [tokenizer.decode([token_id]) for token_id in top_tokens]

    return next_tokens
