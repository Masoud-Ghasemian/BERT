
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


from torch import nn, optim
#from torch.utils.data import Dataset, DataLoader
#import pickle
import streamlit as st
import sklearn
from PIL import Image


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['negative', 'positive']

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert =BertModel.from_pretrained('bert-base-cased'  , return_dict=False )
    self.drop = nn.Dropout(p = 0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

# loading the trained model
PATH = './model.pkl'
model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#tokenizer_path = "/content/tokenizer.pkl"
#tokenizer = pickle.load(open(tokenizer_path, 'rb'))




def prediction(tweet):
  class_names = ['negative', 'positive']
  encoding = tokenizer.encode_plus(tweet, max_length=10,
                        add_special_tokens=True,
                        pad_to_max_length = True,
                        return_attention_mask= True,
                        return_token_type_ids=False,
                        return_tensors = 'pt',
                        truncation = True)
  input_ids = encoding['input_ids']
  attention_mask =encoding['attention_mask']
  #input_ids = input_ids.to(device)
  #attention_mask = attention_mask.to(device)
  outputs = model(input_ids = input_ids,attention_mask = attention_mask)
  _, pred = torch.max(outputs, dim = 1)
  pred = pred.cpu()[0]
  

  return class_names[pred]
         
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Sentiment Analysis using BERT</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
     
    tweet = st.text_input("label goes here")


    result =""

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(tweet)
        st.success('This sentence is {}'.format(result))
        #print(LoanAmount)
     
if __name__=='__main__': 
    main()
