import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from collections import defaultdict
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, AutoModel, AutoTokenizer, TFAutoModel, pipeline, AdamW, BertConfig, BertModel, get_linear_schedule_with_warmup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import argparse
import numba
import util

device = None # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_format, has_output=True):
  raw_lines = None
  with open(f"{file_format}.data.jsonl", "r") as f:
    raw_lines = f.readlines()

  raw_labels = None
  if has_output == True:
    with open(f"{file_format}.label.json", "r") as f:
      raw_labels = json.loads(f.read())

  return ([json.loads(line) for line in raw_lines], raw_labels)

def get_xy(tweets, labels):
  assert len(tweets) == len(labels)
  tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
  X = []
  Y = np.zeros(len(labels))
  for i, tweet_set in enumerate(tweets):
    arr = []
    for tweet in tweet_set:
      tweet = ' '.join(tokenizer.tokenize(tweet['text']))
      arr.append(tweet)
    X.append(arr)

  return X, Y

SEQ_LEN = 50

class TweetDataset(Dataset):
  def __init__(self, tweet_sets, targets, tokenizer, max_len):
    self.tweet_sets = tweet_sets
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.tweet_sets)
  
  def __getitem__(self, index):
    

    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'targets': torch.tensor(self.targets[index], dtype=torch.long)}
  
def create_data_load(X, Y, tokenizer, max_len, batch_size):
  td = TweetDataset(X, Y, tokenizer, max_len)
  return DataLoader(td, batch_size=batch_size, num_workers=0)
  
class RumourClassifier(nn.Module):
  def __init__(self, num_classes, hidden_size, sequence_length, num_layers):
    super(RumourClassifier, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.sequence_length = sequence_length
    
    self.bert = BertModel.from_pretrained("bert-base-cased")
    self.drop = nn.Dropout(p=0.3)
    self.gru = nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
    self.out = nn.Softmax(dim=1)

  def forward(self, batch_size, input_ids, attention_masks):

    exit(0)

def main():
  parser = argparse.ArgumentParser(description="TweetRumourCL")
  parser.add_argument('--train', metavar="T", type=str, required=True, help="Train data file location")
  parser.add_argument("--dev", metavar="D", type=str, required=True, help="Dev data file location")
  parser.add_argument("--test", metavar="P", type=str, required=True, help="Test data file location")
  parser.add_argument("--device", type=str, help="Device", default="cpu")
  args = parser.parse_args()
  print("TRAIN =", args.train)
  print("DEV =", args.dev)
  print("TEST =", args.test)
  print("DEVICE =", args.device)

  global device
  device = torch.device(args.device)

  train_entries, train_labels = load_data(args.train)
  dev_entries, dev_labels = load_data(args.dev)
  trainX, trainY = get_xy(train_entries, train_labels)
  
  tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

  train_ds = create_data_load(trainX, trainY, tokenizer, 512, 2)

  model = RumourClassifier(2, 20, SEQ_LEN, 2)
  model = model.to(device)

  for d in train_ds:
    model(2, d['input_ids'], d['attention_mask'])

if __name__ == "__main__":
  main()

