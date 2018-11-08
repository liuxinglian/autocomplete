import json
import sys
import os
from gensim.models import FastText
from tqdm import tqdm

fname = sys.argv[1]
tname = sys.argv[2]

if not os.path.exists(tname):
  with open(fname) as json_file:
    data = json_file.readlines() 
  print("this file is not here yet\n")
  exit(1)
  with open(tname,'w') as out:
    for s in data:
      dic = json.loads(s)
      print(dic["text"],file=out)

if os.path.exists(tname):
  with open(fname) as f:
    reviews = f.readlines()

tokens = [text.lower().split(" ") for text in tqdm(reviews)]

model = FastText(tokens, min_count=1)
print(len(model.wv.vocab))