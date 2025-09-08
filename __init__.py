import os, io, requests, re
from io import open
import pandas as pd
import glob
import time
import csv
from datasets import Dataset, DatasetDict
from datasets.features import Features
from appdirs import user_data_dir
from pathlib import Path
from .process_underscore impZort restore_docs

DATA_URL="https://raw.githubusercontent.com/disrpt/sharedtask2023/main/data"


def parse_conll_stream(file_stream):
    names = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc','doc_id']
    sentence = {name: [] for name in names}|{'mwe':[]}
    mwe_id=[]
    for line in file_stream:
        line = line.strip()
        if line.startswith("#"):
            if "doc_id" in line:
                doc_id=line.split('=')[-1].strip()
            continue
        if not line:
            if sentence['id']:
                yield sentence
                sentence = {name: [] for name in names}|{'mwe':[]}
            continue
        token_data = line.split('\t') + [doc_id]
        for name, value in zip(names, token_data):
            if name=='id' and not value.isnumeric():
                sentence['mwe'] += [value]
            else:
                sentence[name].append(value)
def read(path):
    if '.conllu' in path :
        df=pd.DataFrame(parse_conll_stream(open(path)))
        df['doc_id']=df.doc_id.map(lambda x:x[0])
        return df
    if '.rels' in path:
        return pd.read_csv(path ,sep='\t',quoting=csv.QUOTE_NONE)


corpora_files={
    "eng.rst.rstdt": "RSTtrees-WSJ-main-1.0/*/*.edus",
    "tur.pdtb.tdb": "*.txt",
    "zho.pdtb.cdtb":"*.raw",
    "eng.pdtb.pdtb":"*/wsj_*"
}


def fetch_files(config_name,corpora_paths):
    return list(Path(corpora_paths[config_name]).glob(corpora_files[config_name]))

def harvest_text(files):
	docs = {}
	for file_ in files:
		docname = os.path.basename(file_).split(".")[0]
		try:
			text = io.open(file_,encoding="utf8").read()
		except:
			text = io.open(file_,encoding="Latin1").read()  # e.g. wsj_0142
		text = text.replace(".START","")  # Remove PDTB .START codes
		text = re.sub(r'\s','', text)  # Remove all whitespace
		docs[docname] = text
	return docs

def download_file(config_name, url):
    directory = user_data_dir(config_name)
    os.makedirs(directory, exist_ok=True)
    
    filename = url.split('/')[-1]
    path = os.path.join(directory, filename)

    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        return path
    else:
        return None

def load_dataset(config_name, ext,corpora_paths,data_url=DATA_URL):
    exts = 'rels','conllu','tok'
    splits = 'train','dev','test'
    urls= [ f"{data_url}/{config_name}/{config_name}_{split}.{ext}" for ext in exts for split in splits]
    paths = [download_file(config_name,u) for u in urls]
    docs2text = harvest_text(fetch_files(config_name,corpora_paths=corpora_paths))
    restore_docs(user_data_dir(config_name),docs2text)
    get_split = lambda x:x.split('_')[-1].split('.')[0].replace('dev','validation')
    split_file = {get_split(p):p for p in paths if p.endswith(ext)}
    dataset = DatasetDict({split: Dataset.from_pandas(read(file)) for split,file in split_file.items()})
    return dataset