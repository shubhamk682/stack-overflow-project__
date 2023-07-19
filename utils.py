# healper functions
# moduls will be writing over here 
from xml.etree import ElementTree as ET 
import re
import random
import yaml 
import pandas as pd
import scipy.sparse as sparse
import joblib 
import numpy as np

def process_posts(f_in, target_tag, f_out_train, f_out_test, split):
    try:
        root = ET.fromstring(f_in)
        for child in root:
            pid = child.get('Id', "")
            label = 1 if target_tag in child.get('Tags', "") else 0
            title = re.sub(r"\s+", " ", child.get('Title', "")).strip()
            body = re.sub(r"\s+", " ", child.get('Body', "")).strip()
            text = title + " " + body
            f_out = f_out_train if random.random() > split else f_out_test
            f_out.write(f"{pid}\t{label}\t{text}\n")
    except Exception as e:
        raise e
    
def read_yaml(config_path):
    try:
        with open (config_path,"r") as data:
            config_data=yaml.safe_load(data)
        return config_data
    except Exception as e:
        raise e
    
def get_df(path_to_data,sep="\t"):
    df=pd.read_csv(path_to_data,encoding="utf-8",header=None,delimiter=sep,names=["ID","label","text"])
    return df    


def save_matrix(df, matrix, out_path):
    id_matrix = sparse.csr_matrix(df.ID.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T
    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")
    joblib.dump(result, out_path)                


#except Exceptional as e:
                #raise e