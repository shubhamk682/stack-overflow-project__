import os,sys
sys.path.append(os.getcwd())
from utils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

STAGE="Stage3 pipeline"

def training(config_path):
    config=read_yaml(config_path)
    #accessing the train.pkl files
    artifacts=config["artifacts"]["ARTIFACTS_DIR"]
    print(artifacts)
    featurized_data_dir_path=os.path.join(artifacts,config["artifacts"]["FEATURE_DIR"])
    featurized_data_train_path=os.path.join(featurized_data_dir_path,config["artifacts"]["FEATURIZED_TRAIN_DATA"])

#Output the model dir
    model_dir_path=os.path.join(artifacts,config["artifacts"]["MODEL_DIR"])
    os.makedirs(model_dir_path,exist_ok=True)
    model_path=os.path.join(model_dir_path,config["artifacts"]["MODEL_PATH"])
    matrix=joblib.load(featurized_data_train_path)
    print(matrix.toarray().shape)
    print(matrix[:,1].toarray())
    X=matrix[:,2:]
    labels=np.squeeze(matrix[:,1].toarray())
    print(labels)
    model=RandomForestClassifier(n_estimators=120,min_samples_split=16,random_state=2021,n_jobs=-1)
    model.fit(X,labels)
    joblib.dump(model,model_path)

    
    









