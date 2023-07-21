import os,sys
sys.path.append(os.getcwd())
from utils import *
import numpy as np
import joblib
import sklearn.metrics as metrics
import math
import json


def evaluate(config_path): 
    config=read_yaml(config_path)
    artifacts=config["artifacts"]["ARTIFACTS_DIR"]
    featurized_data_dir_path=os.path.join(artifacts,config["artifacts"]["FEATURE_DIR"])
    featurized_test_data_path=os.path.join(featurized_data_dir_path,config["artifacts"]["FEATURIZED_TEST_DATA"])
    model_dir=os.path.join(artifacts,config["artifacts"]["MODEL_DIR"])
    model_path=os.path.join(model_dir,config["artifacts"]["MODEL_PATH"])
    
    model=joblib.load(model_path)
    matrix_test=joblib.load(featurized_test_data_path)
    labels=np.squeeze(matrix_test[:,1].toarray())
    X=matrix_test[:,2:]
    prediction_by_class=model.predict_proba(X)
    predictions=prediction_by_class[:,1]
    print(prediction_by_class)
    PREC_JSON_PATH=config["artifacts"]["PREC_JSON_PATH"]
    ROC_JSON_PATH=config["artifacts"]["JSON_ROC_PATH"]
    score_json_path=config["artifacts"]["SCORE_FILE_PATH"]
    avg_precision=metrics.average_precision_score(labels,predictions)
    roc_auc=metrics.roc_auc_score(labels,predictions)
    scores={
        "roc_auc_score":roc_auc,
        "average_precision_score":avg_precision
    }
    print(scores)
    json.dump(scores,open(score_json_path,"w"))
    precision,recall,threshold=metrics.precision_recall_curve(labels,predictions)
    nth_points=math.ceil(len(precision)/100)
    print(nth_points)
    prc_point=list(zip(precision,recall,threshold))[::nth_points]
    prec_points=[{"precision":p,"recall":r,"threshold":t} for p,r,t in prc_point]
    json.dump(prec_points,open(PREC_JSON_PATH,"w"))
    fpr,tpr,threshold=metrics.roc_curve(labels,predictions)
    roc_data={"roc":[{"fpr":fp,"tpr":tp,"threshold":t} for fp,tp,t in zip(fpr,tpr,threshold)]}
    json.dump(roc_data,open(ROC_JSON_PATH,"w"))
