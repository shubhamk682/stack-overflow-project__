import os,sys 
sys.path.append(os.getcwd)
from utils import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline 
import joblib  

STAGE='STAGE 2 PIPELINE'

def featurization(config_path):
    config=read_yaml(config_path=config_path)
    artifacts=config["artifacts"]
    prepared_data_dir=os.path.join(artifacts["ARTIFACTS_DIR"],artifacts["PREPARED_DATA"])
    train_data_path=os.path.join(prepared_data_dir,artifacts["TRAIN_DATA"])
    test_data_path=os.path.join(prepared_data_dir,artifacts["TEST_DATA"])    
    df_train=get_df(train_data_path)
    train_words=np.array(df_train["text"].str.lower().values.astype("U")) 
    ngrams=config["featurize"]["ngrams"]
    max_features=config["featurize"]["max_features"]
    bag_of_words=CountVectorizer(stop_words='english',max_features=max_features,ngram_range=(1,ngrams))
    tf_idf_transformer=TfidfTransformer(smooth_idf=False)
    pipeline =Pipeline([("bag_of_words",bag_of_words),("tf_idf_transform",tf_idf_transformer)])
    featurized_dir_path=os.path.join(artifacts["ARTIFACTS_DIR"],artifacts["FEATURE_DIR"])
    os.makedirs(featurized_dir_path,exist_ok=True)

    featurized_data_train_path=os.path.join(featurized_dir_path,artifacts["FEATURIZED_TRAIN_DATA"])
    featurized_data_test_path=os.path.join(featurized_dir_path,artifacts["FEATURIZED_TEST_DATA"])
    pipeline_save_path=os.path.join(featurized_dir_path,artifacts["TRANSFORMER_FILE"])

    print(featurized_data_train_path,featurized_data_test_path,pipeline_save_path)
    pipeline.fit(train_words)
    joblib.dump(pipeline,pipeline_save_path)
    train_words_tfidf_matrix=pipeline.transform(train_words)
    save_matrix(df_train,train_words_tfidf_matrix,featurized_data_train_path)

    df_test=get_df(test_data_path)
    test_words=np.array(df_test.text.str.lower().values.astype("U"))
    test_words_tfidf_matrix=pipeline.transform(test_words)
    save_matrix(df_test,test_words_tfidf_matrix,featurized_data_test_path)
