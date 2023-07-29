import os,sys
sys.path.append(os.getcwd())
from utils1 import*
import joblib
import numpy as np

def inferencing_pipeline_batch(model_path,transformer_path,batch_data):
    model=joblib.load(open(model_path,"rb"))
    transformer_model=joblib.load(open(transformer_path,"rb"))
    test_data=get_df_for_predict(batch_data)
    test_words=np.array(test_data.text.str.lower().values.astype("U"))
    test_transformer=transformer_model.transform(test_words)
    y_pred=model.predict(test_transformer)
    ypred_array=np.array(y_pred)
    test_data["predicted_values"]=ypred_array
    test_data["predicted_values"]=test_data["predicted_values"].replace({0.0:"not python query",1.0:"python related query"})
    test_data.to_csv("prediction.csv",index=False)
    return y_pred

def inferencing_single(model_path,transformer_path,query):
    model=joblib.load(open(model_path,"rb"))
    transformer_model=joblib.load(open(transformer_path,"rb"))
    test_transformer=transformer_model.transform([query])
    y_pred = model.predict(test_transformer)
    if y_pred==[1.0]:
        return "python related query"
    else:
        return "non python query"
    ''
#if __name__=="__main__":
    model_path="artifacts\models.pkl"
    transformer_path="artifacts\features\transformer.pkl"
    #prediction=inferencing_single(model_path,transformer_path,query=input("please enter query...")
    prediction=inferencing_single(model_path,transformer_path,query='''<p>Yes, of course there is :-) object oriented techniques are a tool ... if you use the wrong tool for a given job, you will be over complicating things (think spoon when all you need is a knife).</p> <p>To me, I judge "how much" by the size and scope of the project. If it is a small project, sometimes it does add too much complexity. If the project is large, you will still be taking on this complexity, but it will pay for itself in ease of maintainability, extensibility, etc.</p>''')
    print(prediction)



