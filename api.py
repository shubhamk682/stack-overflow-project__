from inferencing_pipeline import prediction_service
from fastapi import FastAPI,Response
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
app=FastAPI()

@app.get("/")
async def model():
    message="<h1>Welcome to inferencing api</h1>"
    message2="<h2>Click on the below link to get the predictions</h2>"
    link="<a href='http://127.0.0.1:8000/predict'>Predict for Inferencing on batch data</a>"
    message=message+message2+link
    return HTMLResponse(content=message)
#uvicorn.run(app=app)

@app.get("/predict")
async def predict(response:Response):
    model_path="artifacts/models/model.pkl"
    transformer_path="artifacts/features/transformer.pkl"
    prediction_service.inferencing_pipeline_batch(model_path,transformer_path,"data.tsv")
    predicted_Data="prediction.csv"
    data=pd.read_csv(predicted_Data)
    table_html=data.to_html(index=False)
    header="<h1>Stack overflow data inferencing on batch data</h1>"
    table_html=header+table_html
    response.headers["Content-type"]="text/html"
    return HTMLResponse(content=table_html)


uvicorn.run(app=app)
