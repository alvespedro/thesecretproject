import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#from utils.io_utils import load_config
#from utils.model_utils import load_model, load_tokenizer, predict_comment


app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#config = load_config()
#model = load_model(config['paths']['model'])
#tokenizer = load_tokenizer(config['paths']['tokenizer'])

labels = ['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']

def _comment_values(api_result, labels):
  comment_values = [round((api_result[label]*100),2) for label in labels ]
  return comment_values


@app.get("/")
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def prediction(request: Request, comment: str = Form(...)):
#    api_result = predict_comment(comment, tokenizer, model)
    api_result = {
      "comment":"Teste",
      "toxic":0.1,
      'severe_toxic':0.2,
      'obscene':0.4,
      'threat':0.7, 
      'insult':0.9, 
      'identity_hate':0.8
      }
    comment_values = _comment_values(api_result, labels)
    return templates.TemplateResponse("index.html", {"request": request, 
                                                     "category": labels,
                                                     "cat_values": comment_values})

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
  
