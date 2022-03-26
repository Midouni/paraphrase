from flask import Flask, render_template, make_response
from flask_restful import Api, Resource, reqparse 
from models.model import gpt2_paraphraser, t5_paraphraser

app=Flask(__name__)
api=Api(app)

data=reqparse.RequestParser()
data.add_argument("model",type=str,help="the type of model (T5 or GPT2",required=True)
data.add_argument("text",type=str,help="the text",required=True)

class paraphraser(Resource):
    def get(self):
        args=data.parse_args()
        if args["model"]=="GPT2":
            return {"text":gpt2_paraphraser(args["text"])}
        else:
            return {"text":t5_paraphraser(args["text"])}

class homePage(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers)


api.add_resource(paraphraser,"/paraphrase")
api.add_resource(homePage,"/")

#run the server
if __name__== "__main__":
    app.run(debug=True,host='0.0.0.0')