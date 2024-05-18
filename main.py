from flask import Flask, request, render_template
from query_process import process_prompt

from classify_video import classify_videos
import json

app = Flask(__name__)
@app.route("/")
def index():
  return render_template('index.html')

@app.route("/result")
def result():
  return render_template('results.html')


@app.route("/api/nlp/process")
def nlp_process():
    data = request.args.get('data')
    test_name = request.args.get('test_name')
    json_schema = '''{
  "type": "object",
  "properties": {
    "categories": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "true_case": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["categories", "true_case"]
}'''
    json_response = nlp_process()
    json_data = json.loads(json_response)
    print(json_response)
    categories_list = json_data.get('categories', [])
    type_list = json_data.get('type', [])
    classify_videos("testing-data/accident.mp4", categories_list, type_list)
    return process_prompt(data, model_name = "gemini", json_schema = json_schema)

if __name__ == "__main__":
    app.run()