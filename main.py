from flask import Flask, request
from query_process import process_prompt
app = Flask(__name__)
from classify_video import classify_videos
import json

# @app.route("/api/nlp/process")
# def nlp_process():
#     data = request.args.get('data')
#     json_schema = '''{
#   "type": "object",
#   "properties": {
#     "categories": {
#       "type": "array",
#       "items": {
#         "type": "string"
#       }
#     },
#     "true_case": {
#       "type": "array",
#       "items": {
#         "type": "string"
#       }
#     }
#   },
#   "required": ["categories", "true_case"]
# }'''
#     return process_prompt(data, model_name = "gemini", json_schema = json_schema)


def nlp_process():
    data = "Is there any car crash in the video ?"
    json_schema = '''{
  "type": "object",
  "properties": {
    "categories": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "type": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["categories", "type"]
}'''
    return process_prompt(data, model_name = "gemini", json_schema = json_schema)


json_response = nlp_process()
json_data = json.loads(json_response)
print(json_response)
categories_list = json_data.get('categories', [])
type_list = json_data.get('type', [])
classify_videos("testing-data/accident.mp4", categories_list, type_list)