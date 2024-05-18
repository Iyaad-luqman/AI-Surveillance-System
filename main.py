from flask import Flask, request
from query_process import process_prompt
app = Flask(__name__)

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
    data = "In the following footage, Did any car catch fire after crashing ?"
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
    return process_prompt(data, model_name = "gemini", json_schema = json_schema)

print(nlp_process())