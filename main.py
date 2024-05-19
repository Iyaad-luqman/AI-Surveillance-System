from flask import Flask, request, render_template, url_for, redirect
from query_process import process_prompt
import os
from classify_video import classify_videos
import time
import json
import shutil

app = Flask(__name__)
app.debug = True
app.jinja_env.globals.update(zip=zip)  
@app.route("/")
def index():
  return render_template('index.html')

@app.route("/view_result")
def view_result():
      titles = request.args.getlist('titles')
      time_frames = request.args.getlist('time_frames')
      dir_name = request.args.get('dir_name')
      # titles = ['car accident','car accident']
      # time_frames = ['00:00:01:000-00:00:05:000','00:00:10:000-00:00:14:000']
      # dir_name = 'testing-data'
      return render_template('results.html', time_frames=time_frames, titles=titles, dir_name=dir_name)

def create_directory(base_name):
  counter = 1
  dir_name = base_name
  while os.path.exists('static/run-test/'+dir_name): 
    dir_name = f"{base_name}-{counter}"
    counter += 1
  os.makedirs('static/run-test/'+dir_name)
  return dir_name

def create_directory_for_save(base_name):
  counter = 1
  dir_name = base_name
  while os.path.exists('static/saved-test/'+dir_name): 
    dir_name = f"{base_name}-{counter}"
    counter += 1
  os.makedirs('static/saved-test/'+dir_name)
  return dir_name


@app.route("/save-analysis", methods=["POST"])
def save_analysis():
  dir_name = request.form.get('name')
  titles = json.loads(request.form.get('titles'))  # convert JSON string back to list
  time_frames = json.loads(request.form.get('time_frames'))  # convert JSON string back to list

  data = {
    'titles': titles,
    'time_frames': time_frames,
  }
  dir_name = create_directory_for_save(dir_name)
  with open(f'static/saved-test/{dir_name}/content.json', 'w') as f:
    json.dump(data, f)
  src_dir = 'static/run-test/'+dir_name
  dst_dir = 'static/saved-test/'+dir_name
  files = os.listdir(src_dir)
  for file in files:
    shutil.copy(os.path.join(src_dir, file), dst_dir)
  return 'Success', 200


@app.route("/analyze", methods=["POST"])
def analyze():
    
    prompt = request.form.get('prompt')
    test_name = request.form.get('test_name')
    video_file = request.files['videoFile']
    removeDuplicates = request.form.get('removeDuplicates')
    if not prompt or not test_name or not video_file or not removeDuplicates:
          time.sleep(2)
          return redirect(url_for('index'))
    if removeDuplicates == 'on':
      removeDuplicates = True
    else:
      removeDuplicates = False
    dir_name = create_directory(test_name)
    video_file.save(f"static/run-test/{dir_name}/original.mp4")
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
    json_response = process_prompt(prompt, model_name = "gemini", json_schema = json_schema)
    json_data = json.loads(json_response)
    print(json_response)
    categories_list = json_data.get('categories', [])
    type_list = json_data.get('type', [])
    result = classify_videos(f"static/run-test/{dir_name}/original.mp4", categories_list, type_list, dir_name, remove_duplicate_frames=removeDuplicates)
    time_frames = list(result.keys())
    titles = list(result.values())
    return redirect(url_for('view_result', time_frames=time_frames, titles=titles, dir_name=dir_name))
    

if __name__ == "__main__":
    app.run()