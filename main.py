from flask import Flask, request, render_template, url_for, redirect, send_from_directory
from query_process import process_prompt, opposite_process_prompt
import os
from classify_video import classify_videos
import time
import json
import ast
import shutil
import html
import yaml
from false_positive import report_false_positive



app = Flask(__name__)
app.debug = True
app.jinja_env.globals.update(zip=zip)  

def load_settings():
    with open("settings.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
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

@app.route("/saved_analysis")
def saved_analysis():
  saved_tests = os.listdir('static/saved-test')
  return render_template('saved_analysis.html', saved_tests=saved_tests)

@app.route("/false_positive", methods=["POST"])
def false_positive():
  dir_name= request.form.get('dirName')
  index = request.form.get('index')
  title = request.form.get('title')
  title = title.replace(' ', '_')
  file_name = 'static/run-test/'+dir_name+'/'+title+'-'+index+'.mp4'
  report_false_positive(file_name, title)

  # Open the file in append mode and write the file_name to it
  with open('static/run-test/'+dir_name+'/false_positive.txt', 'a') as f:
    f.write(file_name + '\n')

  return 'Success', 200




@app.route("/view_saved_test/<dir_name>")
def view_saved_test(dir_name):
  with open(f'static/saved-test/{dir_name}/content.json', 'r') as f:
    data = json.load(f)
  titles = ast.literal_eval(html.unescape(data['titles']))
  time_frames = ast.literal_eval(html.unescape(data['time_frames']))
  return render_template('saved_results.html',  time_frames=time_frames, titles=titles, dir_name=dir_name)

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

    settings = load_settings()
    model_name = settings['model']['name']
    device = settings['model']['device']
    json_response = process_prompt(prompt, model_name = model_name, json_schema = json_schema)
    json_data = json.loads(json_response)
    print(json_response)
    
    
    categories_list = json_data.get('categories', [])
    type_list = json_data.get('type', [])
    result = classify_videos(f"static/run-test/{dir_name}/original.mp4", categories_list, type_list, dir_name, remove_duplicate_frames=removeDuplicates, device=device)
    time_frames = list(result.keys())
    titles = list(result.values())
    return redirect(url_for('view_result', time_frames=time_frames, titles=titles, dir_name=dir_name))
    

if __name__ == "__main__":
    app.run()