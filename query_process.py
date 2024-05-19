import json
from langchain_community.llms import Ollama
import llm_switcher
import jsonschema

def process_prompt(data, model_name=None, json_schema=None):
    if model_name is None:
        Exception("Model name is required.")
    if json_schema is None:
        Exception
    prompt = "The user will give a prompt asking question about a particular CCTV Footage. You should analyse what the user is asking for and then give me 2 list. One is 'categories' which would contain the possible incidents that could have happened, for example 'theft', 'car accident', 'explosion' , 'chain snatching', etc. It should also Include opposite cases for that like 'cars passing', 'people walking' or something related to the user's use case inside the 'categories'. Then, the other list called 'type' which contain if the incident is violent, put 'True' or opposite of that. For example, 'fighting' is True and 'people walking' is False of that. The 'categories' and 'type' should be in the same order. The categories should contain spaces instead of underscores"
    
    prompt += "Based on the user input, you are to provide the json response based on the Provided JSON schema which is" + json_schema+  ". It should be properly divided into the necessary format. And the user input is "+  data + ".Only provide the JSON in the response and nothing else, not even GRAVE ACCENT."
    try:
        nlp_model = llm_switcher.get_nlp_model( model_name=model_name)
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM model: {e}")

    try:
        initial_prompt = prompt
        response =  nlp_model.process_text(initial_prompt)
        if is_valid_schema(response, json_schema):
            return response
        else:
            retry_prompt =  prompt +  ". I want you to provide only the JSON response in the proper format and not " + response
            response =  nlp_model.process_text(retry_prompt)
            if is_valid_schema(response, json_schema):
                return response
            else:
                raise ValueError("Response does not match the JSON schema after retry.")
    except Exception as e:
        return ("The prompt did not return a valid JSON. Please Debug to know more." + str(e) + str(response))  

def opposite_process_prompt(title, model_name=None):
    if model_name is None:
        Exception("Model name is required.")
    prompt = '''The user will give a prompt asking question about a particular CCTV Footage. You should analyse what the user is asking for and then give me 2 list. One is 'categories' which would contain the possible incidents that could have happened, for example 'theft', 'car accident', 'explosion' , 'chain snatching', etc. It should also Include opposite cases for that like 'cars passing', 'people walking' or something related to the user's use case inside the 'categories'. Then, the other list called 'type' which contain if the incident is violent, put 'True' or opposite of that. For example, 'fighting' is True and 'people walking' is False of that. The 'categories' and 'type' should be in the same order. Now considering this, TELL ME WHAT IS THE OPPOSITE OF ''' + title + '''. ONLY OUTPUT THAT WORD AND THAT WORD ALONE IN JSON FORMAT { opposite: "word"} . Only provide the JSON in the response and nothing else, not even GRAVE ACCENT. words should contain spaces instead of underscores'''
    
    
    try:
        nlp_model = llm_switcher.get_nlp_model( model_name=model_name)
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM model: {e}")

    initial_prompt = prompt
    response =  nlp_model.process_text(initial_prompt)
    print(response)
    return response

def is_valid_schema(data, schema):
    try:
        jsonschema.validate(instance=json.loads(data), schema=json.loads(schema))
        return True
    except jsonschema.exceptions.ValidationError as ve:
        return False
    except json.decoder.JSONDecodeError as je:
        raise ValueError(f"Invalid JSON format: {je}")
    except jsonschema.SchemaError as se:
        raise ValueError(f"Invalid JSON schema: {se}")