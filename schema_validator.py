import jsonschema
import json

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