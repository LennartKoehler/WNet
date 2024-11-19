import json


# Write the parameters
def save_run_parameters(params, dir_path):
    file_path = f"{dir_path}parameters.json"
    try:
        # Write back to the file
        with open(file_path, 'w') as file:
            json.dump(params, file, indent=4)
        print("Run parameters saved successfully.")
    except Exception as e:
        print(f"Error saving parameters: {e}")

