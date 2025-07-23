import json
import os

# Path to your raw data directory
raw_data_directory = 'D:/danie/Documents/1632 tests'

# Output path for the prepared JSONL file
prepared_data_file = 'prepared_data.jsonl'

def collect_json_data(directory):
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if 'focal_method' in json_data and 'test_case' in json_data:
                        focal_method = json_data['focal_method']['body']
                        test_case = json_data['test_case']['body']
                        chat_formatted_data = {
                            "messages": [
                                {"role": "system", "content": "You are an expert Java software developer focused on generating accurate, logical and functional test cases."},
                                {"role": "user", "content": f"Analyze the following Java method, Break it down logically and generate corresponding accurtae and functional test cases:\n\n{focal_method}"},
                                {"role": "assistant", "content": f"Here is a logically derived functional test case for the provided Java method:\n\n{test_case}"}
                            ]
                        }
                        data.append(chat_formatted_data)
    return data

def prepare_data_for_finetuning(data, output_file):
    # Check if the directory part of the output_file path exists, if not, create it
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
# Collect and prepare the data
collected_data = collect_json_data(raw_data_directory)
prepare_data_for_finetuning(collected_data, prepared_data_file)

print(f"Data prepared and saved to {prepared_data_file}")
