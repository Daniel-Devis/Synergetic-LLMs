import json
from openai import OpenAI

# Set your OpenAI API key
api_key = ''  # Replace with your actual OpenAI API key
client = OpenAI(api_key=api_key)

# Path to the test JSONL file containing focal methods
test_jsonl_file = r'D:\danie\Documents\CSC-40040 19020322 code/extracted_focal_methods2.jsonl'

def collect_focal_methods(file_path):
    focal_methods = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                try:
                    json_data = json.loads(line.strip())
                    print(f"Line {i}: {json_data}")  # Print each line to inspect its contents
                    if 'focal_method' in json_data:
                        focal_methods.append(json_data['focal_method'])
                        print(f"Line {i}: Focal method collected")
                    else:
                        print(f"Line {i}: 'focal_method' key not found")
                except json.JSONDecodeError as e:
                    print(f"Line {i}: JSON decode error: {e}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return focal_methods

# Use the function to collect test focal methods
test_focal_methods = collect_focal_methods(test_jsonl_file)
print(f"Collected {len(test_focal_methods)} focal methods.")

# Predictions storage
predictions = []

# Generate test cases for each focal method using the fine-tuned model
for focal_method in test_focal_methods:
    response = client.chat.completions.create(
      model='ft:gpt-4.1-mini-2025-04-14:personal::CELffg9H',  # Replace with your actual fine-tuned model ID
      messages=[
        {"role": "system", "content": "You are an expert Java software developer focused on generating accurate, logical and functional test cases, Please consider the following when generating the test cases:\n\n1. **Positive Scenarios**: Test the method with typical valid inputs to ensure it behaves as expected.\n2. **Negative Scenarios**: Include test cases that handle invalid inputs and check for appropriate error handling or exceptions.\n3. **Edge Cases**: Identify and include test cases for boundary conditions, such as minimum, maximum, and null values.\n4. **Exception Handling**: Ensure that any expected exceptions are tested, confirming that the method fails gracefully when needed.\n5. **Complex Input Combinations**: If the method accepts multiple parameters, test different combinations of valid and invalid inputs.\n6. **Output Verification**: For each test case, specify the expected output and ensure it matches the actual result.\n"},
        {"role": "user", "content": f"Here is a Java method for which you need to generate comprehensive and accurate functional test cases:\n\n{focal_method}\n"}
      ],
    )
    print(f"Generated response for focal method:\n{focal_method}\nResponse: {response}")
    predictions.append({"focal_method": focal_method, "predicted_test_cases": response.choices[0].message.content.strip()})

# Save predictions to a JSON file
output_file = 'D:/danie/Documents/Disso/4.1minipredictions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=4)

print(f"Predictions saved to {output_file}")
