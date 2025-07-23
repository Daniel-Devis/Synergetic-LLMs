import json

# Path to the original test JSONL file containing focal methods
test_jsonl_file = 'D:/danie/Documents/Disso/test methods 1/prepared_test_data.jsonl'

# Path to save the new JSONL file with integrated prompts
output_jsonl_file = 'D:/danie/Documents/Disso/integrated_prompts.jsonl'

# Template for the prompt
prompt_template = (
    "You are an expert Java software developer focused on generating accurate, logical and functional test cases.\n\n"
    "Here is a Java method for which you need to generate comprehensive and accurate functional test cases:\n\n"
    "{focal_method}\n\n"
    "Please consider the following when generating the test cases:\n\n"
    "1. **Positive Scenarios**: Test the method with typical valid inputs to ensure it behaves as expected.\n"
    "2. **Negative Scenarios**: Include test cases that handle invalid inputs and check for appropriate error handling or exceptions.\n"
    "3. **Edge Cases**: Identify and include test cases for boundary conditions, such as minimum, maximum, and null values.\n"
    "4. **Exception Handling**: Ensure that any expected exceptions are tested, confirming that the method fails gracefully when needed.\n"
    "5. **Complex Input Combinations**: If the method accepts multiple parameters, test different combinations of valid and invalid inputs.\n"
    "6. **Output Verification**: For each test case, specify the expected output and ensure it matches the actual result.\n\n"
    "Test cases:\n"
)

# Create a new JSONL file with the integrated prompts
with open(test_jsonl_file, 'r', encoding='utf-8') as f_in, open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        json_data = json.loads(line)
        focal_method = json_data['focal_method']
        prompt = prompt_template.format(focal_method=focal_method)
        
        # Create the JSON object with the role and content structure
        prompt_json = {
            "messages": [
                {"role": "system", "content": "You are an expert Java software developer focused on generating accurate, logical and functional test cases."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Write each integrated prompt as a separate JSON object in the JSONL file
        f_out.write(json.dumps(prompt_json) + '\n')

print(f"Integrated prompts saved to {output_jsonl_file}")
