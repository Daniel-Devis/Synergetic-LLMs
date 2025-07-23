import json
from openai import OpenAI

# Set your OpenAI API key
api_key = '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'  # Replace with your actual OpenAI API key
client = OpenAI(api_key=api_key)

# Path to the JSON file containing focal methods and generated test cases
input_json_file = 'D:/danie/Documents/Disso/data/predictions.json'
output_json_file = 'D:/danie/Documents/Disso/3.5reviewed_test_cases.json'

def load_test_case_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load the test case data
test_case_data = load_test_case_data(input_json_file)

# Storage for reviewed test cases
reviewed_cases = []

# Review and critique each test case
for entry in test_case_data:
    focal_method = entry['focal_method']
    predicted_test_case = entry['predicted_test_cases']

    # Create a review prompt
    review_prompt = (
        f"Here is a method for which a test case has been generated:\n\n"
        f"{focal_method}\n\n"
        f"Generated test case:\n\n"
        f"{predicted_test_case}\n\n"
        f"consider the following when reviewing the test cases:\n\n1. **Positive Scenarios**: Test the method with typical valid inputs to ensure it behaves as expected.\n2. **Negative Scenarios**: Include test cases that handle invalid inputs and check for appropriate error handling or exceptions.\n3. **Edge Cases**: Identify and include test cases for boundary conditions, such as minimum, maximum, and null values.\n4. **Exception Handling**: Ensure that any expected exceptions are tested, confirming that the method fails gracefully when needed.\n5. **Complex Input Combinations**: If the method accepts multiple parameters, test different combinations of valid and invalid inputs.\n6. **Output Verification**: For each test case, specify the expected output and ensure it matches the actual resul. "
        f"Provide improved test cases with increased functionality and correctness.\n"
    )

    response = client.chat.completions.create(
        model='ft:gpt-3.5-turbo-0125:personal::9wAFewSr',  # Ensure this is the correct model ID for your review model
        messages=[
            {"role": "system", "content": "You are a software testing expert."},
            {"role": "user", "content": review_prompt}
        ],
        max_tokens=4000
    )

    review_feedback = response.choices[0].message.content.strip()
    reviewed_cases.append({
        "focal_method": focal_method,
        "predicted_test_cases": predicted_test_case,
        "review_feedback": review_feedback
    })

    print(f"Reviewed test case for focal method:\n{focal_method}\nFeedback: {review_feedback}")

# Save reviewed test cases to a JSON file
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(reviewed_cases, f, indent=4)

print(f"Reviewed test cases saved to {output_json_file}")
