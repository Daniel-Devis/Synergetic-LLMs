from openai import OpenAI
import openai
import os
import time


# Set your OpenAI API key
api_key = ''
Client = OpenAI(api_key=api_key)
training_file_path = r'D:\danie\Documents\Disso\data\prepared_data.jsonl'

# Ensure the file exists
if not os.path.exists(training_file_path):
    raise FileNotFoundError(f"Training file not found at {training_file_path}")

with open(training_file_path, "rb") as file:
    training_file = Client.files.create(
        file=file,
        purpose='fine-tune'
    )

training_file_id = training_file.id
print(f"Training file uploaded with ID: {training_file_id}")

hyperparameters = {
    "batch_size": "auto",
    "learning_rate_multiplier": "auto",
    "n_epochs": 15  # Specify the number of epochs
}

response = Client.fine_tuning.jobs.create(
  model='gpt-3.5-turbo',  # You can choose a different base model if needed
  training_file=training_file_id,
  hyperparameters=hyperparameters
)

job_id = response.id
print(f"Fine-tuning job created with ID: {job_id}")


while True:
    status = Client.fine_tuning.jobs.retrieve(job_id)
    if status.status == 'succeeded':
        fine_tuned_model_id = status.fine_tuned_model
        print(f"Fine-tuning succeeded. Model ID: {fine_tuned_model_id}")
        break
    elif status.status == 'failed':
        raise Exception("Fine-tuning failed")
    else:
        print(f"Fine-tuning status: {status.status}. Checking again in 60 seconds...")
        time.sleep(60)
