from aleph_vision_pipeline_2 import Pipeline

# Initialize the pipeline
pipeline = Pipeline()

# Define a test user message
user_message = "Summarize this text: Large language models are transforming AI."

# Define a test model (Ensure it exists in `model_mapping`)
model_id = "1"  # This maps to `claude-3-7-sonnet-20250219`

# Define messages in the expected format
messages = [
    {"role": "user", "content": user_message}
]

# Define the request body
body = {
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": False  # Testing without streaming first
}

# Run the pipeline
response = pipeline.pipe(user_message, model_id, messages, body)

# Print the response
print(response)
