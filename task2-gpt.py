from gpt4all import GPT4All
import time

# Initialize the LLM
model = GPT4All(model_name='wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0')

print("Welcome to the Dialogue System. Type 'exit' to end the conversation.")

# Continuous chat conversation
with model.chat_session():
    # Looping conversation
    while True:
        user_input = input("Say: ")
        if user_input.lower() == 'exit':
            print("System Exit. See you next time.")
            break

        # Calculate the processing time
        start_time = time.time()
        # Generate response
        response = model.generate(prompt=user_input, top_k=1)
        response = response.strip().replace("###User:", "") # Response format optimization

        end_time = time.time()
        processing_time = end_time - start_time

        print("Response:", response)
        print("Processing Time: {:.2f} seconds".format(processing_time))
