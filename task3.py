import json
import random
from gpt4all import GPT4All
import string

# Load the validation dataset of the BoolQ dataset from a local file
# local_path = "path/to/your/local/dev.jsonl"
# Read the JSON file and convert it into a list where each element is a dictionary
with open('dev.jsonl', "r", encoding="utf-8") as json_file:
    data_list = [json.loads(line) for line in json_file]

# Randomly select 500 samples using the last three digits of student ID as the random seed
random_seed = 328  # random seed
random.seed(random_seed)
selected_samples = random.sample(data_list, 500)

# Initialize the LLM
model = GPT4All(model_name='wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0')

# Dialogue accuracy
correct_predictions = 0

# Read the samples
for i, sample in enumerate(selected_samples, 1):
    question = sample["question"]
    passage = sample["passage"]
    expected_answer = sample["answer"]
    title = sample["title"]

    print("waiting for response...")

    # Generate answers, combing sample information
    with model.chat_session():
        prompt = "According to the paasage: "+ passage + " About "+ title + ", " + question + "? Just answer yes or no."
        # print('question:', prompt)
        response = model.generate(prompt = prompt, max_tokens=2)

    # Remove punctuation, keeping only the characters
    generated_answer = response.strip().replace("###User:", "")
    generated_answer = generated_answer.translate(str.maketrans('', '', string.punctuation))
    print('answer:', generated_answer)
    expected_answer = str(expected_answer).translate(str.maketrans('', '', string.punctuation))

    # Map the generated answers to "true" or "false"
    if generated_answer.lower() == "yes":
        generated_answer = "true"
    elif generated_answer.lower() == "no":
        generated_answer = "false"

    # Compare the generated answers with the expected answers
    if generated_answer.lower() == expected_answer.lower():
        correct_predictions += 1
        print("Answer: Correct")
    else:
        print("Answer: Incorrect")

    # Output the current number of processed samples
    print(f"Processed {i} samples. Correct samples so far: {correct_predictions}")

# Output the accuracy
overall_accuracy = correct_predictions / len(selected_samples)
print("Overall Accuracy:", overall_accuracy)
