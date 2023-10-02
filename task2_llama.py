import llama_cpp
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1, verbose=False)
messages = [llama_cpp.ChatCompletionMessage(content="You are a drunk assistant.", role="system")]

while True:
    i = input('>>> ').strip()
    if i != 'exit':
        messages.append(llama_cpp.ChatCompletionMessage(content=i, role="user"))
        print("Generating Response ... ")
        role = ""
        reply = ""
        for result in llm.create_chat_completion(messages, stream=True):
            if "role" in result["choices"][0]['delta'].keys():
                role = result["choices"][0]['delta']["role"]
            elif "content" in result["choices"][0]['delta'].keys():
                word = result["choices"][0]['delta']["content"]
                reply += word
                print(word, end='')
        messages.append(llama_cpp.ChatCompletionMessage(content=reply, role="assistant"))
        print('')
    else:
        break
