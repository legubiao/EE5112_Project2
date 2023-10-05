import tkinter as tk
import os
import time
import llama_cpp
from llama_cpp import Llama

# Main Window
root = tk.Tk()

# Setting Item
n_batch = tk.StringVar()  # Total Dialogue Limit
n_batch.set("512")

max_tokens = tk.StringVar()  # Single Response Limit
max_tokens.set("200")

LLM_role = tk.StringVar()  # Role of LLM
LLM_role.set("You are a helpful assistant.")

model_folder = './models'
model_file = tk.StringVar()  # Model File Path
files = []
for f in os.listdir(model_folder):
    if f.endswith('.gguf'):
        files.append(f)
model_file.set(files[0])

# LLM and Conversation Initialization
llm = Llama(model_path="./models/" + model_file.get(), n_gpu_layers=1, verbose=False, n_batch=int(n_batch.get()))
messages = [llama_cpp.ChatCompletionMessage(content=LLM_role.get(), role="system")]

root.title("Chat with Llama")
root.geometry("600x400+100+100")

# Chat Messages Frame
frame_chat = tk.Frame(root)
frame_chat.pack(fill=tk.BOTH, expand=True)

text_chat = tk.Text(frame_chat)
text_chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_chat = tk.Scrollbar(frame_chat)
scrollbar_chat.pack(side=tk.RIGHT, fill=tk.Y)

text_chat.config(yscrollcommand=scrollbar_chat.set)
scrollbar_chat.config(command=text_chat.yview)

# Input Frame
frame_input = tk.Frame(root)
frame_input.pack(fill=tk.X)

entry_input = tk.Entry(frame_input)
entry_input.pack(side=tk.LEFT, fill=tk.X, expand=True)


def open_settings():
    settings = tk.Toplevel(root)
    settings.title("Settings")
    settings.geometry("400x300")

    # LLM Role
    global LLM_role

    llm_role_frame = tk.Frame(settings)
    llm_role_frame.pack()
    tk.Label(llm_role_frame, text="LLM Role").pack(side=tk.LEFT)
    tk.Entry(llm_role_frame, textvariable=LLM_role).pack(side=tk.RIGHT)

    # Model File
    global model_folder
    global files

    files = []
    for f1 in os.listdir(model_folder):
        if f1.endswith('.gguf'):
            files.append(f1)

    model_select = tk.Frame(settings)
    model_select.pack()
    tk.Label(model_select, text="Model Files").pack(side=tk.LEFT)

    model_option_menu = tk.OptionMenu(model_select, model_file, *files)
    model_option_menu.pack(side=tk.RIGHT)

    # Total Dialogue Length
    global n_batch
    n_batch_input = tk.Frame(settings)
    n_batch_input.pack()

    tk.Spinbox(n_batch_input, from_=0, to=2048, increment=256,
               textvariable=n_batch).pack(side=tk.RIGHT)
    tk.Label(n_batch_input, text="Total Dialogue Length").pack(side=tk.LEFT)

    # Single Response Length
    global max_tokens
    max_tokens_input = tk.Frame(settings)
    max_tokens_input.pack()
    tk.Spinbox(max_tokens_input, from_=0, to=200, increment=10,
               textvariable=max_tokens).pack(side=tk.RIGHT)
    tk.Label(max_tokens_input, text="Single Response Length").pack(side=tk.LEFT)

    def save():
        global n_batch
        global llm

        llm = Llama(model_path="./models/" + model_file.get(), n_gpu_layers=1, verbose=False,
                    n_batch=int(n_batch.get()))

        clear_message()
        settings.destroy()

    bottom_frame = tk.Frame(settings)
    bottom_frame.pack(side=tk.BOTTOM)
    tk.Button(bottom_frame, text="Save and Exit", command=save).pack(side=tk.RIGHT)
    tk.Button(bottom_frame, text="Exit", command=settings.destroy).pack(side=tk.RIGHT)


button_setting = tk.Button(frame_input, text="Settings", command=open_settings)
button_setting.pack(side=tk.RIGHT)


def clear_message():
    global messages
    global LLM_role
    text_chat.delete("1.0", "end")
    messages = [llama_cpp.ChatCompletionMessage(content=LLM_role.get(), role="system")]
    text_chat.insert(tk.END, "LLAMA: Hello \n")


button_clear = tk.Button(frame_input, text="Reset", command=clear_message)
button_clear.pack(side=tk.RIGHT)


def send_message():
    # 获取输入框的内容
    message = entry_input.get()
    # 如果内容不为空，就在聊天记录中显示，并清空输入框
    if message:
        text_chat.insert(tk.END, "You: " + message + "\n")
        entry_input.delete(0, tk.END)
        time.sleep(1)

        messages.append(llama_cpp.ChatCompletionMessage(content=message, role="user"))
        root.update()
        # 模拟聊天机器人的回复延迟
        role = ""
        reply = ""
        text_chat.insert(tk.END, "LLAMA: ")
        for result in llm.create_chat_completion(messages, stream=True, max_tokens=int(max_tokens.get())):
            if "role" in result["choices"][0]['delta'].keys():
                role = result["choices"][0]['delta']["role"]
            elif "content" in result["choices"][0]['delta'].keys():
                word = result["choices"][0]['delta']["content"]
                reply += word
                text_chat.insert(tk.END, word)
                root.update()
        messages.append(llama_cpp.ChatCompletionMessage(content=reply, role=role))
        # 在聊天记录中显示回复内容
        text_chat.insert(tk.END, "\n")
        text_chat.insert(tk.END, "------------- *** --------------\n")


button_send = tk.Button(frame_input, text="Send", command=send_message)
button_send.pack(side=tk.RIGHT)

# Say Hello
text_chat.insert(tk.END, "LLAMA: Hello \n")

root.mainloop()
