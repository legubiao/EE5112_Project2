# 导入tkinter模块
import this
import tkinter as tk
# 导入time模块
import time
import llama_cpp
from llama_cpp import Llama

# 创建主窗口对象
root = tk.Tk()
n_batch = 512  # 总对话长度限制
max_tokens = 256  # 单次回复最大长度限制

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1, verbose=False, n_batch=n_batch)
messages = [llama_cpp.ChatCompletionMessage(content="You are a helpful assistant.", role="system")]

# 设置窗口标题
root.title("Chat with Llama")

# 设置窗口大小和位置
root.geometry("600x400+100+100")

# 创建一个Frame容器，用于放置聊天记录
frame_chat = tk.Frame(root)
frame_chat.pack(fill=tk.BOTH, expand=True)

# 创建一个Text组件，用于显示聊天记录
text_chat = tk.Text(frame_chat)
text_chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 创建一个Scrollbar组件，用于滚动聊天记录
scrollbar_chat = tk.Scrollbar(frame_chat)
scrollbar_chat.pack(side=tk.RIGHT, fill=tk.Y)

# 关联Text和Scrollbar
text_chat.config(yscrollcommand=scrollbar_chat.set)
scrollbar_chat.config(command=text_chat.yview)

# 创建一个Frame容器，用于放置输入框和发送按钮
frame_input = tk.Frame(root)
frame_input.pack(fill=tk.X)

# 创建一个Entry组件，用于输入聊天内容
entry_input = tk.Entry(frame_input)
entry_input.pack(side=tk.LEFT, fill=tk.X, expand=True)


def open_settings():
    global n_batch
    global max_tokens

    # 创建一个新的窗口对象
    settings = tk.Toplevel(root)
    # 设置窗口标题
    settings.title("设置")
    # 设置窗口大小
    settings.geometry("300x200")
    # 在窗口中添加一些控件
    tk.Label(settings, text="这是一个设置窗口").pack()
    tk.Checkbutton(settings, text="启用声音").pack()
    tk.Radiobutton(settings, text="选择模式：平衡", value=1).pack()
    tk.Radiobutton(settings, text="选择模式：创造", value=2).pack()
    tk.Radiobutton(settings, text="选择模式：精确", value=3).pack()

    # 总对话长度设置
    n_batch_input = tk.Frame(settings)
    n_batch_input.pack()
    n_batch_spin = tk.Spinbox(n_batch_input, from_=0, to=2048, increment=256,
                              textvariable=tk.StringVar(value=str(n_batch)))
    n_batch_spin.pack(side=tk.RIGHT)
    tk.Label(n_batch_input, text="总对话长度").pack(side=tk.LEFT)

    max_tokens_input = tk.Frame(settings)
    max_tokens_input.pack()
    max_tokens_spin = tk.Spinbox(max_tokens_input, from_=0, to=200, increment=10,
                                 textvariable=tk.StringVar(value=str(max_tokens)))
    max_tokens_spin.pack(side=tk.RIGHT)
    tk.Label(max_tokens_input, text="单次回复长度").pack(side=tk.LEFT)

    def save():
        global n_batch
        global max_tokens
        global llm

        n_batch = int(n_batch_spin.get())
        max_tokens = int(max_tokens_spin.get())
        llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf", n_gpu_layers=1, verbose=False,
                    n_batch=n_batch)
        clear_message()
        settings.destroy()

    tk.Button(settings, text="保存并关闭", command=save).pack()


button_setting = tk.Button(frame_input, text="打开设置", command=open_settings)
button_setting.pack(side=tk.RIGHT)


def clear_message():
    global messages
    text_chat.delete("1.0", "end")
    messages = [llama_cpp.ChatCompletionMessage(content="You are a helpful assistant.", role="system")]
    text_chat.insert(tk.END, "LLAMA: Hello \n")


button_clear = tk.Button(frame_input, text="重置", command=clear_message)
button_clear.pack(side=tk.RIGHT)


# 定义一个函数，用于发送聊天内容和获取回复内容
def send_message():
    # 获取输入框的内容
    message = entry_input.get()
    # 如果内容不为空，就在聊天记录中显示，并清空输入框
    if message:
        text_chat.insert(tk.END, "你: " + message + "\n")
        entry_input.delete(0, tk.END)
        time.sleep(1)

        messages.append(llama_cpp.ChatCompletionMessage(content=message, role="user"))
        root.update()
        # 模拟聊天机器人的回复延迟
        role = ""
        reply = ""
        text_chat.insert(tk.END, "LLAMA: ")
        for result in llm.create_chat_completion(messages, stream=True, max_tokens=max_tokens):
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


# 创建一个Button组件，用于发送聊天内容
button_send = tk.Button(frame_input, text="发送", command=send_message)
button_send.pack(side=tk.RIGHT)

# 在启动时让聊天机器人打招呼
text_chat.insert(tk.END, "LLAMA: Hello \n")

# 启动主循环
root.mainloop()
