#!/usr/bin/env python3

import os
import threading
import tkinter as tk
from tkinter import scrolledtext

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_DIR = "./phi-3-mini-local"
MAX_NEW_TOKENS = 170

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in the same language as the user, in a few sentences, then stop. Do not invent exercises or puzzles."
)

STOP_STRINGS = [
    "<|end|>",
    "User:",
    "Assistant:",
]


class StopOnStrings(StoppingCriteria):

    def __init__(self, tokenizer, stop_strings, prompt_len, window=64):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_len = prompt_len
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_len:]
        tail = self.tokenizer.decode(
            generated[-self.window:], skip_special_tokens=False
        )
        return any(s in tail for s in self.stop_strings)


def trim_to_last_sentence(text):
    # If the answer was cut mid-sentence, cut it back to the last
    # complete sentence (ends with . ! ? or …).
    best = -1
    for punct in (".", "!", "?", "…"):
        idx = text.rfind(punct)
        if idx > best:
            best = idx
    if best != -1 and best > len(text) // 2:
        return text[: best + 1]
    # No usable sentence end (or trimming would remove too much): keep all
    return text


class ChatbotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Phi-3-mini Chatbot")
        root.geometry("600x600")

        self.history: list[tuple[str, str]] = []

        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", font=("Segoe UI", 11)
        )
        self.chat_area.pack(padx=10, pady=(10, 5), fill=tk.BOTH, expand=True)

        bottom = tk.Frame(root)
        bottom.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry = tk.Entry(bottom, font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.entry.bind("<Return>", self.on_send)

        self.send_button = tk.Button(
            bottom, text="Send", width=12, command=self.on_send
        )
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))

        self.status = tk.Label(root, text="Loading model…", fg="gray", anchor="w")
        self.status.pack(padx=10, pady=(0, 5), fill=tk.X)

        self.model = None
        self.tokenizer = None
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            if os.path.isfile(os.path.join(LOCAL_DIR, "config.json")):
                model_path = LOCAL_DIR
            else:
                model_path = HF_MODEL_NAME
                os.makedirs(LOCAL_DIR, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )

            if model_path == HF_MODEL_NAME:
                self.model.save_pretrained(LOCAL_DIR)
                self.tokenizer.save_pretrained(LOCAL_DIR)

            self.model.eval()
            self.set_status("Model ready.")
            self.entry.configure(state=tk.NORMAL)
            self.entry.focus_set()
        except Exception as e:
            self.set_status(f"Failed to load model: {e}")

    def set_status(self, text: str):
        self.root.after(0, lambda: self.status.configure(text=text))

    def append_message(self, role: str, text: str):
        self.chat_area.configure(state="normal")
        tag = "user" if role == "You" else "bot"
        self.chat_area.tag_configure("user", foreground="#6d4aff")
        self.chat_area.tag_configure("bot", foreground="#333333")
        self.chat_area.insert(tk.END, f"{role}: ", tag)
        self.chat_area.insert(tk.END, f"{text}\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.configure(state="disabled")

    def build_messages(self, question: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in self.history[-6:]:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})
        return messages

    def generate(self, question: str):
        self.append_message("You", question)
        self.history.append(("user", question))
        self.set_status("Generating…")

        try:
            messages = self.build_messages(question)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]

            end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
            stopping = StoppingCriteriaList(
                [StopOnStrings(self.tokenizer, STOP_STRINGS, prompt_len)]
            )

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    stopping_criteria=stopping,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    eos_token_id=[self.tokenizer.eos_token_id, end_id],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_len = output.shape[1] - prompt_len
            answer = self.tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

            for marker in STOP_STRINGS:
                idx = answer.find(marker)
                if idx != -1:
                    answer = answer[:idx].strip()

            # If the answer hit the token limit, it was cut mid-sentence:
            # trim it back to the last complete sentence.
            if generated_len >= MAX_NEW_TOKENS:
                answer = trim_to_last_sentence(answer)

            if not answer:
                answer = "(empty response)"
        except Exception as e:
            answer = f"Error: {e}"

        self.root.after(0, lambda: self.show_answer(answer))

    def show_answer(self, answer: str):
        self.append_message("Phi-3-mini", answer)
        self.history.append(("assistant", answer))
        self.set_status("Model ready.")
        self.send_button.configure(state=tk.NORMAL)
        self.entry.configure(state=tk.NORMAL)
        self.entry.focus_set()

    def on_send(self, event=None):
        question = self.entry.get().strip()
        if not question or self.model is None:
            return
        self.entry.delete(0, tk.END)
        self.entry.configure(state="disabled")
        self.send_button.configure(state=tk.DISABLED)
        threading.Thread(target=self.generate, args=(question,), daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
