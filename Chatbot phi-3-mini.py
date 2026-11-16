#!/usr/bin/env python3
"""
Local chatbot with Microsoft Phi-3-mini (microsoft/Phi-3-mini-4k-instruct) in a tkinter window.

Prerequisites:
    pip install torch transformers accelerate

Remarks:
- On first launch, the model (~7.6 GB in bf16) AND the tokenizer are downloaded
  from Hugging Face and saved locally (./phi-3-mini-local), so subsequent
  launches work fully offline.
- v4 : migrated from Phi-2 to Phi-3-mini (native chat template),
  offline caching of model + tokenizer, English UI strings,
  stop-list trimmed to the ChatML markers Phi-3-mini actually emits.
- v5 : removed trust_remote_code=True — transformers >= 4.39 (and especially
  5.x) ships a native Phi-3 implementation, and the old remote code
  (modeling_phi3.py) breaks on v5 with a confusing KeyError: 'type'.
"""

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
LOCAL_DIR = "./phi-3-mini-local"   # local copy used for offline operation
MAX_NEW_TOKENS = 130

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in the same language as the user, in a few sentences, then stop. Do not invent exercises or puzzles."
)

# ChatML markers emitted by Phi-3-mini when it rambles past its answer
STOP_STRINGS = [
    "<|end|>",
    "User:",
    "Assistant:",
]


class StopOnStrings(StoppingCriteria):
    """Stop generation as soon as a forbidden string appears in the GENERATED
    tokens (the prompt is excluded via prompt_len). Note: decoding must keep
    special tokens visible, otherwise '<|end|>' can never be detected."""

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


class ChatbotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Phi-3-mini Chatbot")
        root.geometry("600x600")

        self.history: list[tuple[str, str]] = []  # ("user" | "assistant", text)

        # --- Conversation area ---
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", font=("Segoe UI", 11)
        )
        self.chat_area.pack(padx=10, pady=(10, 5), fill=tk.BOTH, expand=True)

        # --- Input frame ---
        bottom = tk.Frame(root)
        bottom.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry = tk.Entry(bottom, font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.entry.bind("<Return>", self.on_send)

        self.send_button = tk.Button(
            bottom, text="Send", width=12, command=self.on_send
        )
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))

        # --- Status bar ---
        self.status = tk.Label(root, text="Loading model…", fg="gray", anchor="w")
        self.status.pack(padx=10, pady=(0, 5), fill=tk.X)

        # Load the model in the background so the UI stays responsive
        self.model = None
        self.tokenizer = None
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            # Offline support: if a local copy already exists, load it;
            # otherwise download once and save BOTH model and tokenizer.
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
                # Persist for offline use on the next launches
                self.model.save_pretrained(LOCAL_DIR)
                self.tokenizer.save_pretrained(LOCAL_DIR)

            self.model.eval()
            self.set_status("Model ready.")
            self.entry.configure(state=tk.NORMAL)
            self.entry.focus_set()
        except Exception as e:
            self.set_status(f"Failed to load model: {e}")

    def set_status(self, text: str):
        # tkinter must be touched from the main thread only
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
        """Build the chat messages for Phi-3-mini (chat template)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in self.history[-6:]:  # keep the last 6 turns
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

            answer = self.tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

            # Safety net: re-cut on markers in case the stop criterion let
            # one slip through (e.g. string split across the decode window).
            for marker in STOP_STRINGS:
                idx = answer.find(marker)
                if idx != -1:
                    answer = answer[:idx].strip()

            if not answer:
                answer = "(empty response)"
        except Exception as e:
            answer = f"Error: {e}"

        # Back to the main thread for display
        self.root.after(0, lambda: self.show_answer(answer))

    def show_answer(self, answer: str):
        self.append_message("Assistant", answer)
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
