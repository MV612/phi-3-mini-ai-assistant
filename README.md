# phi-3-mini-ai-assistant
An AI assistant using the model Microsoft phi-3-mini

This is an AI assistant coded with Python that uses the model Microsoft phi-3-mini.
It makes a first download that is XXX minuts long. After the downlaod, you can use it and it won't need to download each time you use it.
The Python script makes that everything needed is downloaded so that the AI is able to work offline.
You will have to use the AI assistant on a computer because it uses tkinter for the window, which doesn't work on phones.

# How to start ?
First, install a Python code interpreter on your computer.
Then, install the dependencies "accelerate", "torch" and "transformers".
After that, copy paste the script on your code interpreter and run the script.
Then, some red messages might appear in your console. Ignore them. If you see something like "Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]", this means the script is downloading.
A window might appear while the model is downloading. Ignore it because it won't be working at this moment.
Wait approximately XXX minuts that the model downloads itself. You will see the "Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]" message progressing after a while (the progress isn't on live, but there is only a visible progress when a complete file has been downloaded).
When the downloading is completed, you will see a message saying it in the console. You will also see "Model loaded" on the window.
Now, the model has been downloaded and you can use the AI assistant on the window.

# Strenghts
The script uses the model Microsoft phi-3-mini. This model is quite performent and is approximately at the same level than the model GPT-3.5 (the default model of ChatGPT until 2024).
The script is running locally so your data stays in majority private. This allows also a offline use of the AI assistant.
The script uses a downloaded model, in this case Microsoft phi-3-mini, which makes it mostly planet friendly, because it doen't need any data centers.
Let's compare its performance with GPT-3.5's :
                                                                                                                                                                            
| Benchmark                                  | Phi-3-mini (3.8B)	| GPT-3.5 Turbo	|                                                                                         
| MMLU (general knowledge, 5-shot)           |	 68.8	            | 71.4          |                                                                                         
| MT-bench (conversational quality)          |	8.38	            | 8.35	        |                                                                                         
| GSM-8K (math/reasoning, 8-shot CoT)        |  82.5	            | 78.1	        |                                                                                         
| HellaSwag (commonsense, 5-shot)	           | 76.7	              | 78.8	        |                                                                                         
| ANLI (adversarial reasoning, 7-shot)       | 52.8	              | 58.1	        |                                                                                         
| BigBench Hard (hard reasoning, 3-shot CoT) | 73.5	              | 68.3	        |                                                                                         
| AGI Eval (5-shot)	                         | 39.0	              | 48.4	        |                                                                                         
                                                                                                                                                                            
Sources:                                                                                                                                                                    
Phi-3 Technical Report (Abdin et al., Microsoft, April 2024), arxiv.org/abs/2404.14219 — the primary source. All benchmark numbers for both models come from its Figure/Table comparing the Phi-3 family against Mistral 7B, Gemma 7B, Llama-3 8B, Mixtral 8x7B and GPT-3.5-Turbo (version 1106), evaluated with the same pipeline. Full HTML version: arxiv.org/html/2404.14219v3
Microsoft / Phi-3-mini-4k-instruct model card on Hugging Face: huggingface.co/microsoft/Phi-3-mini-4k-instruct — reproduces the same table (MMLU 70.9 for the June 2024 updated checkpoint; the 68.8 figure is the original release), explicitly stating all reported numbers were produced with the exact same evaluation pipeline for comparability.
Microsoft Azure blog, "Introducing Phi-3: Redefining what's possible with SLMs" — secondary confirmation of the headline results.


# Attentions
Microsoft phi-3-mini is slow to answer. It is a model running locally and without any bigtech, so the speed si not like a real AI.
Microsoft phi-3-mini, like any AI, makes mistakes. Please check the model's answers.

