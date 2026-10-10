# Own AI: fine-tune on Kaggle, run locally
1. `python make_dataset.py` (optional; train.jsonl / val.jsonl are already included). Edit the KB list inside to match your real project first.
2. Kaggle -> Datasets -> New Dataset -> upload train.jsonl + val.jsonl.
3. Kaggle -> New Notebook -> File -> Import Notebook -> finetune_kaggle.ipynb. Settings: GPU on, Internet on. Add the dataset as input. Run All.
4. Download project-ai-q8_0.gguf from the Output panel.
5. Run it locally:  llama-server -m "C:\path\project-ai-q8_0.gguf" -c 16384 --port 8080
6. The app's local routes call http://localhost:8080/v1/chat/completions, so nothing in the app changes.
Use ONLY synthetic/public data. Kaggle runs on Google's servers, so never upload real patient data.

Dataset v2 jobs: report summary (3-20 params), RAG (3-6 sources), research comparison (2-4 papers), long-paper map-reduce (extract / merge / compare-from-facts), assistant.
Longest training example is ~9,000 characters (~2.2K tokens). It does NOT teach 50K+ character inputs; the app must chunk long documents.
If training runs out of GPU memory, set per_device_train_batch_size=1 and gradient_accumulation_steps=16.
If it is too slow, set num_train_epochs=1.
