# Own AI v3 model

The app uses the team's fine-tuned Qwen2.5-1.5B-Instruct v3 GGUF directly
inside the Flask process through `llama-cpp-python`.

1. Install the project dependencies with `python -m pip install -r requirements.txt`.
2. Set `OWN_MODEL_PATH` in `.env` to
   `models\own_ai_v3\Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`.
3. Optionally set `OWN_N_CTX`; the default is 4096.
4. Start Flask normally. The model is loaded once, lazily, on the first own-model request.

The own-model integration does not require a separate model server. Use only
synthetic or public data for fine-tuning; never upload real patient data to a
third-party training service.

Dataset v2 jobs: report summary (3-20 params), RAG (3-6 sources), research comparison (2-4 papers), long-paper map-reduce (extract / merge / compare-from-facts), assistant.
Longest training example is ~9,000 characters (~2.2K tokens). It does NOT teach 50K+ character inputs; the app must chunk long documents.
If training runs out of GPU memory, set per_device_train_batch_size=1 and gradient_accumulation_steps=16.
If it is too slow, set num_train_epochs=1.
