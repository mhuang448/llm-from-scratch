# Building and Fine-tuning GPT Models From Scratch

This project provides a hands-on implementation and exploration of Generative Pre-trained Transformer (GPT) models using PyTorch. It covers the fundamental building blocks of the Transformer architecture, data preparation pipelines for large text datasets, and fine-tuning techniques for adapting pre-trained models to specific downstream tasks like instruction following and text classification.

**Goal:** To demonstrate a practical understanding of modern Large Language Model (LLM) architecture, pre-training data handling, and fine-tuning methodologies relevant to AI Engineering roles.

## Table of Contents

- [Key Features](#key-features)
- [Technical Implementation](#technical-implementation)
  - [Core GPT Model (`gpt_model.py`)](#core-gpt-model-gpt_modelpy)
  - [Data Preparation (`prep_fineweb.py`, `prep_hellaswag.py`)](#data-preparation-prep_finewebpy-prep_hellaswagpy)
  - [Instruction Fine-tuning (`finetune_instruction.py`)](#instruction-fine-tuning-finetune_instructionpy)
  - [Classification Fine-tuning (`finetune_spam_classifier.py`)](#classification-fine-tuning-finetune_spam_classifierpy)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Fine-tuning](#fine-tuning)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Demonstrated Skills](#demonstrated-skills)
- [Technologies Used](#technologies-used)
- [Potential Future Work](#potential-future-work)

## Key Features

- **GPT Model Implementation:** A clear PyTorch implementation of the decoder-only Transformer architecture, including Multi-Head Self-Attention, Layer Normalization, GELU activation, and Positional Embeddings.
- **Pre-trained Weight Loading:** Functionality to download and load weights from official OpenAI GPT-2 checkpoints (various sizes) into the custom model implementation.
- **Data Preparation Pipelines:** Scripts for downloading, preprocessing, and tokenizing large datasets suitable for language model training:
  - **FineWeb-Edu:** For pre-training (demonstrated in `prep_fineweb.py`).
  - **HellaSwag:** For evaluating common-sense reasoning (demonstrated in `prep_hellaswag.py`).
  - **Instruction Dataset:** Custom JSON loading for instruction fine-tuning (`finetune_instruction.py`).
  - **SMS Spam Dataset:** Loading and preprocessing for classification fine-tuning (`finetune_spam_classifier.py`).
- **Instruction Fine-tuning:** Adapting a pre-trained GPT-2 model to follow natural language instructions based on an Alpaca-style dataset. Includes custom data loading, padding/masking strategies, and generation logic.
- **Classification Fine-tuning:** Adapting a pre-trained GPT-2 model for a binary classification task (SMS spam detection) by modifying the output head and selectively training layers.
- **Evaluation:**
  - Standard loss and accuracy metrics for training and validation.
  - Text generation capabilities for qualitative assessment.
  - LLM-as-Judge evaluation using Ollama (e.g., Llama 3.1) to score the quality of instruction-following responses (`finetune_instruction.py`).
- **Hardware Acceleration:** Supports training and inference on CUDA GPUs or Apple Silicon (MPS) where available.

## Technical Implementation

### Core GPT Model (`gpt_model.py`)

The heart of the project is the `GPTModel` class, which implements a decoder-only Transformer architecture inspired by GPT-2.

- **Embeddings:** Uses separate embeddings for tokens (`nn.Embedding`) and positions (`nn.Embedding`).
- **Transformer Blocks:** Composed of multiple `TransformerBlock` layers. Each block contains:
  - `MultiHeadAttention`: Implements multi-head self-attention with optional masking for causal language modeling and dropout. Includes optional query, key, value biases (`qkv_bias`).
  - `LayerNorm`: Custom Layer Normalization implementation.
  - `FeedForward`: A position-wise feed-forward network using GELU activation.
  - **Residual Connections & Dropout:** Standard residual connections (`x = x + dropout(sublayer(norm(x)))`) are used around the attention and feed-forward layers.
- **Output Head:** A final Layer Normalization followed by a Linear layer (`out_head`) projects the final hidden states to the vocabulary size for generating token probabilities.
- **Weight Loading:** The `load_weights_into_gpt` function carefully maps and reshapes weights from OpenAI's TensorFlow checkpoints into the PyTorch model's layers, handling differences in naming conventions and tensor dimensions.

### Data Preparation (`prep_fineweb.py`, `prep_hellaswag.py`)

Efficient data handling is crucial for LLMs.

- **`prep_fineweb.py`:**
  - Downloads the FineWeb-Edu dataset (a large, filtered web crawl) using the `datasets` library.
  - Uses `tiktoken` (GPT-2 encoding) for fast tokenization.
  - Employs multiprocessing (`multiprocessing.Pool`) to parallelize tokenization across CPU cores.
  - Implements sharding: Saves the tokenized data into multiple smaller files (`.npy`) of a fixed token count (`shard_size`) to handle datasets that don't fit into memory. Includes logic to split documents across shards correctly.
  - Designates the first shard as the validation set.
- **`prep_hellaswag.py`:**
  - Downloads the HellaSwag dataset (multiple-choice common-sense reasoning).
  - Provides functions (`render_example`, `iterate_examples`) to process the JSONL data, tokenize context and endings using `tiktoken`, and format them into tensors suitable for evaluation (calculating loss on different endings).
  - Includes an example evaluation loop using Hugging Face's `transformers` library implementation of GPT-2 for comparison/validation purposes (though the main focus is the custom model).

### Instruction Fine-tuning (`finetune_instruction.py`)

This script demonstrates fine-tuning a pre-loaded GPT-2 model (e.g., `gpt2-medium`) on an instruction-following task.

- **Data Loading:** Reads a JSON dataset (Alpaca format assumed: `instruction`, `input`, `output`).
- **Formatting:** Creates prompts by combining instruction, input (optional), and response fields with specific separators (`### Instruction:`, `### Response:`).
- **`InstructionDataset` & `DataLoader`:** Uses a custom PyTorch `Dataset` and `DataLoader`.
- **`custom_collate_fn`:** A critical function that handles batching sequences of varying lengths. It pads sequences to the maximum length in the batch using a specified `pad_token_id` and creates target tensors for language modeling loss. Importantly, it sets the loss target to `ignore_index` (-100) for padding tokens and potentially for the prompt tokens to focus training only on generating the response (though the current implementation seems to train on the whole sequence).
- **Training Loop:** Implements a standard fine-tuning loop using `AdamW` optimizer, calculating cross-entropy loss, and performing backpropagation. Includes periodic evaluation on a validation set and generation of sample outputs.
- **Generation:** Implements a `generate` function with options for temperature scaling and top-k sampling for more diverse outputs, along with an End-Of-Sequence (`eos_id`) check.
- **LLM-as-Judge Evaluation:** After fine-tuning, it generates responses for a test set and uses an external LLM (via Ollama, e.g., Llama 3.1) to score the quality of the generated responses against the ground truth, providing an automated quality assessment metric.

### Classification Fine-tuning (`finetune_spam_classifier.py`)

This script adapts the pre-trained GPT-2 model for SMS spam classification.

- **Data Loading:** Downloads the SMS Spam Collection dataset, preprocesses it using `pandas`, creates a balanced dataset (downsampling the majority class 'ham'), and splits it into train/validation/test sets.
- **`SpamDataset`:** Custom dataset that tokenizes text messages and pads them to a maximum length.
- **Model Adaptation:**
  - **Freezing Layers:** Most of the pre-trained model parameters are frozen (`param.requires_grad = False`) to preserve the learned language representations.
  - **Modifying Output Head:** The original `out_head` (mapping to vocabulary size) is replaced with a new `nn.Linear` layer mapping the final hidden state dimension (`emb_dim`) to the number of classes (2 for spam/ham).
  - **Selective Unfreezing:** Only the parameters of the newly added output head, the final `LayerNorm`, and the last `TransformerBlock` are unfrozen (`requires_grad = True`) for training. This is a common and efficient fine-tuning strategy.
- **Training & Evaluation:**
  - Uses `AdamW` optimizer on the unfrozen parameters.
  - Calculates classification accuracy (`calc_accuracy_loader`) and cross-entropy loss (`calc_loss_batch`, `calc_loss_loader`), focusing on the logits produced for the _last token_ position of the input sequence.
  - Plots training/validation loss and accuracy over epochs.
- **Inference:** Provides a `classify_review` function to classify new text messages using the fine-tuned model.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mhuang448/llm-from-scratch.git
    cd llm-from-scratch
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Likely dependencies include: torch, tiktoken, numpy, datasets, tqdm, requests, pandas, matplotlib, transformers, psutil
    # Make sure to create a requirements.txt file!
    ```
4.  **(Optional) Install Ollama:** For LLM-as-Judge evaluation in `finetune_instruction.py`, follow the instructions at [https://ollama.com/](https://ollama.com/) to install and run Ollama. Pull a model like Llama 3.1: `ollama pull llama3.1`.

## Usage

### Data Preparation

- **FineWeb-Edu:**
  ```bash
  python prep_fineweb.py
  # This will download data and create tokenized shards in ./edu_fineweb10B/
  ```
- **HellaSwag (Download only):** The `prep_hellaswag.py` script is primarily for evaluation setup but includes download logic triggered by `iterate_examples`. Running the evaluation script (see below) will handle this.
- **SMS Spam:** The `finetune_spam_classifier.py` script handles downloading and preprocessing automatically on the first run.
- **Instruction Data:** Place your `instruction-data.json` file in the project's root directory (or update the path in `finetune_instruction.py`).

### Fine-tuning

- **Instruction Fine-tuning:**

  ```bash
  # Ensure Ollama is running if you want the final evaluation step
  ollama serve & # Run in background (example, may vary by OS)

  # Run the fine-tuning script (adjust model size, epochs, etc. inside the script)
  python finetune_instruction.py
  # This will download GPT-2 weights, fine-tune, save the model to instruction_tuned_medium.pth (or similar),
  # generate examples, and run Ollama evaluation.
  ```

- **Spam Classification Fine-tuning:**
  ```bash
  # Run the fine-tuning script (adjust model size, epochs, etc. inside the script)
  python finetune_spam_classifier.py
  # This will download data/weights, fine-tune, save the model to spam_classifier.pth,
  # plot metrics, and show example classifications.
  ```

### Evaluation

- **HellaSwag (using Hugging Face model):**
  ```bash
  python prep_hellaswag.py -m gpt2-xl -d cuda # Or gpt2, gpt2-medium, etc. Use -d cpu if no GPU
  ```
- **Instruction Following (LLM-as-Judge):** Performed automatically at the end of `finetune_instruction.py`. Requires Ollama to be running.
- **Spam Classification:** Accuracy/Loss metrics are printed and plotted during and after running `finetune_spam_classifier.py`.

## Project Structure

```
.
├── gpt_model.py             # Core GPT model implementation & weight loading
├── finetune_instruction.py  # Script for instruction fine-tuning & evaluation
├── finetune_spam_classifier.py # Script for spam classification fine-tuning
├── prep_fineweb.py          # Script for downloading and tokenizing FineWeb-Edu
├── prep_hellaswag.py        # Script for downloading and evaluating HellaSwag
├── instruction-data.json    # Example instruction dataset (User needs to provide)
├── edu_fineweb10B/          # Directory for FineWeb tokenized shards (created by prep_fineweb.py)
├── hellaswag/               # Directory for HellaSwag data (created by prep_hellaswag.py)
├── gpt2/                    # Directory for downloaded GPT-2 model weights (created by scripts)
├── *.csv                    # Train/Val/Test splits for spam data (created by finetune_spam_classifier.py)
├── *.pth                    # Saved fine-tuned model weights (created by fine-tuning scripts)
├── *.pdf                    # Plots generated by fine-tuning scripts
├── requirements.txt         # Project dependencies (User should create)
└── README.md                # This file
```

## Demonstrated Skills

- **Deep Learning Frameworks:** Proficient use of PyTorch for building and training complex neural networks.
- **Transformer Architecture:** Deep understanding and implementation of core components like Multi-Head Self-Attention, Positional Encodings, Layer Normalization, and Residual Connections.
- **LLM Fine-tuning:** Practical application of various fine-tuning strategies:
  - Full model fine-tuning (implied in instruction tuning, though depends on optimizer setup).
  - Parameter-efficient fine-tuning principles (layer freezing, adapting output heads for classification).
- **Data Engineering for AI:**
  - Handling large datasets (sharding, efficient tokenization).
  - Preprocessing diverse data formats (JSON, CSV, web text).
  - Creating custom Datasets and DataLoaders in PyTorch, including sophisticated collation (padding, masking).
  - Dataset balancing techniques.
- **Tokenization:** Using standard tokenizers (`tiktoken`) effectively.
- **MLOps Concepts:**
  - Model persistence (saving/loading weights).
  - Integration with pre-trained models (GPT-2).
  - Evaluation methodologies (loss, accuracy, generation, LLM-as-Judge).
  - Dependency management.
- **Software Engineering Practices:** Modular code structure, clear function definitions, use of virtual environments.

## Technologies Used

- **Python 3.x**
- **PyTorch:** Core deep learning framework.
- **tiktoken:** Fast BPE tokenizer from OpenAI.
- **NumPy:** Numerical operations.
- **Pandas:** Data manipulation for CSV files.
- **Hugging Face `datasets`:** Downloading large public datasets.
- **Matplotlib:** Plotting training metrics.
- **tqdm:** Progress bars.
- **requests:** Downloading files (used in `prep_hellaswag.py`).
- **Ollama (Optional):** Local LLM runner for evaluation.
- **psutil (Optional):** Checking if Ollama process is running.

## Potential Future Work

- Implement more advanced fine-tuning techniques (e.g., QLoRA).
- Integrate with ML experiment tracking tools (e.g., Weights & Biases, MLflow).
- Add support for different model architectures or pre-trained weights.
- Implement distributed training for larger models/datasets.
- Develop a simple API (e.g., using FastAPI) to serve the fine-tuned models.
- Expand evaluation suites (e.g., ROUGE scores for summarization, BLEU for translation if applicable tasks are added).

```

```
