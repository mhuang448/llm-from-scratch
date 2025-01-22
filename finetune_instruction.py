import urllib.request
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from gpt_model import (
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
    generate_and_print_sample,
    load_weights_into_gpt,
    download_and_load_gpt2,
    plot_losses
)

file_path = "./instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)

print("Number of entries:", len(data))

def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f'\n\n### Input:\n{entry['input']}' if entry['input'] else ""

    return instruction_text + input_text

entry = data[50]

input = format_input(entry)

response = f'\n\n### Response:\n{entry['output']}'

alpaca_prompt = input + response
print("Alpaca prompt:", alpaca_prompt)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            input = format_input(entry)
            response = f'\n\n### Response:\n{entry['output']}'
            self.encoded_texts.append(tokenizer.encode(input + response))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)
    
# Custom collate function for padding and creating targets
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        item += [pad_token_id] * (batch_max_length - len(item))
        input = torch.tensor(item[:-1])
        target = torch.tensor(item[1:])
        mask = target == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            target[indices[1:]] = ignore_index
            
        if allowed_max_length is not None:
            input, target = input[:allowed_max_length], target[:allowed_max_length]
            
        inputs_lst.append(input)
        targets_lst.append(target)

    inputs = torch.stack(inputs_lst).to(device)
    targets = torch.stack(targets_lst).to(device)
    return inputs, targets
    

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

from functools import partial

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length = 1024)
    

torch.manual_seed(123)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion: train_portion + test_portion]
val_data = data[train_portion + test_portion:]


num_workers=0
batch_size = 8

tokenizer = tiktoken.get_encoding("gpt2")


train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    collate_fn = customized_collate_fn,
    batch_size=batch_size,
    drop_last = True,
    shuffle=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    collate_fn = customized_collate_fn,
    batch_size=batch_size,
    drop_last = False,
    shuffle=True,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    collate_fn = customized_collate_fn,
    batch_size=batch_size,
    drop_last = False,
    shuffle=True,
    num_workers=num_workers
)

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss



model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)

print("num Params before fine-tuning:", f'{sum(p.numel() for p in model.parameters()):,}')


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
    
model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )


print('train loss before fine-tuning:', train_loss)
print('val loss before fine-tuning:', val_loss)


# FINE-TUNING TRAINING
start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs=2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


torch.save(model.state_dict(), "instruction_tuned_medium.pth")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.manual_seed(123)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, top_idx = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_logits[:, -1],
                torch.tensor(float('-inf'), device=logits.device),
                logits
            )
        if temperature > 0.0:
            logits /= temperature

        probs = torch.softmax(logits, dim=-1)

        idx_new = torch.multinomial(probs, num_samples=1)

        if idx_new == eos_id:
            break

        
        idx = torch.cat((idx, idx_new), dim=-1)

    return idx

for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------") 

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

# Save the test data with model responses
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing


# start ollama with  `ollama serve`

import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))

def query_model (
    prompt, model="llama3.2", url="http://localhost:11434/api/chat"
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data=""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode('utf-8')
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

print("3 Examples of Ollama Evaluation")
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")


def generate_model_scores(json_data, json_key, model="llama3.2"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f'Could not convert score: {score}')
            continue

    return scores

scores = generate_model_scores(test_data, "model_response")
print(f'Number of scores: {len(scores)} of {len(test_data)}')
print(f'Average score: {sum(scores) / len(scores) : 0.2f}')


with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )

print('train loss:', train_loss)
print('val loss:', val_loss)

'''

GPT-2 MEDIUM

Got average score of 41.69 with GPT-2 Medium with llama3.2
Able to score 90 out of 110 test dataset entries


Got average score of 43.19 with GPT-2 Medium withllama3
able to score 109 out of 110 test dataset entries

resulting train and val loss:
train loss: 0.3332286596298218
val loss: 0.6615054965019226


GPT-2 LARGE

Got average score of 47.16 with GPT-2 Large with llama3

Losses
train loss: 0.26346977055072784
val loss: 0.6283906042575836

'''