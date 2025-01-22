import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
from gpt_model import (
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    load_weights_into_gpt,
    download_and_load_gpt2,
    plot_values
)
tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    def __len__(self):
        return len(self.data)
    def _longest_encoded_length(self):
        max_len = 0
        for item in self.encoded_texts:
            max_len = max(max_len, len(item))
        return max_len
    
train_dataset = SpamDataset("train.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataset("validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset("test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

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

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)

text_1 = "Every effort moves you"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))


# replace output layer with 2 class classification layer

torch.manual_seed(123)
num_classes = 2
model.out_head = nn.Linear(in_features=768, out_features=num_classes)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0,0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)
            logits = logits[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += logits.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    
    loss = F.cross_entropy(logits, target_batch)
    return loss

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy before finetuning: {train_accuracy*100:.2f}%")
print(f"Validation accuracy before finetuning: {val_accuracy*100:.2f}%")
print(f"Test accuracy before finetuning: {test_accuracy*100:.2f}%")

# Training accuracy: 46.25%
# Validation accuracy: 45.00%
# Test accuracy: 48.75%

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable params before: {total_params}')

for param in model.parameters():
    param.requires_grad = False

# Total trainable params before: 124441346

# replace Linear with LinearWithLoRA
replace_linear_with_lora(model, rank=16, alpha=16)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable LoRA params: {total_params:,}')

# Total trainable LoRA params: 2,666,528


# LORA FINETUNING
start_time = time.time()

torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5
)
end_time = time.time()

execution_time_min = (end_time - start_time) / 60

print(f'Training completed in {execution_time_min:0.2f} min')


# LOSS PLOT
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy after finetuning: {train_accuracy*100:.2f}%")
print(f"Validation accuracy after finetuning: {val_accuracy*100:.2f}%")
print(f"Test accuracy after finetuning: {test_accuracy*100:.2f}%")

# Training accuracy after finetuning: 97.79%
# Validation accuracy after finetuning: 94.63%
# Test accuracy after finetuning: 93.00%

save_path = "lora_spam_classifier.pth"

def save_lora_layers(model, save_path):
    """
    Saves all LoRALayer parameters in a model to a file.
    
    Args:
        model (torch.nn.Module): The model containing LoRALayer components.
        save_path (str): Path to save the state dictionary of LoRALayers.
    """
    lora_state_dict = {}

    # Recursive function to collect LoRALayer parameters
    def collect_lora_parameters(module, prefix=""):
        for name, child in module.named_children():
            if isinstance(child, LoRALayer):  # Assuming LoRALayer is the target class
                lora_state_dict[f"{prefix}{name}"] = child.state_dict()
            else:
                collect_lora_parameters(child, prefix=f"{prefix}{name}.")

    # Collect LoRALayer parameters
    collect_lora_parameters(model)

    # Save the LoRALayer state dictionary
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRALayer parameters to {save_path}")

save_lora_layers(model, save_path)

def load_lora_layers(model, load_path):
    """
    Loads LoRALayer parameters into a model from a saved file.
    
    Args:
        model (torch.nn.Module): The model containing LoRALayer components.
        load_path (str): Path to the file containing the saved LoRALayer parameters.
    """
    # Load the saved LoRA parameters
    lora_state_dict = torch.load(load_path)
    
    # Recursive function to load parameters into LoRALayer
    def load_parameters(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}{name}"
            if full_name in lora_state_dict:  # Check if this layer has saved parameters
                if isinstance(child, LoRALayer):
                    child.load_state_dict(lora_state_dict[full_name])
                    print(f"Loaded parameters for {full_name}")
                else:
                    raise ValueError(f"Expected LoRALayer at {full_name}, but found {type(child)}.")
            else:
                load_parameters(child, prefix=f"{full_name}.")
    
    # Load parameters into the model
    load_parameters(model)
    print("Loaded all LoRALayer parameters.")


'''
BEFORE LORA FINETUNING ACCURACY
Training accuracy before finetuning: 46.25%
Validation accuracy before finetuning: 45.00%
Test accuracy before finetuning: 48.75%

AFTER LORA FINETUNING ACCURACY
Training accuracy after finetuning: 97.79%
Validation accuracy after finetuning: 94.63%
Test accuracy after finetuning: 93.00%
'''