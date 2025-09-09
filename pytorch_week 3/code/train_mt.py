import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from custom_transformer import Transformer, PositionalEncoding, MultiHeadAttention
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Hyperparameters
# -----------------------------
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
dropout = 0.1
num_epochs = 100
batch_size = 64
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Toy Translation Dataset
# -----------------------------
data = [
    ("hello world", "bonjour le monde"),
    ("what is your name", "quel est ton nom"),
    ("my name is", "mon nom est"),
    ("i am fine", "je vais bien"),
    ("good morning", "bonjour matin"),
    ("good night", "bonne nuit")
]

src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
src_idx, tgt_idx = 3, 3

for s, t in data:
    for w in s.split():
        if w not in src_vocab:
            src_vocab[w] = src_idx
            src_idx += 1
    for w in t.split():
        if w not in tgt_vocab:
            tgt_vocab[w] = tgt_idx
            tgt_idx += 1

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

idx_to_src = {v: k for k, v in src_vocab.items()}
idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

max_seq_len = 10
def preprocess(sentence, vocab, is_target=False):
    tokens = sentence.split()
    tokens = ['<sos>'] + tokens + ['<eos>']
    indices = [vocab.get(t, 0) for t in tokens]
    if len(indices) < max_seq_len:
        indices.extend([vocab['<pad>']] * (max_seq_len - len(indices)))
    else:
        indices = indices[:max_seq_len]
    return torch.tensor(indices, dtype=torch.long)

src_data = torch.stack([preprocess(s, src_vocab) for s, t in data])
tgt_data = torch.stack([preprocess(t, tgt_vocab, is_target=True) for s, t in data])

dataset = TensorDataset(src_data, tgt_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Masking functions
# -----------------------------
def create_src_mask(src):
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_tgt_mask(tgt):
    tgt_pad_mask = (tgt != tgt_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    nopeek_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    nopeek_mask = nopeek_mask.unsqueeze(0).unsqueeze(0).to(device)
    return tgt_pad_mask & nopeek_mask

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training
# -----------------------------
train_losses = []
print("Starting Transformer training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        src_mask = create_src_mask(src)
        tgt_mask = create_tgt_mask(tgt[:, :-1])
        
        outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
        loss = criterion(outputs.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete!")
os.makedirs("runs/mt", exist_ok=True)
torch.save(model.state_dict(), "runs/mt/transformer.pth")

# -----------------------------
# Evaluation and Visualization
# -----------------------------
model.eval()
def translate(model, sentence, src_vocab, tgt_vocab, device, max_len=10):
    src = preprocess(sentence, src_vocab).unsqueeze(0).to(device)
    src_mask = create_src_mask(src)
    
    with torch.no_grad():
        encoder_output = model.encoder_embedding(src)
        encoder_output = model.positional_encoding(encoder_output)
        for layer in model.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
    
    tgt = torch.tensor([[tgt_vocab['<sos>']]]).to(device)
    for _ in range(max_len):
        tgt_mask = create_tgt_mask(tgt)
        
        with torch.no_grad():
            decoder_input = model.decoder_embedding(tgt)
            decoder_output = model.positional_encoding(decoder_input)
            
            for layer in model.decoder_layers:
                decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
            outputs = model.linear(decoder_output)
            
        next_token_probs = outputs[:, -1, :]
        next_token = next_token_probs.argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == tgt_vocab['<eos>']:
            break
            
    decoded_tokens = [idx_to_tgt[idx.item()] for idx in tgt.squeeze(0)]
    return decoded_tokens

print("\nGenerating decoded outputs and calculating BLEU score...")
decoded_outputs = []
references = []
decodes_table_md = "### Decoded Translations\n\n| Source | Ground Truth | Decoded Output |\n|---|---|---|\n"

for src, tgt in data:
    translated_tokens = translate(model, src, src_vocab, tgt_vocab, device)
    decoded_sentence = " ".join(translated_tokens[1:-1])
    
    references.append([t.split()])
    decoded_outputs.append(translated_tokens[1:])
    
    decodes_table_md += f"| {src} | {tgt} | {decoded_sentence} |\n"

bleu = corpus_bleu(references, decoded_outputs)
print(f"Corpus BLEU score: {bleu:.4f}")

with open("runs/mt/bleu_report.txt", "w") as f:
    f.write(f"Corpus BLEU score: {bleu:.4f}\n")
with open("runs/mt/decodes_table.md", "w") as f:
    f.write(decodes_table_md)
    
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.title("Transformer Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("runs/mt/curves_mt.png")
plt.close()

def get_attention_weights(model, sentence, src_vocab, device):
    src = preprocess(sentence, src_vocab).unsqueeze(0).to(device)
    src_mask = create_src_mask(src)
    
    attention_weights = []
    def hook_fn(module, input, output):
        attention_weights.append(output[1].detach())
        
    hooks = []
    for layer in model.encoder_layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        embedded_src = model.encoder_embedding(src)
        encoder_output = model.positional_encoding(embedded_src)
        for layer in model.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
    
    for hook in hooks:
        hook.remove()
        
    return attention_weights

print("Generating attention heatmaps...")
sample_sentence = "what is your name"
attention_weights = get_attention_weights(model, sample_sentence, src_vocab, device)
src_tokens = ['<sos>'] + sample_sentence.split() + ['<eos>']

for layer_idx, attn_per_layer in enumerate(attention_weights):
    for head_idx in range(n_heads):
        plt.figure(figsize=(8, 8))
        attn_matrix = attn_per_layer[0, head_idx, :len(src_tokens), :len(src_tokens)].cpu().numpy()
        plt.imshow(attn_matrix, cmap='viridis')
        plt.colorbar(label='Attention Score')
        plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
        plt.yticks(range(len(src_tokens)), src_tokens)
        plt.title(f"Layer {layer_idx+1}, Head {head_idx+1} Self-Attention")
        plt.tight_layout()
        plt.savefig(f"runs/mt/attention_layer{layer_idx+1}_head{head_idx+1}.png")
        plt.close()

def visualize_mask(mask, title, filename):
    plt.figure(figsize=(5, 5))
    plt.imshow(mask.cpu().squeeze().numpy(), cmap='gray')
    plt.title(title)
    plt.savefig(f"runs/mt/{filename}")
    plt.close()

print("Generating mask visualizations...")
src = torch.tensor([[1, 3, 4, 2, 0, 0, 0, 0, 0, 0]])
tgt = torch.tensor([[1, 5, 6, 7, 2, 0, 0, 0, 0, 0]])
src_mask = create_src_mask(src)
tgt_mask = create_tgt_mask(tgt[:, :-1])

visualize_mask(src_mask, "Source Padding Mask", "masks_src.png")
visualize_mask(tgt_mask.squeeze(0), "Target Causal + Padding Mask", "masks_tgt.png")

print("All Transformer tasks complete! Visuals saved in runs/mt/")