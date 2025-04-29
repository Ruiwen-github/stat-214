import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from encoder import Encoder

# --- MLM MASKING ---
# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15 ):
    '''
    TODO: Implement MLM masking
    Args:
        input_ids: Input IDs
        vocab_size: Vocabulary size
        mask_token_id: Mask token ID
        pad_token_id: Pad token ID
        mlm_prob: Probability of masking
    '''
    # Clone inputs for labels (we’ll ignore -100 positions in loss)
    device = input_ids.device  # <-- ensures all tensors are on same device

    labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mlm_prob, device=device)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% random
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels

def train_bert(model, train_loader, val_loader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    '''
    TODO: Implement training loop for BERT
    Args:
        model: BERT model
        dataloader: Data loader
        tokenizer: Tokenizer
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run the model on
    '''
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            masked_input_ids, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            outputs = model(masked_input_ids, token_type_ids, attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                masked_input_ids, labels = mask_tokens(
                    input_ids.clone(),
                    vocab_size=tokenizer.vocab_size,
                    mask_token_id=tokenizer.mask_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

                masked_input_ids = masked_input_ids.to(device)
                labels = labels.to(device)

                outputs = model(masked_input_ids, token_type_ids, attention_mask)
                val_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_encoder_model.pt')
            print(f"✅ Model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
        

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses