import torch
from transformers import AutoTokenizer
from src.models.transformer import Transformer
from src.utils.data_loader import create_data_loader

# Configuration
VOCAB_SIZE = 30522  # Example for BERT tokenizer
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 6
DROPOUT = 0.1
MAX_SEQ_LEN = 128
BATCH_SIZE = 16

def train_model():
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = Transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN)

    # Example dummy data
    texts = ["This is a sample sentence.", "Another example for NLP.", "Machine learning is fun."]
    data_loader = create_data_loader(texts, tokenizer, MAX_SEQ_LEN, BATCH_SIZE)

    # Simple training loop (for demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            # Dummy target for demonstration
            target = torch.randint(0, VOCAB_SIZE, input_ids.shape)
            loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
