import torch
from transformers import AutoTokenizer, BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, texts, summaries, max_length):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        # Tokenize text and summary
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")[0]
        target_ids = self.tokenizer.encode(summary, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")[0]
        return input_ids, target_ids

