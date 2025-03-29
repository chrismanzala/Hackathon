import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    pipeline
)

class Preprocessing:
    """Data preprocessing"""
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def data_tokenize(self, data, batch_size = 2, max_source_length=None, max_target_length=None):
        source_tokenized = []
        target_tokenized = []
        
        num_batches = (len(data) + batch_size - 1) // batch_size  # Calculate the number of batches
    
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data[start_idx:end_idx]  # Extract a batch of data
            
            for idx in range(len(batch_data)):
                source, target = batch_data["original_text"][idx], batch_data["reference_summary"][idx]
                source_encoded = self.tokenizer(
                    source, padding="max_length", truncation=True, max_length=max_source_length, return_tensors="pt"
                )["input_ids"]
                target_encoded = self.tokenizer(
                    target, padding="max_length", truncation=True, max_length=max_target_length, return_tensors="pt"
                )["input_ids"]
                
                source_tokenized.append(source_encoded)
                target_tokenized.append(target_encoded)
        
        return source_tokenized, target_tokenized

    # def data_tokenize(self, data, batch,max_source_length=None, max_target_length=None):
        
    #     source_tokenized = []
    #     target_tokenized = []
        
    #     for batch in data.shape[]
    #         for idx in np.arange(data.shape[0]):
    #             source, target = data["original_text"][idx], data["reference_summary"][idx]
    #             source_encoded = self.tokenizer(
    #                 source, padding="max_length", truncation=True, max_length=max_source_length, return_tensors="pt"
    #             )["input_ids"]
    #             target_encoded = self.tokenizer(
    #                 target, padding="max_length", truncation=True, max_length=max_target_length, return_tensors="pt"
    #             )["input_ids"]
                
    #             source_tokenized.append(source_encoded)
    #             target_tokenized.append(target_encoded)
            
    #     return source_tokenized, target_tokenized
    
    def data_embed(self, source_tokenized, target_tokenized):
        source_embeddings = []
        target_embeddings = []
        
        for source_input_ids, target_input_ids in zip(source_tokenized, target_tokenized):
            eos_token_id_tensor = torch.tensor([[self.tokenizer.eos_token_id]])
            
            source_outputs = self.model(input_ids=source_input_ids, decoder_input_ids=eos_token_id_tensor)
            target_outputs = self.model(input_ids=target_input_ids, decoder_input_ids=eos_token_id_tensor)
            
            source_embeddings.append(source_outputs.encoder_last_hidden_state)
            target_embeddings.append(target_outputs.encoder_last_hidden_state)
        
        return source_embeddings, target_embeddings


# class Preprocessing:
#     """Data preprocessing"""
#     def __init__(self):
#         self.model_name = "facebook/bart-large-cnn"
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
#     def data_tokenize_embed(self, data, max_source_length=None, max_target_length=None):
#         source_embeddings = []
#         target_embeddings = []
        
#         for idx in np.arange(data.shape[0]):
#             source, target = data["original_text"][idx], data["reference_summary"][idx]
#             source_tokenized = self.tokenizer(
#                 source, padding="max_length", truncation=True, max_length=max_source_length, return_tensors="pt"
#             )["input_ids"]
#             target_tokenized = self.tokenizer(
#                 target, padding="max_length", truncation=True, max_length=max_target_length, return_tensors="pt"
#             )["input_ids"]

#             # Create tensor containing the EOS token ID
#             eos_token_id_tensor = torch.tensor([[self.tokenizer.eos_token_id]])

#             source_outputs = self.model(input_ids=source_tokenized, decoder_input_ids=eos_token_id_tensor)
#             target_outputs = self.model(input_ids=target_tokenized, decoder_input_ids=eos_token_id_tensor)
        
#             source_embeddings.append(source_outputs.encoder_last_hidden_state)
#             target_embeddings.append(target_outputs.encoder_last_hidden_state)
        
#         return zip(source_embeddings, target_embeddings)

# class Training:
    # """Model training"""
    # def __init__(self, data_embedded):
    #     self.data_embedded = data_embedded
    #     self.model_name = "facebook/bart-large-cnn"
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #     self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        
    #     self.training_args = Seq2SeqTrainingArguments(
    #         output_dir="results",
    #         num_train_epochs=1,
    #         do_train=True,
    #         do_eval=True,
    #         per_device_train_batch_size=4,
    #         per_device_eval_batch_size=4,
    #         warmup_steps=500,
    #         weight_decay=0.1,
    #         label_smoothing_factor=0.1,
    #         predict_with_generate=True,
    #         logging_dir="logs",
    #         logging_steps=50,
    #         save_total_limit=3
    #     )
        
    #     self.trainer = Seq2SeqTrainer(
    #         model=self.model,
    #         args=self.training_args,
    #         data_collator=self.data_collator,
    #         train_dataset=self.data_embedded['original_text'],
    #         eval_dataset=self.data_embedded['reference_summary'],
    #         tokenizer=self.tokenizer
    #     )
        
    # def training(self):
    #     self.trainer.train()
        
    # def saving(self):
    #     self.trainer.save_model('GGM')
        
class Training:
    """Model training"""
    def __init__(self, source_embeddings, target_embeddings):
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        
        self.model_name = "facebook/bart-large-cnn"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        
        # Combine source and target embeddings into datasets
        self.train_dataset = [{"source": source, "target": target} for source, target in zip(source_embeddings, target_embeddings)]
        self.eval_dataset = self.train_dataset  # For simplicity, use the same data for evaluation
        
        self.training_args = Seq2SeqTrainingArguments(
            output_dir="results",
            num_train_epochs=1,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.1,
            label_smoothing_factor=0.1,
            predict_with_generate=True,
            logging_dir="logs",
            logging_steps=50,
            save_total_limit=3
        )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer
        )
        
    def training(self):
        self.trainer.train()
        
    def saving(self):
        self.trainer.save_model('GGM')
  
class CustomSummarizer:
    def __init__(self, model_name_or_path, device="cpu"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device
    
    def summarize(self, text, max_length=128, min_length=10, num_beams=4):
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length").input_ids.to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            decoder_start_token_id=self.model.config.decoder.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id
        )
        
        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary

# Example usage:
summarizer = CustomSummarizer("GGM", device="cuda" if torch.cuda.is_available() else "cpu")
summary = summarizer.summarize("Your input text goes here.")
print("Summary:", summary)


# class Pipe:
#     def __init__(self):
#         self.pipe = pipeline('summarization', model='GGM')
#         self.gen_kwargs = {'length_penalty': 0.8, 'num_beams': 8, "max_length": 128}
    
#     def summarization(self, text):
#         return self.pipe(text, **self.gen_kwargs)