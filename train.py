import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

#LOAD DATASET

#dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
dataset = dataset.train_test_split(test_size=0.05)

train_data = dataset["train"]
val_data = dataset["test"]

print(train_data[0])

#PREPROCESS

max_input_length = 384 #512
max_target_length = 96 #128

def preprocess(batch):
    inputs = ["summarize: " + doc for doc in batch["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        #padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["highlights"],
            max_length=max_target_length,
            truncation=True,
            #padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


#TOKENIZATION

tokenized_train = train_data.map(
    preprocess,
    batched=True,
    remove_columns=train_data.column_names
)

tokenized_val = val_data.map(
    preprocess,
    batched=True,
    remove_columns=val_data.column_names
)

#MODEL LOADING

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

#Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_summarizer",
    eval_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=1,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    report_to="none",

    predict_with_generate=True,
    generation_max_length=80,
    generation_num_beams=2,
    eval_accumulation_steps=4
)

#TRAINING

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

#SAVE MODEL

model.save_pretrained("t5_finetuned_summarizer")
tokenizer.save_pretrained("t5_finetuned_summarizer")

