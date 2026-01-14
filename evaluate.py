# LOAD FINETUNED MODEL

from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="t5_finetuned_summarizer",
    tokenizer="t5_finetuned_summarizer"
)

text = """
The Transformer architecture has transformed NLP by replacing recurrence with
self-attention, enabling better parallelization and performance on long sequences.
"""

print(summarizer(text, max_length=50, min_length=20, do_sample=False))

# ROUGE COMPUTATION

import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Sometimes predictions come as a tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert to numpy
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Clip values to valid token range (safety)
    vocab_size = tokenizer.vocab_size
    predictions = np.clip(predictions, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)

    # Decode
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("t5_finetuned_summarizer_with_rouge")
tokenizer.save_pretrained("t5_finetuned_summarizer_with_rouge")

import torch, gc
torch.cuda.empty_cache()
gc.collect()

results = trainer.evaluate()
print(results)

# Print ROUGE Scores

print("\nROUGE Scores:")
print(f"ROUGE-1 : {results['eval_rouge1']:.4f}")
print(f"ROUGE-2 : {results['eval_rouge2']:.4f}")
print(f"ROUGE-L : {results['eval_rougeL']:.4f}")
