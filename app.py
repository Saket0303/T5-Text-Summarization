from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load your fine-tuned model
MODEL_PATH = "t5_finetuned_summarizer"   # change if your folder name is different

summarizer = pipeline(
    "summarization",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

app = FastAPI(title="T5 Text Summarization API")

class TextInput(BaseModel):
    text: str


@app.post("/summarize")
def summarize_text(input: TextInput):
    summary = summarizer(
        input.text,
        max_length=120,
        min_length=40,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3
    )[0]["summary_text"]

    return {
        "input_length": len(input.text.split()),
        "summary": summary
    }
