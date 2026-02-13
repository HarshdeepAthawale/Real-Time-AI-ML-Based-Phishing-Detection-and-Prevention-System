"""
Download pretrained AI-generated text detector from Hugging Face.
Falls back to DistilBERT base + binary classifier head if no specialized model found.

Run: python scripts/download_ai_detector.py
"""
import os
import sys

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "ai-detector"
    )
    os.makedirs(output_dir, exist_ok=True)

    alternatives = [
        "roberta-base-openai-detector",
        "eliegron/ai-detector",
    ]

    for model_id in alternatives:
        try:
            print(f"Downloading {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved to {output_dir}")
            return
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Fallback: DistilBERT with binary classification head (human=0, ai=1)
    print("Using fallback: DistilBERT with binary classifier...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "human_written", 1: "ai_generated"},
        label2id={"human_written": 0, "ai_generated": 1},
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fallback model to {output_dir}")

if __name__ == "__main__":
    main()
