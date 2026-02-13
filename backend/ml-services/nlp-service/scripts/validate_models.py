"""
Validation framework for phishing detection models.
Runs detections against labeled test sets and reports TPR, FPR, precision, recall, F1.
Outputs results for ModelPerformance tracking and drift detection.

Usage:
    # Validate local model
    python scripts/validate_models.py --model-dir models/phishing-detector --output results/

    # Validate via detection API (full stack)
    python scripts/validate_models.py --api-url http://localhost:3000/api/v1/detect/text --api-key YOUR_KEY

    # CI mode: parseable output, exits with code 1 if targets not met
    CI=1 python scripts/validate_models.py --model-dir models/phishing-detector
"""
import os
import json
import time
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import numpy as np


# --- Validation test set (separate from training data) ---
VALIDATION_SET: List[Tuple[str, int]] = [
    # Phishing (label=1)
    ("ALERT: Your bank account has been flagged for suspicious activity. Verify your identity immediately to prevent account suspension.", 1),
    ("Your Netflix account will be suspended in 24 hours. Update your payment details now: netf1ix-billing.com", 1),
    ("Congratulations! You have been randomly selected for a $500 cash reward. Click to claim before it expires.", 1),
    ("IT Department: All employees must re-enter their login credentials on the new security portal by end of day.", 1),
    ("Your Amazon Prime membership auto-renewal of $119 failed. Update payment at amaz0n-billing.xyz to avoid cancellation.", 1),
    ("URGENT: Legal action will be taken against you within 48 hours for unpaid tax. Call 1-888-555-0123 immediately.", 1),
    ("Your WhatsApp will be deactivated. Verify your number at whatsapp-verify.com to keep your account active.", 1),
    ("CEO Request: I need you to purchase 5 Apple gift cards worth $200 each for a client. Send me the codes ASAP.", 1),
    ("Your Spotify Premium account was accessed from an unknown device in China. Secure your account now.", 1),
    ("Microsoft Security Alert: We detected 3 unauthorized sign-ins to your Outlook account. Reset password immediately.", 1),
    ("Your DHL package requires customs payment of $4.99. Pay at dh1-customs.com/pay to release your shipment.", 1),
    ("Account verification required: Your PayPal account has been limited due to suspicious activity.", 1),
    ("Your computer has been infected with malware. Download our security tool immediately to protect your data.", 1),
    ("Wells Fargo notice: Your online banking access will be restricted. Click here to verify your identity.", 1),
    ("Job offer: Work from home and earn $5000/week. No experience needed. Apply now at easy-money-jobs.com", 1),

    # Legitimate (label=0)
    ("Hi Mike, I attached the revised budget proposal. Let me know if the numbers look right to you.", 0),
    ("Your Uber ride receipt: $23.47 from Downtown to Airport. Rate your driver.", 0),
    ("Team standup notes: Sprint velocity at 42 points. Two blockers identified for the API migration.", 0),
    ("Your dental appointment on Thursday has been confirmed. Remember to bring your insurance card.", 0),
    ("New blog post: 10 Tips for Better Code Reviews. Read on our engineering blog.", 0),
    ("Reminder: Submit your timesheet by Friday 5 PM. Late submissions may delay payroll.", 0),
    ("The Q3 earnings call is scheduled for October 15th at 4 PM EST. Dial-in details below.", 0),
    ("Your gym membership has been renewed for 2024. Annual fee of $599 charged to your card on file.", 0),
    ("Conference badges are ready for pickup. The event starts at 9 AM in Hall B.", 0),
    ("Your GitHub Actions workflow completed successfully. All 47 tests passed.", 0),
    ("Please review the attached contract amendment before the signing deadline on Friday.", 0),
    ("Your Costco order is ready for pickup at the warehouse on Oak Street.", 0),
    ("The project retrospective meeting is at 2 PM today in Room 301. Snacks provided!", 0),
    ("Your credit card statement is ready to view. Current balance: $1,234.56.", 0),
    ("Welcome to the team! Your desk is in Section C, Row 3. IT will set up your laptop today.", 0),
    ("Your flight LH 456 has been confirmed. Boarding at Gate 12, departs 14:30.", 0),
    ("Quarterly report attached. Revenue up 12% YoY. Please review before the board meeting.", 0),
    ("Phishing attempt: 'Click here to claim your prize at fake-giveaway.xyz' - obvious scam.", 0),
    ("IRS refund scam: 'You owe $0. Update at irs-fake.gov' - do not click.", 1),
]


def load_model(model_dir: str, device: str = "cpu"):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_batch(
    model, tokenizer, texts: List[str], device: str = "cpu", max_length: int = 256
) -> List[Dict]:
    """Run predictions on a batch of texts."""
    results = []
    for text in texts:
        start = time.time()
        inputs = tokenizer(
            text, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        phishing_prob = probs[0][1].item()
        latency_ms = (time.time() - start) * 1000

        results.append({
            "text_preview": text[:80],
            "phishing_probability": phishing_prob,
            "prediction": 1 if phishing_prob > 0.5 else 0,
            "latency_ms": round(latency_ms, 2),
        })

    return results


def compute_metrics(labels: List[int], predictions: List[int], probabilities: List[float]) -> Dict:
    """Compute comprehensive classification metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "auc_roc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "total_samples": len(labels),
        "phishing_samples": sum(labels),
        "legitimate_samples": len(labels) - sum(labels),
    }


def validate_via_api(
    api_url: str,
    api_key: str,
    output_dir: str = "results",
) -> Dict:
    """Validate by calling detection API (full-stack validation)."""
    try:
        import requests
    except ImportError:
        print("Error: requests library required for API validation. pip install requests")
        sys.exit(1)

    texts = [s[0] for s in VALIDATION_SET]
    labels = [s[1] for s in VALIDATION_SET]
    predictions = []
    probabilities = []
    latencies = []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    for text in texts:
        start = time.time()
        try:
            resp = requests.post(
                api_url,
                json={"text": text},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Detection API returns analysis_result.nlp.phishing_probability or similar
            nlp = (data.get("analysis_result") or {}).get("nlp") or data.get("nlp") or {}
            prob = nlp.get("phishing_probability", 0.5)
            probabilities.append(prob)
            predictions.append(1 if prob > 0.5 else 0)
        except Exception as e:
            print(f"API error: {e}")
            probabilities.append(0.5)
            predictions.append(0)
        latencies.append((time.time() - start) * 1000)

    metrics = compute_metrics(labels, predictions, probabilities)
    metrics["latency"] = {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
    }
    metrics["targets"] = {
        "tpr_target": 0.95,
        "tpr_met": metrics["tpr"] >= 0.95,
        "fpr_target": 0.02,
        "fpr_met": metrics["fpr"] <= 0.02,
        "latency_target_ms": 100,
        "latency_met": metrics["latency"]["p95_ms"] <= 100,
    }
    return metrics


def validate(model_dir: str, output_dir: str = "results", device: str = "cpu"):
    """Run full validation pipeline."""
    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_model(model_dir, device)

    texts = [s[0] for s in VALIDATION_SET]
    labels = [s[1] for s in VALIDATION_SET]

    print(f"Running predictions on {len(texts)} samples...")
    results = predict_batch(model, tokenizer, texts, device)

    predictions = [r["prediction"] for r in results]
    probabilities = [r["phishing_probability"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    metrics = compute_metrics(labels, predictions, probabilities)

    # Latency stats
    metrics["latency"] = {
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "mean_ms": round(np.mean(latencies), 2),
    }

    # Problem statement targets
    metrics["targets"] = {
        "tpr_target": 0.95,
        "tpr_met": metrics["tpr"] >= 0.95,
        "fpr_target": 0.02,
        "fpr_met": metrics["fpr"] <= 0.02,
        "latency_target_ms": 100,
        "latency_met": metrics["latency"]["p95_ms"] <= 100,
    }

    # Print report
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"  Model:     {model_dir}")
    print(f"  Samples:   {metrics['total_samples']} ({metrics['phishing_samples']} phishing, {metrics['legitimate_samples']} legitimate)")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TPR:       {metrics['tpr']:.4f} (target: >= 0.95) {'PASS' if metrics['targets']['tpr_met'] else 'FAIL'}")
    print(f"  FPR:       {metrics['fpr']:.4f} (target: <= 0.02) {'PASS' if metrics['targets']['fpr_met'] else 'FAIL'}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Latency p95: {metrics['latency']['p95_ms']:.1f}ms (target: <= 100ms) {'PASS' if metrics['targets']['latency_met'] else 'FAIL'}")
    print("=" * 60)

    # Detailed results
    print("\nDetailed Predictions:")
    for i, result in enumerate(results):
        label = "PHISH" if labels[i] == 1 else "LEGIT"
        pred = "PHISH" if result["prediction"] == 1 else "LEGIT"
        correct = "OK" if labels[i] == result["prediction"] else "MISS"
        print(f"  [{correct:4s}] True={label} Pred={pred} P={result['phishing_probability']:.3f} | {result['text_preview']}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": model_dir,
        "metrics": metrics,
        "predictions": results,
    }

    report_path = os.path.join(output_dir, f"validation_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also save latest metrics for drift detection
    latest_path = os.path.join(output_dir, "latest_metrics.json")
    with open(latest_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nReport saved to {report_path}")
    return metrics


def print_ci_summary(metrics: Dict):
    """Print parseable one-line summary for CI."""
    tpr = metrics.get("tpr", 0)
    fpr = metrics.get("fpr", 0)
    f1 = metrics.get("f1", 0)
    p95 = metrics.get("latency", {}).get("p95_ms", 0)
    passed = (
        metrics.get("targets", {}).get("tpr_met", False)
        and metrics.get("targets", {}).get("fpr_met", False)
    )
    print(
        f"TPR={tpr:.4f} FPR={fpr:.4f} F1={f1:.4f} p95_ms={p95:.1f} PASS={passed}"
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate phishing detection models")
    parser.add_argument("--model-dir", default="models/phishing-detector")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--api-url", default=None, help="Validate via detection API instead of local model")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key for detection API")
    parser.add_argument("--ci", action="store_true", help="CI mode: parseable output, exit 1 if targets not met")
    args = parser.parse_args()

    if args.api_url:
        metrics = validate_via_api(args.api_url, args.api_key, args.output)
        print("\n" + "=" * 60)
        print("VALIDATION REPORT (via API)")
        print("=" * 60)
        print(f"  TPR: {metrics['tpr']:.4f} (target: >= 0.95) {'PASS' if metrics['targets']['tpr_met'] else 'FAIL'}")
        print(f"  FPR: {metrics['fpr']:.4f} (target: <= 0.02) {'PASS' if metrics['targets']['fpr_met'] else 'FAIL'}")
        print(f"  F1:  {metrics['f1']:.4f}")
        print(f"  Latency p95: {metrics['latency']['p95_ms']:.1f}ms")
        print("=" * 60)
        if args.ci or os.getenv("CI"):
            print_ci_summary(metrics)
    else:
        metrics = validate(args.model_dir, args.output, args.device)
        if args.ci or os.getenv("CI"):
            print_ci_summary(metrics)
