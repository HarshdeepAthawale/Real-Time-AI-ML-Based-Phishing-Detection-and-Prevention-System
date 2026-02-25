#!/usr/bin/env python3
"""
Quick sanity check that both NLP models (phishing classifier + AI detector) load and run.
Run from nlp-service dir: python scripts/test_both_models.py
"""
import sys
import os

# Ensure src is on path when run from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector


def main():
    print("=" * 60)
    print("NLP Service – testing both models")
    print("=" * 60)

    phishing = None
    ai_detector = None

    # 1. Phishing classifier
    print("\n[1] Phishing classifier")
    print(f"    Path: {settings.phishing_model_path}")
    print(f"    Exists: {os.path.isdir(settings.phishing_model_path)}")
    try:
        phishing = PhishingClassifier(
            model_path=settings.phishing_model_path,
            device=settings.inference_device,
        )
        if phishing.model is None:
            print("    Status: NOT LOADED (path missing or load failed)")
        else:
            out = phishing.predict(
                "Urgent! Your account will be suspended. Click here to verify now."
            )
            print("    Status: LOADED")
            print(f"    Sample: phishing_prob={out.get('phishing_probability', 0):.3f} prediction={out.get('prediction', '?')}")
    except Exception as e:
        print(f"    Status: ERROR - {e}")

    # 2. AI detector
    print("\n[2] AI detector")
    print(f"    Path: {settings.ai_detector_model_path}")
    print(f"    Exists: {os.path.isdir(settings.ai_detector_model_path)}")
    try:
        ai_detector = AIGeneratedDetector(
            model_path=settings.ai_detector_model_path,
            device=settings.inference_device,
        )
        if ai_detector.model is None:
            print("    Status: NOT LOADED (path missing or load failed)")
        else:
            out = ai_detector.detect(
                "This is a test message to check if the AI detector is working."
            )
            print("    Status: LOADED")
            print(f"    Sample: ai_generated_prob={out.get('ai_generated_probability', 0):.3f} is_ai={out.get('is_ai_generated', False)}")
    except Exception as e:
        print(f"    Status: ERROR - {e}")

    # Summary
    print("\n" + "=" * 60)
    phishing_ok = phishing is not None and phishing.model is not None
    ai_ok = ai_detector is not None and ai_detector.model is not None
    if phishing_ok and ai_ok:
        print("Result: BOTH MODELS WORKING")
        return 0
    if phishing_ok or ai_ok:
        print("Result: ONE MODEL WORKING (see above)")
        return 0
    print("Result: NO MODELS LOADED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
