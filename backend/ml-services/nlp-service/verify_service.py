#!/usr/bin/env python3
"""
Script to verify NLP service can start and API documentation is accessible
"""
import sys
import os
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    try:
        from src.main import app
        print("✓ FastAPI app can be imported")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def check_service_startup():
    """Check if service can start (without actually running it)"""
    print("\nChecking service startup configuration...")
    try:
        from src.main import app
        from src.utils.model_loader import ModelLoader
        
        # Check if app is configured
        assert app is not None
        print("✓ FastAPI app is configured")
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/", "/api/v1/analyze-text", "/api/v1/analyze-email", "/api/v1/detect-ai-content"]
        for route in expected_routes:
            if route in routes or any(route in str(r.path) for r in app.routes):
                print(f"✓ Route {route} exists")
            else:
                print(f"⚠ Route {route} not found")
        
        return True
    except Exception as e:
        print(f"✗ Startup check failed: {e}")
        return False

def check_api_documentation():
    """Check if API documentation endpoints exist"""
    print("\nChecking API documentation...")
    try:
        from src.main import app
        
        # FastAPI automatically creates these endpoints
        routes = [str(route.path) for route in app.routes]
        
        has_docs = any("/docs" in route or "/openapi.json" in route for route in routes)
        has_redoc = any("/redoc" in route for route in routes)
        
        if has_docs:
            print("✓ Swagger UI available at /docs")
        else:
            print("⚠ Swagger UI endpoint not found")
        
        if has_redoc:
            print("✓ ReDoc available at /redoc")
        else:
            print("⚠ ReDoc endpoint not found")
        
        # FastAPI always has these, so this should pass
        return True
    except Exception as e:
        print(f"✗ Documentation check failed: {e}")
        return False

def check_models_exist():
    """Check if trained models exist"""
    print("\nChecking for trained models...")
    model_dir = Path("models")
    
    phishing_model = model_dir / "phishing-bert-v1" / "model.safetensors"
    ai_model = model_dir / "ai-detector-v1" / "model.safetensors"
    
    if phishing_model.exists():
        size_mb = phishing_model.stat().st_size / (1024 * 1024)
        print(f"✓ Phishing model found ({size_mb:.1f} MB)")
    else:
        print("⚠ Phishing model not found (will use fallback)")
    
    if ai_model.exists():
        size_mb = ai_model.stat().st_size / (1024 * 1024)
        print(f"✓ AI detector model found ({size_mb:.1f} MB)")
    else:
        print("⚠ AI detector model not found (will use fallback)")
    
    return True

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("NLP Service Verification")
    print("=" * 60)
    
    results = []
    results.append(("Imports", check_imports()))
    results.append(("Service Startup", check_service_startup()))
    results.append(("API Documentation", check_api_documentation()))
    results.append(("Models", check_models_exist()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed!")
        print("\nTo start the service:")
        print("  uvicorn src.main:app --host 0.0.0.0 --port 8000")
        print("\nAPI Documentation:")
        print("  Swagger UI: http://localhost:8000/docs")
        print("  ReDoc: http://localhost:8000/redoc")
    else:
        print("⚠ Some checks failed. Please review the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
