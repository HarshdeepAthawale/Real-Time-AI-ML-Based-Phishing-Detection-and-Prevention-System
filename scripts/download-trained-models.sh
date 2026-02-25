#!/bin/bash
# =============================================================================
# Download Trained Models from Google Drive / Local ZIP
# =============================================================================
# This script places Colab-trained models into the correct service directories.
#
# Usage:
#   ./scripts/download-trained-models.sh                    # Interactive prompts
#   ./scripts/download-trained-models.sh --from-dir /path   # From local directory
#
# Expected input structure (zips or directories):
#   phishing-detector/       → nlp-service/models/phishing-detector/
#   ai-detector/             → nlp-service/models/ai-detector/
#   gnn-domain-classifier/   → url-service/models/gnn-domain-classifier/
#   brand-classifier/        → visual-service/models/brand-classifier/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ML_SERVICES="$PROJECT_ROOT/backend/ml-services"

# Model directories
NLP_MODELS="$ML_SERVICES/nlp-service/models"
URL_MODELS="$ML_SERVICES/url-service/models"
VISUAL_MODELS="$ML_SERVICES/visual-service/models"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create model directories
mkdir -p "$NLP_MODELS/phishing-detector"
mkdir -p "$NLP_MODELS/ai-detector"
mkdir -p "$URL_MODELS/gnn-domain-classifier"
mkdir -p "$VISUAL_MODELS/brand-classifier"

install_model() {
    local src="$1"
    local dest="$2"
    local name="$3"

    if [ -d "$src" ]; then
        log_info "Installing $name from directory: $src"
        cp -r "$src"/* "$dest/"
        log_info "$name installed successfully"
        return 0
    elif [ -f "$src" ] && [[ "$src" == *.zip ]]; then
        log_info "Installing $name from zip: $src"
        unzip -o "$src" -d /tmp/model_extract > /dev/null 2>&1
        # Handle both flat and nested zip structures
        if [ -d "/tmp/model_extract/$name" ]; then
            cp -r "/tmp/model_extract/$name"/* "$dest/"
        else
            cp -r /tmp/model_extract/* "$dest/"
        fi
        rm -rf /tmp/model_extract
        log_info "$name installed successfully"
        return 0
    else
        log_warn "$name not found at $src (skipping)"
        return 1
    fi
}

# Parse args
FROM_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --from-dir)
            FROM_DIR="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--from-dir /path/to/models]"
            exit 1
            ;;
    esac
done

if [ -z "$FROM_DIR" ]; then
    echo "============================================"
    echo "  Trained Model Installer"
    echo "============================================"
    echo ""
    echo "Enter the path to the directory containing your"
    echo "trained model folders (or zip files):"
    echo ""
    echo "Expected contents:"
    echo "  phishing-detector/   (or phishing-detector.zip)"
    echo "  ai-detector/         (or ai-detector.zip)"
    echo "  gnn-domain-classifier/ (or gnn-domain-classifier.zip)"
    echo "  brand-classifier/    (or brand-classifier.zip)"
    echo ""
    read -rp "Path: " FROM_DIR
fi

if [ ! -d "$FROM_DIR" ]; then
    log_error "Directory not found: $FROM_DIR"
    exit 1
fi

echo ""
log_info "Installing models from: $FROM_DIR"
echo ""

installed=0
failed=0

# NLP: Phishing Detector
for src in "$FROM_DIR/phishing-detector" "$FROM_DIR/phishing-detector.zip"; do
    if install_model "$src" "$NLP_MODELS/phishing-detector" "phishing-detector" 2>/dev/null; then
        ((installed++))
        break
    fi
done || ((failed++))

# NLP: AI Detector
for src in "$FROM_DIR/ai-detector" "$FROM_DIR/ai-detector.zip"; do
    if install_model "$src" "$NLP_MODELS/ai-detector" "ai-detector" 2>/dev/null; then
        ((installed++))
        break
    fi
done || ((failed++))

# URL: GNN Domain Classifier
for src in "$FROM_DIR/gnn-domain-classifier" "$FROM_DIR/gnn-domain-classifier.zip"; do
    if install_model "$src" "$URL_MODELS/gnn-domain-classifier" "gnn-domain-classifier" 2>/dev/null; then
        ((installed++))
        break
    fi
done || ((failed++))

# Visual: Brand Classifier
for src in "$FROM_DIR/brand-classifier" "$FROM_DIR/brand-classifier.zip"; do
    if install_model "$src" "$VISUAL_MODELS/brand-classifier" "brand-classifier" 2>/dev/null; then
        ((installed++))
        break
    fi
done || ((failed++))

echo ""
echo "============================================"
log_info "Installed: $installed models"
if [ $failed -gt 0 ]; then
    log_warn "Missing: $failed models (run Colab notebooks to train them)"
fi
echo "============================================"

# Verify installed models
echo ""
log_info "Verification:"

check_model() {
    local dir="$1"
    local name="$2"
    local required_file="$3"

    if [ -f "$dir/$required_file" ]; then
        local size
        size=$(du -sh "$dir/$required_file" | cut -f1)
        echo -e "  ${GREEN}OK${NC}  $name ($size)"
    else
        echo -e "  ${RED}MISSING${NC}  $name (no $required_file)"
    fi
}

check_model "$NLP_MODELS/phishing-detector" "Phishing Classifier" "model.safetensors"
check_model "$NLP_MODELS/ai-detector" "AI Detector" "model.safetensors"
check_model "$URL_MODELS/gnn-domain-classifier" "GNN URL Classifier" "model.pth"
check_model "$VISUAL_MODELS/brand-classifier" "CNN Visual Classifier" "model.pth"
