# Logo Templates Directory

This directory stores logo templates for brand detection and impersonation detection.

## Structure

Place logo template images in this directory with the naming convention:
```
{brand_id}_{brand_name}_logo.png
```

Example:
- `1_amazon_logo.png`
- `2_paypal_logo.png`
- `3_microsoft_logo.png`

## Logo Requirements

- **Format**: PNG (with transparency preferred)
- **Size**: 200x200 to 500x500 pixels
- **Background**: Transparent or white
- **Quality**: High resolution, clear edges

## Loading Templates

Templates can be loaded programmatically:

```python
from src.analyzers.logo_detector import LogoDetector
import os

detector = LogoDetector()

# Load all templates from this directory
logo_dir = 'logo_templates'
for filename in os.listdir(logo_dir):
    if filename.endswith('.png'):
        brand_id = filename.split('_')[0]  # Extract brand ID
        with open(os.path.join(logo_dir, filename), 'rb') as f:
            detector.add_logo_template(brand_id, f.read())
```

## Alternative: S3 Storage

For production deployments, consider storing logos in S3:
- Upload to: `s3://your-bucket/logos/`
- Load from S3 on service startup
- Benefits: Centralized management, versioning, CDN support

## Next Steps

1. Collect logo images for major brands (banks, tech companies, etc.)
2. Process and normalize logos to consistent format
3. Add logos to this directory or S3
4. Update logo loading logic in service startup
