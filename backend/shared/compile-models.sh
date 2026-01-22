#!/bin/bash

#
# Compile TypeORM Models Script
# 
# Compiles all TypeScript model files to JavaScript
# Run this after adding new models or making changes
#

echo "🔧 Compiling TypeORM Models..."
echo ""

cd "$(dirname "$0")/database/models" || exit 1

# Count files
TS_COUNT=$(ls -1 *.ts 2>/dev/null | wc -l)
echo "📋 Found $TS_COUNT TypeScript model files"
echo ""

# Compile each TypeScript file
for file in *.ts; do
    if [ -f "$file" ]; then
        echo "   Compiling $file..."
        npx tsc "$file" \
            --target ES2017 \
            --module commonjs \
            --experimentalDecorators \
            --emitDecoratorMetadata \
            --esModuleInterop \
            --skipLibCheck \
            2>&1 | grep -v "^$" || true
    fi
done

echo ""

# Verify compilation
JS_COUNT=$(ls -1 *.js 2>/dev/null | wc -l)
echo "✅ Compiled $JS_COUNT JavaScript files"

if [ "$TS_COUNT" -eq "$JS_COUNT" ]; then
    echo "✅ All models compiled successfully!"
    echo ""
    echo "📁 Models directory:"
    ls -1 *.js | sed 's/^/   ✓ /'
    exit 0
else
    echo "⚠️  Warning: Not all models were compiled"
    echo "   TypeScript files: $TS_COUNT"
    echo "   JavaScript files: $JS_COUNT"
    exit 1
fi
