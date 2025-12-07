#!/bin/bash
# Basic test script for Lament Engine

echo "Running basic tests..."

# Test 1: Check if binary exists
if [ ! -f "./lament" ] && [ ! -f "./lament.exe" ]; then
    echo "❌ Test 1 FAILED: Binary not found. Run 'make' first."
    exit 1
fi
echo "✅ Test 1 PASSED: Binary exists"

# Test 2: Check help output
if ./lament --help 2>&1 | grep -q "Usage"; then
    echo "✅ Test 2 PASSED: Help command works"
else
    echo "❌ Test 2 FAILED: Help command failed"
    exit 1
fi

# Test 3: Check mode parsing
MODES=("witness" "judge" "rebuilder" "silence")
for mode in "${MODES[@]}"; do
    if ./lament --mode "$mode" --help 2>&1 > /dev/null; then
        echo "✅ Test 3 PASSED: Mode '$mode' accepted"
    else
        echo "❌ Test 3 FAILED: Mode '$mode' rejected"
        exit 1
    fi
done

echo "All basic tests passed!"

