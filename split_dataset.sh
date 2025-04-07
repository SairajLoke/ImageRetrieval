#!/bin/bash

# Configurable Parameters
IMAGE_DIR="Datasets/wang/Images"
TRAIN_DIR="$IMAGE_DIR/train"
TEST_DIR="$IMAGE_DIR/test"
K=20  # percentage for test split

# Step 1: Create initial class folders
for i in {0..9}; do
    mkdir -p "$IMAGE_DIR/class$i"
done

# Step 2: Distribute images into class folders
for img in "$IMAGE_DIR"/*.jpg; do
    img_name=$(basename "$img")
    img_num="${img_name%%.*}"
    class_idx=$((img_num / 100))
    mv "$img" "$IMAGE_DIR/class$class_idx/"
done

echo "Initial class split complete."

# Step 3: Create train/test class subdirectories
for i in {0..9}; do
    mkdir -p "$TRAIN_DIR/class$i"
    mkdir -p "$TEST_DIR/class$i"

    class_folder="$IMAGE_DIR/class$i"
    files=($(ls "$class_folder"/*.jpg | sort))  # sort for consistency
    total=${#files[@]}
    test_count=$((total * K / 100))
    train_count=$((total - test_count))

    # Move files to train
    for ((j=0; j<train_count; j++)); do
        mv "${files[$j]}" "$TRAIN_DIR/class$i/"
    done

    # Move files to test
    for ((j=train_count; j<total; j++)); do
        mv "${files[$j]}" "$TEST_DIR/class$i/"
    done

    echo "Class $i -> Train: $train_count, Test: $test_count"
done

echo "Train/Test split complete."
