#!/bin/bash
# Define pod names in an array.
pods=(deepfashion1 deepfashion2 deepfashion3 deepfashion4 deepfashion5 deepfashion6 deepfashion7)

counter=1
for pod in "${pods[@]}"; do
  local_filename="deepfashion_small_batch.py"
  echo "Copying script from $local_filename to $pod"
  kubectl cp "$local_filename" "$pod":/home/harry/ResNeSt/deepfashion_small_batch.py
done