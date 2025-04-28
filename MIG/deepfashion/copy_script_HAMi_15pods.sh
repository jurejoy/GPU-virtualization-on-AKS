#!/bin/bash
# Define pod names in an array.
pods=(deepfashion1 deepfashion2 deepfashion3 deepfashion4 deepfashion5 deepfashion6 deepfashion7 deepfashion8 deepfashion9 deepfashion10 deepfashion11 deepfashion12 deepfashion13 deepfashion14 deepfashion15)

counter=1
for pod in "${pods[@]}"; do
  local_filename="deepfashion_batch.py"
  echo "Copying script from $local_filename to $pod"
  kubectl cp "$local_filename" "$pod":/home/harry/ResNeSt/deepfashion_batch.py
done