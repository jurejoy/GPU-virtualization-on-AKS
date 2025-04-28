#!/bin/bash
# Record start time
start_time=$(date +%s)

# Define pod names in an array.
pods=(deepfashion1 deepfashion2 deepfashion3 deepfashion4 deepfashion5 deepfashion6 deepfashion7)

# Step 1: Run the commands in each pod simultaneously.
echo "Starting command execution in pods..."
for pod in "${pods[@]}"; do
  echo "Starting the Python command in $pod..."
  # Execute the command inside the pod in the background.
  kubectl exec "$pod" -- bash -c "python3 deepfashion_batch.py > output_MIG.txt 2>&1" &
  echo "Command started on $pod."
done

# Wait for all background processes to complete
echo "Waiting for all pods to complete their work..."
wait
echo "Command execution completed on all pods."

# Calculate and print the execution time after Step 1
end_time_step1=$(date +%s)
duration_step1=$((end_time_step1 - start_time))
hours=$((duration_step1 / 3600))
minutes=$(( (duration_step1 % 3600) / 60 ))
seconds=$((duration_step1 % 60))
echo "----------------------------------------"
echo "Total time for running commands: ${hours}h ${minutes}m ${seconds}s"
echo "----------------------------------------"

# Step 2: Copy out the log files from each pod.
echo "Retrieving log files..."
counter=1
for pod in "${pods[@]}"; do
  local_filename="output${counter}_MIG.txt"
  echo "Copying log from $pod to local file: $local_filename"
  kubectl cp "$pod":/home/harry/ResNeSt/output_MIG.txt "$local_filename"
  ((counter++))
done

echo "All log files have been retrieved."