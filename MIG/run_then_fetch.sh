#!/bin/bash
# Define pod names in an array.
pods=(resnest1 resnest2 resnest3 resnest4 resnest5 resnest6 resnest7 resnest8)

# Step 1: Run the commands in each pod simultaneously.
echo "Starting command execution in pods..."
for pod in "${pods[@]}"; do
  echo "Starting the Python command in $pod..."
  # Execute the command inside the pod in the background.
  kubectl exec "$pod" -- bash -c "cd /home/harry/ResNeSt/scripts/torch && python3 verify.py --model resnest50 --crop-size 224 > output.txt 2>&1" &
  echo "Command started on $pod."
done

# Wait for all background processes to complete
echo "Waiting for all pods to complete their work..."
wait
echo "Command execution completed on all pods."

# Step 2: Copy out the log files from each pod.
echo "Retrieving log files..."
counter=1
for pod in "${pods[@]}"; do
  local_filename="output${counter}.txt"
  echo "Copying log from $pod to local file: $local_filename"
  kubectl cp "$pod":/home/harry/ResNeSt/scripts/torch/output.txt "$local_filename"
  ((counter++))
done

echo "All log files have been retrieved."