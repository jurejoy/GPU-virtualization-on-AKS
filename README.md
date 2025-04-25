# Ray inference and training on Azure Kubernetes services #


## Environment Prepartion ##
## 1. Create AKS Cluster Node pool for existing CPU cluster ##
```bash
az aks nodepool add \
    --resource-group <rg name> \
    --cluster-name <cluster name> \
    --name <nodepool name> \
    --node-count 1 \
    --node-vm-size Standard_NC24ads_A100_v4 \
    --node-taints sku=gpu:NoSchedule \
    --priority Spot \
    --eviction-policy Delete \
    --spot-max-price -1
```

## 2. Install nvidia device plugin for this AKS cluster w/ spot GPU instanace

*notify to add special spot tolerance in the yaml file.*
 ```bash
kubectl apply -f nvidia-device-plugin-ds.yaml
```

## 3. build a docker image based anyscale/ray-llm:2.44.0-py311-cu124 ##
*MacOS example*
```
docker buildx build --platform linux/amd64 -t my-ray-llm:latest .
```
<br>
<br>
<br>

## Serve LLM on Ray cluster ##

## 1. install kuberay operator ##
*notify to add special spot tolerance in the yaml file.*
```bash
helm install kuberay-operator kuberay/kuberay-operator -f values.yaml
```

## 2. Create a secret for huggingface token ##
```bash
kubectl create secret generic huggingface-secret --from-literal=HUGGING_FACE_HUB_TOKEN=<your token>
```

## 3.  Create the ray cluster ##
*change the worknodes replicat if you would like to deploy 2 or more workes to distribute the load.*
```bash
kubernetes apply -f raycluster.yaml
```

## 4. Access the head node and run the job ##
* Submit job for serving LLM on 1 worker node
```
kubectl cp inline_deploy_1node.py <head node pod>:/<working directory>/inline_deploy_1node.py
kubectl exec -it <head node pod> -- /bin/bash
python3 inline_deploy_1node.py
```

* Submit job for serving LLM on 2 worker node without Tensor distributed
```
kubectl cp inline_deploy_2node.py <head node pod>:/<working directory>/inline_deploy_2node.py
kubectl exec -it <head node pod> -- /bin/bash
python3 inline_deploy_2node.py
```

* Submit job for serving LLM on 2 worker node with Tensor distributed
```
kubectl cp inline_deploy_distributed2node.py <head node pod>:/<working directory>/inline_deploy_distributed2node.py
kubectl exec -it <head node pod> -- /bin/bash
python3 inline_deploy_distributed2node.py
```

## 5. Test the serving LLM  ##
```
kubectl port-forward <head node pod> 8265:8265 10001:10001 8000:8000

curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_tokens": 100}'

```
<br>
<br>
<br>

## Training/Finetune the LLM on Ray clsuter ##
## 1. install kuberay operator ##
*notify to add special spot tolerance in the yaml file.*
```bash
helm install kuberay-operator kuberay/kuberay-operator -f values.yaml
```

## 2. Create a secret for huggingface token ##
```bash
kubectl create secret generic huggingface-secret --from-literal=HUGGING_FACE_HUB_TOKEN=<your token>
```

## 3.  Create the ray cluster ##
*change the worknodes replicat if you would like to deploy 2 or more workes to distribute the load.*
```bash
kubernetes apply -f rayclusterFT.yaml
```
or 2 node Raycluster configuration
```bash
kubernetes apply -f rayclusterFT2Node.yaml
```

## 4. Access the head node and run the job ##
* Submit job for serving LLM on 1 worker node
```
kubectl cp run_llama_ft_1node.sh <head node pod>:/<working directory>/run_llama_ft_1node.sh
kubectl cp finetune_hf_llm_1node.py <head node pod>:/<working directory>/finetune_hf_llm_1node.py
kubectl exec -it <head node pod> -- /bin/bash
./run_llama_ft_1node.sh  --size=7b  -nd 1 --lora
```
<br>
<br>

* Submit job for serving LLM on 1 worker node with resume checkpoint
```
kubectl cp run_llama_ft_1node_resume.sh <head node pod>:/<working directory>/run_llama_ft_1node_resume.sh
kubectl cp finetune_hf_llm_1node_resume.py <head node pod>:/<working directory>/finetune_hf_llm_1node_resume.py
kubectl exec -it <head node pod> -- /bin/bash
./run_llama_ft_1node_resume.sh  --size=7b  -nd 1 --lora
# get the checkpoint and start train with it.
./run_llama_ft_1node_resume.sh --size=7b --resume-from-checkpoint=/<directory for checkpoint>/checkpoint_000000 -nd 1 --lora
```
<br>
<br>

* Submit job for serving LLM on 2 worker node with Tensor distributed
```
kubectl cp run_llama_ft_2node.sh <head node pod>:/<working directory>/run_llama_ft_2node.sh
kubectl cp finetune_hf_llm_2node.py <head node pod>:/<working directory>/finetune_hf_llm_2node.py
kubectl exec -it <head node pod> -- /bin/bash
./run_llama_ft_2node.sh  --size=7b  -nd 2 --lora
```

## Serve Stable Diffustion on Ray cluster ##

## 1. install kuberay operator ##
*notify to add special spot tolerance in the yaml file.*
```bash
helm install kuberay-operator kuberay/kuberay-operator -f values.yaml
```

## 2. Create a secret for huggingface token ##
```bash
kubectl create secret generic huggingface-secret --from-literal=HUGGING_FACE_HUB_TOKEN=<your token>
```

## 3.  Create the ray cluster ##
*change the worknodes replicat if you would like to deploy 2 or more workes to distribute the load.*
```bash
kubernetes apply -f raycluster.yaml
```

## 4. Access the head node and run the job ##
* Submit job for serving SD on 1 worker node
```
kubectl cp stable_diffusion.py <head node pod>:/home/ray/stable_diffusion.py
kubectl exec -it <head node pod> -- /bin/bash
python stable_diffusion.py
serve run stable_diffusion:entrypoint
```

## 5. Forward the port to local devices(laptop/desktop) ##
```
k port-forward <head node pod> 8265:8265 10001:10001 8000:8000
```

## 6. Run the script to get SD generated images in parrelly ##
```
python test.py
```

## Serve LLM intergrated with MIG ##

## 1. Create AKS Cluster Node pool for existing CPU cluster ##
```bash
az aks nodepool add \
  --resource-group mig \
  --cluster-name mig \
  --name gpu \
  --node-count 1 \
  --node-vm-size Standard_NC24ads_A100_v4 \
  --node-taints sku=gpu:NoSchedule \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1
```

# Note: GPU drivers are pre-installed on Azure GPU VMs. The driver installation is skipped
# in the GPU Operator configuration in step 2 below with the `--set driver.enabled=false` parameter

## 2. install node-feature-discovery and GPU Operator ##
```
helm install --wait --create-namespace -n gpu-operator node-feature-discovery node-feature-discovery --create-namespace --repo https://kubernetes-sigs.github.io/node-feature-discovery/charts --set-json master.config.extraLabelNs='["nvidia.com"]' --set-json worker.tolerations='[{ "effect": "NoSchedule", "key": "sku", "operator": "Equal", "value": "gpu"},{"effect": "NoSchedule", "key": "mig", "value":"notReady", "operator": "Equal"},{"effect": "NoSchedule", "key": "kubernetes.azure.com/scalesetpriority", "operator": "Equal", "value": "spot"}]'



helm upgrade --install --wait gpu-operator -n gpu-operator --create-namespace nvidia/gpu-operator --version=v25.3.0  --set-json daemonsets.tolerations='[{ "effect": "NoSchedule", "key": "sku", "operator": "Equal", "value": "gpu"},{"effect": "NoSchedule", "key": "kubernetes.azure.com/scalesetpriority", "operator": "Equal", "value": "spot"}]' --set nfd.enabled=false --set driver.enabled=true --set operator.runtimeClass=nvidia-container-runtime --set migManager.enabled=true \
  --set mig.strategy=mixed \
  --set migManager.config=all-1g.10gb
```

# Enable MIG to Single mode #
kubectl patch clusterpolicies.nvidia.com/cluster-policy --type='json' -p='[{"op":"replace","path":"/spec/mig/strategy","value":"single"}]'

# Force MIG configuration with reboot allowed
```
kubectl patch clusterpolicy/cluster-policy -n gpu-operator --type merge -p '{"spec": {"migManager": {"enabled": true, "env": [{"name": "WITH_REBOOT", "value": "true"}]}}}'
```

# Verify the cluster policy has MIG Manager enabled
```
kubectl get clusterpolicy -n gpu-operator cluster-policy -o yaml | grep -A 10 migManager
```

# Apply MIG configuration to the nodepool
```
az aks nodepool update --cluster-name mig --resource-group mig --name gpu --labels "nvidia.com/mig.config=all-1g.10gb"
```
