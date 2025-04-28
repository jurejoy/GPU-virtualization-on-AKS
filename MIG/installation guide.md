# AKS cluster Preparation #
## 1. install the AKS GPU Node pool ##
```
az aks nodepool add --resource-group resnest --cluster-name resnest --name gpu --node-count 1 --node-vm-size standard_nc40ads_h100_v5
```

## 2. install node-feature-discovery and GPU Operator ##


```
helm upgrade --install --wait gpu-operator -n gpu-operator --create-namespace nvidia/gpu-operator --version=v25.3.0   --set nfd.enabled=true --set driver.enabled=false --set migManager.enabled=true --set operator.runtimeClass=nvidia-container-runtime
```
<br>
<br>
<br>

# MIG Configuration #
## 1. Enable MIG to Single mode ##
```
kubectl patch clusterpolicies.nvidia.com/cluster-policy --type='json' -p='[{"op":"replace","path":"/spec/mig/strategy","value":"single"}]'
```


## 2. Verify the cluster policy has MIG Manager enabled ##
```
kubectl get clusterpolicy -n gpu-operator cluster-policy -o yaml | grep -A 10 migManager
```

## 3. Apply MIG configuration to the nodepool ##
```
az aks nodepool update --cluster-name resnest --resource-group resnest --name gpu --labels "nvidia.com/mig.config=all-1g.12gb"
```

## 4. Verify MIG configuration was applied properly ##
```
kubectl describe node -l agentpool=gpu | grep -i nvidia.com
```

## 5. Disable MIG and roll back to a single GPU instance ##
```
# Remove the MIG configuration label
az aks nodepool update --cluster-name resnest --resource-group resnest --name gpu --labels "nvidia.com/mig.config="

# Patch the cluster policy to disable MIG
kubectl patch clusterpolicies.nvidia.com/cluster-policy --type='json' -p='[{"op":"replace","path":"/spec/mig/strategy","value":"none"}]'

# Wait for the node to update (this may take a few minutes)
kubectl get nodes -l agentpool=gpu -w
```
<br>
<br>
<br>

# HAMi comfiguraton #

## 1. Label the GPU nodepool with gpu=on ##
```
az aks nodepool update --cluster-name resnest --resource-group resnest --name gpu --labels "gpu=on"
```

## 2. install the HAMi Helm Chart ##
```
helm repo add hami-charts https://project-hami.github.io/HAMi/
```

```
helm install hami hami-charts/hami -n kube-system
#optional, if you want to engage dynamic MIG
#helm install hami ./HAMi/charts/hami -n kube-system 
```

## 3. forward the HAMi Monitor pod to local for monitoring ##
```
kubectl port-forward -n kube-system service/hami-device-plugin-monitor 31992:31992
```
<br>
<br>
<br>

# Debug Tools #


## Create a gpu-debug pod. ## 
```
# Get the name of a node with "gpu" in the name
NODE_NAME=$(kubectl get nodes -o name | grep -i gpu | sed 's/node\///' | head -1)


# Create a debug pod to access the node
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-debug
  namespace: default
spec:
  containers:
  - name: nvidia-smi
    image: nvidia/cuda:11.8.0-base-ubuntu22.04
    command: ["sleep", "infinity"]
    securityContext:
      privileged: true
  nodeSelector:
    kubernetes.io/hostname: ${NODE_NAME}
EOF
```

## Check HAMi Mode ##
```
kubectl logs -n kube-system $(kubectl get pods -n kube-system -l app.kubernetes.io/component=hami-device-plugin -o jsonpath='{.items[0].metadata.name}') | grep -i "MIG"
````
<br>
<br>
<br>

# Build/Tag/Push Docker image #

## 1. build the docker image ##
copy all the image folder like "Anno_coarse","Anno_fine","img"(extract from img.zip)
```
cd MIG/deepfashion
docker build -t deepfashion-gpu-image -f dockerfile .
```


## 2. run the docker image and test it ##
docker run -it --rm --gpus all deepfashion-gpu-image


## 3. install Nvidia container Toolkit(optional) ##
* If you encounter: "Error response from daemon: could not select device driver "" with capabilities: [[gpu]]"
You need to install the NVIDIA Container Toolkit:


* Add the NVIDIA repository
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
* Install nvidia-docker2
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

docker run -it --rm --gpus all --shm-size=32g deepfashion-gpu-image
```


## 4. Tag and push to acr ##
```
# Tag the docker image
docker tag deepfashion-gpu-image:latest renjie.azurecr.io/deepfashion-gpu-image:latest
# Push to ACR
docker push renjie.azurecr.io/deepfashion-gpu-image:latest
```
<br>
<br>
<br>

# Run the Benchmark #

## Run Hami 8 pods Benchmark ##
```
# Start all the pods share 1 physical GPU, each with 12% resource
kubectl apply -f deepfashion_sharing_card.yaml
# copy the deepfashion_batch.py to all the pods
./copy_script.sh
# run the deepfashion_batch.py in parallel on all pods
./run_then_fetch.sh
```

## Run MIG 7 pods Benchmark ##
```
# follow MIG configuration, set to "all-1g.12gb" ###
# copy the deepfashion_batch.py to all the pods
./copy_script_MIG.sh
# run the deepfashion_batch.py in parallel on all pods
./run_then_fetch_MIG.sh
```