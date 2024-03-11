# Deploy CodeActAgent with Kubernetes

## Prepare containers for different components

You either need to setup your own docker registry (similar to [this](https://www.paulsblog.dev/how-to-install-a-private-docker-container-registry-in-kubernetes/)), or just use docker hub. I will be using dockerhub for example.

```bash
# This will build Chat-UI into docker container and push to `xingyaoww/chat-ui` on docker.io
# Your should replace `xingyaoww/xxx` with your dockerhub/private registry path!
./scripts/chat/kubernetes/build_chat_ui_and_push.sh xingyaoww/chat-ui
# Similarily:
./scripts/chat/kubernetes/build_code_execute_api_and_push.sh xingyaoww/codeact-execute-api
./scripts/chat/kubernetes/build_code_executor_container_and_push.sh xingyaoww/codeact-executor
```

## Modify K8S configuration file

```bash
cp scripts/chat/kubernetes/k8s.template.yml scripts/chat/kubernetes/k8s.yml
```

Search for `TODO_USER_REPLACE` in `scripts/chat/kubernetes/k8s.yml` and replace each to your specific value. Raise an issue if you have any questions!

## Start ALL service!

```bash
# You may need to run this as root
kubectl apply -f scripts/chat/kubernetes/k8s.yml
```
