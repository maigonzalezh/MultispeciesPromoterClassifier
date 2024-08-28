# MultispeciesPromoterClassifier

This repository contains Python scripts that run in Docker containers using `docker-compose` for orchestration.

## Prerequisites

- A Linux-based operating system (tested in Ubuntu 22.04)
- [Docker](https://www.docker.com/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

## Project structure


## Scripts

### Dataset generation

For generation of dataset, execute the following command:

```
docker compose up dataset-build
```

### Hyperparameter tuning
