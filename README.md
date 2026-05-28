# Negative dataset selection impacts machine learning-based predictors for multiple bacterial species promoters
This repository contains the dataset and code for the study on how negative dataset selection impacts machine learning-based predictors for promoters in multiple bacterial species.

> **Published in:** *Bioinformatics*, Volume 41, Issue 4, April 2025.
> [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtaf135-blue)](https://doi.org/10.1093/bioinformatics/btaf135)

## Prerequisites

- A Linux-based operating system (tested in Ubuntu 22.04)
- [Docker](https://www.docker.com/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- [Weights & Biases](https://wandb.ai/site) account for experiment tracking
- AWS S3 bucket for storing training checkpoints and logs

## Project structure
- **data/**: This directory contains all the data used in the project, divided into different subfolders.
  - **processed/**: Stores processed data, based on dataset type (CDS/SRS).
  - **raw/**: Holds unprocessed data files originally obtained from the [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/) for CDS strains and the [Prokaryotic Promoter Database](http://lin-group.cn/database/ppd/download.php) for promoters.
  
- **docker/**:
  - **pytorch/**: Includes configurations and scripts necessary to set up and run a PyTorch environment within a Docker container.
  - **tensorflow/**: Contains configurations and scripts to establish a TensorFlow environment in Docker. Additionally, specific entry point scripts are included in the `entrypoints/` subfolder, such as `build_dataset.sh` for dataset generation and `entrypoint.sh` as the main startup script for the container for training/tuning RF and CNN models.

- **notebooks/**: Contains Jupyter notebooks used for exploratory data analysis, and DNABERT analysis (latent space and genoma evaluation)

- **scripts/**: Includes all scripts used in the project, including common scripts, model definitions, data preprocessing, experiment execution, and utility functions.

## Setup

### Environment variables
Create a `.env` file in the root directory with the following environment variables:
- **`WANDB_API_KEY`**: This is your Weights & Biases API key, which is necessary for logging experiments and results to the Weights & Biases platform.

- **`AWS_ACCESS_KEY_ID`**: This is your AWS Access Key ID, necessary for authenticating requests to AWS services (tested with S3).

- **`AWS_SECRET_ACCESS_KEY`**: This is your AWS Secret Access Key, which pairs with the AWS Access Key ID to securely authenticate your AWS requests.
## Scripts

### Dataset generation (optional)
If you want to generate the dataset from scratch, you can run the following command:

```
docker compose up dataset-build
```

### Hyperparameter tuning and training

#### Random Forest and Convolutional Neural Network models
Edit `/docker/tensorflow/entrypoints/entrypoint.sh` based on the model and dataset and then run:

```
docker compose up training
```

#### BERT-based models
Edit `/docker/pytorch/entrypoint.sh` based on the dataset and then run:

```
docker compose up training-bert
```

#### Jupyter notebooks
If you want to run the Jupyter notebooks, execute the following command to start the Jupyter server:
    
```
docker compose up <jupyter-torch|jupyter-tf>
```

## Model Availability  
Trained BERT-based models are available on [Zenodo](https://doi.org/10.5281/zenodo.15016403). An example of their usage can be found in `notebooks/bert_results.ipynb`. The models are provided as `models_cds.zip` (trained on the CDS dataset) and `models_random.zip` (trained on the SRS dataset).  

## How to Cite This Work

If this work has contributed to your research, please consider citing the paper, the software, or both depending on your use:

**Paper:**
```
Marcelo González, Roberto E Durán, Michael Seeger, Mauricio Araya, Nicolás Jara,
Negative dataset selection impacts machine learning-based predictors for multiple
bacterial species promoters, Bioinformatics, Volume 41, Issue 4, April 2025, btaf135,
https://doi.org/10.1093/bioinformatics/btaf135
```

**Software:**
```
Marcelo González, «maigonzalezh/MultispeciesPromoterClassifier: v1.0.0». Zenodo, mar. 13, 2025.
doi: 10.5281/zenodo.15016403.
```

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15016403-blue)](https://doi.org/10.5281/zenodo.15016403)
