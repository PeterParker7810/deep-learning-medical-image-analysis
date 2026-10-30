# Deployment guide

The repository includes a Streamlit configuration and Docker health check. A public deployment is **not active**. An ISIC 2016 Task 3 ResNet-18 checkpoint is trained locally; the NIH and APTOS datasets/checkpoints are not available. Do not deploy a version that implies all three analysis models are available.

## Local container smoke check

```bash
docker build -t medical-image-research .
docker run --rm -p 8501:8501 medical-image-research
```

Open `http://localhost:8501`. The Streamlit health endpoint is `/_stcore/health`. Container smoke checks should verify that endpoint and the dashboard; they do not verify model inference unless a genuine checkpoint is mounted.

## Hosting considerations

- **Streamlit Community Cloud:** convenient for the app, but private repository access, resource limits, and model artifact hosting depend on the owner's account. Large clinical datasets and model checkpoints should not be uploaded to the app repository.
- **Hugging Face Spaces:** supports Streamlit/Docker deployment and model storage, but gated datasets/checkpoints require the owner's account and configured secrets. No credentials or raw datasets are committed; the trained ISIC checkpoint is tracked with this app so its real inference path is available.
- Use an authenticated private artifact store for any trained weights; configure its access through the host's secrets manager. Never add access tokens to source or Docker layers.

Before public deployment, obtain and train on all approved data (or explicitly scope deployment to the available task), set a clear research-only disclaimer, confirm dataset and model licensing, test health and real inference endpoints, and record the verified URL. No public URL is currently claimed.
