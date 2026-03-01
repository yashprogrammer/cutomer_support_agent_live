# Deploy to AWS EC2 with GitHub Actions (Simple CI/CD)

This is a simple SSH-based deployment pipeline for this project.

- CI: runs tests on pull requests and non-main pushes.
- CD: on `main` (or manual trigger), uploads source to EC2 and runs `docker compose up -d --build`.

## 1) One-time EC2 setup

Assumptions:

- Ubuntu EC2 instance
- Ports open in security group:
  - `22` (SSH)
  - `8000` (FastAPI)
  - `8501` (Streamlit)

Run on EC2:

```bash
sudo apt-get update
# Ubuntu 24.04 repo package name is usually docker-compose-v2.
# (docker-compose-plugin may not exist in the default Ubuntu repo)
sudo apt-get install -y docker.io docker-compose-v2 curl

# verify install
docker --version
docker compose version

sudo usermod -aG docker $USER
newgrp docker

sudo mkdir -p /opt/customer_support_agent
sudo chown -R $USER:$USER /opt/customer_support_agent
```

If `docker compose version` still fails, install Docker from the official Docker apt repo (which provides `docker-compose-plugin`):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

docker --version
docker compose version
```

Create runtime env file once:

```bash
cat > /opt/customer_support_agent/.env <<'EOF_ENV'
GROQ_API_KEY=your_real_key
GROQ_MODEL=openai/gpt-oss-20b
API_BASE_URL=http://localhost:8000
EOF_ENV
```

Optional: instead of creating `.env` manually, set GitHub secret `EC2_ENV_FILE` with full `.env` content, and add repository variable `INJECT_ENV_FILE` = `true` (Settings → Variables).

## 2) GitHub Actions workflows added

- `/.github/workflows/ci.yml`
  - triggers: `pull_request`, push to non-main branches
  - steps: checkout -> setup python+uv -> `uv sync --dev` -> `uv run pytest -q`

- `/.github/workflows/deploy-ec2.yml`
  - triggers: `push` to `main`, manual `workflow_dispatch`
  - jobs:
    1. test (same as CI)
    2. deploy (needs test)
  - deploy steps:
    - optional `.env` injection from secret
    - SSH key setup
    - package release tar
    - upload tar to EC2 (`scp`)
    - extract to app dir and run `docker compose up -d --build --remove-orphans`
    - health check via `curl http://127.0.0.1:8000/health`

## 3) Required GitHub secrets

Set in repository settings -> Secrets and variables -> Actions:

- `EC2_HOST` : public IP/DNS of EC2
- `EC2_USER` : SSH user (usually `ubuntu`)
- `EC2_SSH_KEY` : private key content for EC2 access

Optional:

- `EC2_PORT` : default `22`
- `EC2_APP_DIR` : default `/opt/customer_support_agent`
- `EC2_ENV_FILE` : full `.env` file content (multi-line). When using this, also add variable `INJECT_ENV_FILE` = `true`.

## 4) Deployment flow

1. Merge/push changes to `main`.
2. GitHub Actions runs tests.
3. If tests pass, deploy job uploads project and restarts containers on EC2.
4. Verify:

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
# expected: {"status":"ok"}
```

Open UI:

- FastAPI docs: `http://<EC2_PUBLIC_IP>:8000/docs`
- Streamlit: `http://<EC2_PUBLIC_IP>:8501`

## 5) Notes

- This is intentionally simple (SSH + docker compose).
- For stronger production posture later:
  - add Nginx + TLS (Let's Encrypt)
  - move secrets to AWS SSM/Secrets Manager
  - use GHCR images instead of source upload
  - add rollback strategy and blue/green deploy
