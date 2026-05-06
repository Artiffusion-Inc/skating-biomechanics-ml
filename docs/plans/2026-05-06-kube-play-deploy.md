# Podman Kube Play Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace podman-compose v1.3.0 deploy with `podman kube play --replace` using clean Kubernetes YAML, eliminating --force-recreate hacks.

**Architecture:** Write `infra/deploy.yaml` (Kubernetes Pod spec) that mirrors `compose.prod.yaml`. Workflow deploys via `podman kube play --replace infra/deploy.yaml` with `--env-file` substitution. Migrations run via `podman exec` into the backend container. Health check polls pod status.

**Tech Stack:** Podman 5.4.2, Kubernetes YAML, GitHub Actions, systemd-less deploy.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `infra/deploy.yaml` | Kubernetes-style Pod spec for all services. Single source of truth for production deploy. |
| `infra/compose.prod.yaml` | Local dev / backup reference. Kept but not used by deploy workflow. |
| `.github/workflows/deploy.yml` | Updated to use `podman kube play --replace` instead of podman-compose. |

---

## Task 1: Write `infra/deploy.yaml`

**Files:**
- Create: `infra/deploy.yaml`
- Reference: `infra/compose.prod.yaml`

**Rationale:** `podman generate kube` produces unreadable output with hardcoded annotations. Hand-written YAML is cleaner and maintainable.

- [ ] **Step 1: Create `infra/deploy.yaml`**

```yaml
# Production deploy spec for podman kube play --replace
# Mirrors compose.prod.yaml in Kubernetes Pod format
apiVersion: v1
kind: Pod
metadata:
  name: skating-app
  labels:
    app: skating-app
spec:
  restartPolicy: Always
  containers:
    - name: postgres
      image: docker.io/library/postgres:16-alpine
      env:
        - name: POSTGRES_DB
          value: "skating"
        - name: POSTGRES_USER
          value: "skating"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: postgres-password
      volumeMounts:
        - mountPath: /var/lib/postgresql/data
          name: postgres-data
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U skating"]
        interval: 10s
        timeout: 3s
        retries: 5
        startPeriod: 10s

    - name: valkey
      image: docker.io/valkey/valkey:8-alpine
      volumeMounts:
        - mountPath: /data
          name: valkey-data
      healthcheck:
        test: ["CMD", "valkey-cli", "ping"]
        interval: 10s
        timeout: 3s
        retries: 5
        startPeriod: 10s

    - name: backend
      image: ghcr.io/artiffusion-inc/skating-backend:latest
      env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: database-url
        - name: VALKEY_URL
          value: "redis://127.0.0.1:6379/0"
        - name: R2_ENDPOINT_URL
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: r2-endpoint-url
        - name: R2_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: r2-access-key-id
        - name: R2_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: r2-secret-access-key
        - name: R2_BUCKET
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: r2-bucket
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: jwt-secret
        - name: VASTAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: vastai-api-key
        - name: LOG_LEVEL
          value: "info"
      healthcheck:
        test: ["CMD-SHELL", "bash -c 'python -c \"import urllib.request; urllib.request.urlopen(\\\"http://127.0.0.1:8000/api/v1/health\\\", timeout=2)\"'"]
        interval: 30s
        timeout: 3s
        retries: 3
        startPeriod: 15s

    - name: frontend
      image: ghcr.io/artiffusion-inc/skating-frontend:latest
      env:
        - name: NEXT_PUBLIC_API_URL
          value: "/api"
      healthcheck:
        test: ["CMD", "wget", "-q", "--spider", "http://127.0.0.1:3000/"]
        interval: 30s
        timeout: 3s
        retries: 3
        startPeriod: 10s

    - name: caddy
      image: ghcr.io/caddybuilds/caddy-cloudflare:latest
      env:
        - name: CLOUDFLARE_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: skating-secrets
              key: cloudflare-api-token
        - name: DOMAIN
          value: "bbank.pro"
      volumeMounts:
        - mountPath: /etc/caddy/Caddyfile
          name: caddyfile
          readOnly: true
        - mountPath: /data
          name: caddy-data
        - mountPath: /config
          name: caddy-config
      ports:
        - containerPort: 80
          hostPort: 80
        - containerPort: 443
          hostPort: 443

    - name: prometheus
      image: docker.io/prom/prometheus:v3.3.0
      args:
        - "--config.file=/etc/prometheus/prometheus.yml"
        - "--storage.tsdb.path=/prometheus"
        - "--web.enable-lifecycle"
      volumeMounts:
        - mountPath: /etc/prometheus/prometheus.yml
          name: prometheus-config
          readOnly: true
        - mountPath: /etc/prometheus/rules
          name: prometheus-rules
          readOnly: true
        - mountPath: /prometheus
          name: prometheus-data
      ports:
        - containerPort: 9090
          hostIP: 127.0.0.1
          hostPort: 9090

  volumes:
    - name: postgres-data
      persistentVolumeClaim:
        claimName: postgres-data
    - name: valkey-data
      persistentVolumeClaim:
        claimName: valkey-data
    - name: caddyfile
      hostPath:
        path: /opt/skating-app/Caddyfile
        type: File
    - name: caddy-data
      persistentVolumeClaim:
        claimName: caddy-data
    - name: caddy-config
      persistentVolumeClaim:
        claimName: caddy-config
    - name: prometheus-config
      hostPath:
        path: /opt/skating-app/prometheus.yml
        type: File
    - name: prometheus-rules
      hostPath:
        path: /opt/skating-app/prometheus/rules
        type: Directory
    - name: prometheus-data
      persistentVolumeClaim:
        claimName: prometheus-data

---
apiVersion: v1
kind: Secret
metadata:
  name: skating-secrets
type: Opaque
stringData:
  postgres-password: "${POSTGRES_PASSWORD}"
  database-url: "${DATABASE_URL}"
  r2-endpoint-url: "${R2_ENDPOINT_URL}"
  r2-access-key-id: "${R2_ACCESS_KEY_ID}"
  r2-secret-access-key: "${R2_SECRET_ACCESS_KEY}"
  r2-bucket: "${R2_BUCKET}"
  jwt-secret: "${JWT_SECRET}"
  vastai-api-key: "${VASTAI_API_KEY}"
  resend-api-key: "${RESEND_API_KEY}"
  cloudflare-api-token: "${CLOUDFLARE_API_TOKEN}"
```

**Note:** Podman does NOT support `depends_on` in kube play. Containers start in parallel. Backend healthcheck handles the wait implicitly.

- [ ] **Step 2: Test deploy.yaml syntax locally**

Run: `podman kube play --dry-run infra/deploy.yaml`
Expected: No syntax errors (dry-run accepted by podman 5.x+)

- [ ] **Step 3: Commit**

```bash
git add infra/deploy.yaml
git commit -m "infra(deploy): add Kubernetes YAML for podman kube play"
```

---

## Task 2: Create Persistent Volumes for Podman

**Files:**
- Create: `infra/volumes.yaml`
- Modify: `infra/deploy.yaml` (add PVC references)

Podman kube play auto-creates named volumes for `persistentVolumeClaim` refs, but only if the volume doesn't exist. We need to ensure volume names match.

- [ ] **Step 1: Verify volume names in deploy.yaml**

The `persistentVolumeClaim.claimName` values in deploy.yaml map to Podman named volumes:
- `postgres-data` → `podman volume create postgres-data` (auto-created by kube play)
- `valkey-data` → `podman volume create valkey-data`
- `caddy-data` → `podman volume create caddy-data`
- `caddy-config` → `podman volume create caddy-config`
- `prometheus-data` → `podman volume create prometheus-data`

Podman kube play creates these automatically on first run. No extra files needed.

**However:** existing volumes from compose.prod.yaml use prefixes. Check if data migration needed:

- [ ] **Step 2: Check existing volume names**

Run on VPS:
```bash
podman volume ls
```

If volumes are prefixed (e.g., `skating-app_postgres-data`), we need to either:
- Rename volumes, OR
- Update deploy.yaml claimName to match existing names

- [ ] **Step 3: Update deploy.yaml claimName if needed**

If existing volume is `skating-app_postgres-data`, change:
```yaml
claimName: skating-app_postgres-data
```

- [ ] **Step 4: Commit**

```bash
git add infra/deploy.yaml
git commit -m "infra(deploy): align PVC names with existing podman volumes"
```

---

## Task 3: Update Deploy Workflow

**Files:**
- Modify: `.github/workflows/deploy.yml`

Replace podman-compose steps with `podman kube play --replace`.

- [ ] **Step 1: Replace "Sync compose & Caddyfile" with "Sync deploy files"**

```yaml
      - name: Sync deploy files to VPS
        id: sync
        uses: appleboy/scp-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          source: "infra/deploy.yaml,infra/Caddyfile,infra/prometheus.yml,infra/prometheus/rules"
          target: "/opt/skating-app/"
          strip_components: 1
```

- [ ] **Step 2: Replace "Pull images" with "Login to GHCR"**

Remove the explicit pull step — `podman kube play --replace` pulls automatically.

```yaml
      - name: Login to GHCR
        id: login
        uses: appleboy/ssh-action@v1.2.5
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          command_timeout: 1m
          script: |
            echo "${{ secrets.GHCR_PAT }}" | podman login ghcr.io -u ${{ github.actor }} --password-stdin
```

- [ ] **Step 3: Replace "Rolling restart" with "Kube play replace"**

```yaml
      - name: Deploy pod
        id: deploy
        uses: appleboy/ssh-action@v1.2.5
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          command_timeout: 3m
          script: |
            cd /opt/skating-app
            # Substitute env vars into deploy.yaml and apply
            export POSTGRES_PASSWORD="${{ secrets.POSTGRES_PASSWORD }}"
            export DATABASE_URL="postgresql+asyncpg://skating:${{ secrets.POSTGRES_PASSWORD }}@127.0.0.1:5432/skating"
            export R2_ENDPOINT_URL="${{ secrets.CF_R2_ENDPOINT_URL }}"
            export R2_ACCESS_KEY_ID="${{ secrets.CF_R2_ACCESS_KEY_ID }}"
            export R2_SECRET_ACCESS_KEY="${{ secrets.CF_R2_SECRET_ACCESS_KEY }}"
            export R2_BUCKET="${{ secrets.CF_R2_BUCKET }}"
            export JWT_SECRET="${{ secrets.JWT_SECRET }}"
            export VASTAI_API_KEY="${{ secrets.VASTAI_API_KEY }}"
            export RESEND_API_KEY="${{ secrets.RESEND_API_KEY }}"
            export CLOUDFLARE_API_TOKEN="${{ secrets.CLOUDFLARE_API_TOKEN }}"
            export TAG="${{ github.sha }}"
            export GHCR_OWNER="${{ steps.ghcr.outputs.owner }}"

            # envsubst replaces ${VAR} placeholders in deploy.yaml
            envsubst < deploy.yaml > deploy-rendered.yaml

            # Replace images with sha-tagged versions
            sed -i "s|ghcr.io/artiffusion-inc/skating-backend:latest|ghcr.io/${GHCR_OWNER}/skating-backend:${TAG}|g" deploy-rendered.yaml
            sed -i "s|ghcr.io/artiffusion-inc/skating-frontend:latest|ghcr.io/${GHCR_OWNER}/skating-frontend:${TAG}|g" deploy-rendered.yaml

            podman kube play --replace deploy-rendered.yaml
```

- [ ] **Step 4: Replace "Run database migrations"**

```yaml
      - name: Run database migrations
        id: migrate
        uses: appleboy/ssh-action@v1.2.5
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          command_timeout: 1m
          script: |
            # Wait for backend to be running
            sleep 5
            podman exec skating-app-backend alembic upgrade head
```

**Note:** Container name in kube play pod is `<pod-name>-<container-name>`. With pod name `skating-app`, backend container is `skating-app-backend`.

- [ ] **Step 5: Replace "Health check"**

```yaml
      - name: Health check
        id: health
        uses: appleboy/ssh-action@v1.2.5
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          command_timeout: 3m
          script: |
            timeout 120 bash -c '
              while true; do
                # Check pod is running and containers are healthy
                status=$(podman pod ps --format "{{.Name}}\t{{.Status}}" | grep skating-app | awk "{print \$2}")
                if [ "$status" = "Running" ]; then
                  # Additional check: backend health endpoint
                  if podman exec skating-app-backend \
                    bash -c "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)\"" 2>/dev/null; then
                    echo "All services healthy"
                    exit 0
                  fi
                fi
                sleep 10
              done
            '
```

- [ ] **Step 6: Commit workflow changes**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci(deploy): migrate to podman kube play --replace"
```

---

## Task 4: Test Deploy on VPS

**Files:**
- Modify: `infra/deploy.yaml` (if fixes needed)
- Modify: `.github/workflows/deploy.yml` (if fixes needed)

- [ ] **Step 1: Manual test on VPS**

SSH to VPS:
```bash
cd /opt/skating-app
# Create env file for substitution
cat > .env.kube <<'EOF'
POSTGRES_PASSWORD=your-password
DATABASE_URL=postgresql+asyncpg://skating:your-password@127.0.0.1:5432/skating
R2_ENDPOINT_URL=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=...
JWT_SECRET=...
VASTAI_API_KEY=...
RESEND_API_KEY=...
CLOUDFLARE_API_TOKEN=...
TAG=efbea188620f111df8e517574acda662cb76187c
GHCR_OWNER=artiffusion-inc
EOF

set -a && source .env.kube && set +a
envsubst < deploy.yaml > deploy-rendered.yaml
podman kube play --replace deploy-rendered.yaml
```

- [ ] **Step 2: Verify pod status**

```bash
podman pod ps
podman ps --format "{{.Names}} {{.Status}}"
```

Expected: `skating-app` pod Running, all containers healthy.

- [ ] **Step 3: Verify API**

```bash
curl -s http://localhost:8000/api/v1/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Test replace idempotency**

Run `podman kube play --replace deploy-rendered.yaml` again.
Expected: New pod created, old pod removed, no 502 downtime.

- [ ] **Step 5: Fix any issues and commit**

If issues found, fix deploy.yaml or workflow, commit:
```bash
git add infra/deploy.yaml .github/workflows/deploy.yml
git commit -m "fix(deploy): adjust kube play config after testing"
```

---

## Task 5: Cleanup Old Compose References

**Files:**
- Modify: `.github/workflows/deploy.yml` (remove unused steps)
- Modify: `infra/compose.prod.yaml` (add deprecation comment)

- [ ] **Step 1: Mark compose.prod.yaml as deprecated**

Add header comment:
```yaml
# DEPRECATED: Use infra/deploy.yaml + podman kube play for production.
# Kept for local development reference only.
```

- [ ] **Step 2: Remove podman-compose references from workflow**

Ensure no `podman compose` commands remain in deploy.yml.

- [ ] **Step 3: Commit**

```bash
git add infra/compose.prod.yaml .github/workflows/deploy.yml
git commit -m "chore(deploy): deprecate compose.prod.yaml, use kube play"
```

---

## Task 6: Verify GitHub Actions Workflow

**Files:**
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Trigger test deploy**

Push to master (or use `workflow_dispatch` if configured).

- [ ] **Step 2: Monitor workflow run**

Check GitHub Actions UI for:
- CI passes
- Build steps complete
- Deploy step: Sync → Login → Kube play → Migrations → Health check → Cleanup
- No failures

- [ ] **Step 3: Verify on VPS**

```bash
ssh root@150.241.124.159 "podman pod ps && podman ps --format '{{.Names}} {{.Image}}' | grep backend"
```

Expected: `skating-app` Running, backend image matches latest commit SHA.

- [ ] **Step 4: Commit any final fixes**

---

## Self-Review

**1. Spec coverage:**
- ✅ Eliminate podman-compose v1.3.0 workaround
- ✅ Use `podman kube play --replace` for atomic recreate
- ✅ Keep existing volumes and data
- ✅ Health check after deploy
- ✅ Database migrations
- ✅ GHCR image tags with commit SHA

**2. Placeholder scan:** No TBD/TODO placeholders found.

**3. Type consistency:** Not applicable (YAML + bash).

**4. Risk: Data loss during volume migration.**
- Mitigation: Check existing volume names in Task 2 Step 2 before running kube play.
- If volume names differ, update claimName instead of recreating.

**5. Risk: Downtime during migration.**
- Mitigation: `podman kube play --replace` creates new pod before removing old one in Podman 5.x+.
- Verify on VPS in Task 4 Step 4.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-06-kube-play-deploy.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - Fresh subagent per task + review loop. Commit after every step. Verify on VPS before next task.

**2. Inline Execution** - Execute tasks in this session. Risk: single context window may fill with YAML/bash.

**Which approach?**
