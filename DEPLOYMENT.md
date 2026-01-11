# Deployment Guide

This document explains how to run TicketsAnalyzerAI in different environments with both the Streamlit UI and FastAPI webhook service.

## Quick Start

### Local Development (Recommended)

**Python Script (Cross-platform):**
```bash
python start_local.py
```

**Bash Script (Linux/Mac):**
```bash
chmod +x start_local.sh
./start_local.sh
```

This will start:
- Streamlit UI on http://localhost:8502
- FastAPI webhook on http://127.0.0.1:8010
- API docs on http://127.0.0.1:8010/docs

### Docker

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. **Start with docker-compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access services:**
   - Streamlit UI: http://localhost:8503
   - FastAPI webhook: http://localhost:8010
   - API docs: http://localhost:8010/docs

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

### Streamlit Cloud

⚠️ **Important Limitations:**

Streamlit Cloud has restrictions that affect the FastAPI webhook service:

1. **No Custom Ports**: Cannot expose ports other than the Streamlit port
2. **No Background Services**: Cannot run uvicorn alongside Streamlit
3. **Webhook Service Disabled**: The FastAPI webhook will NOT be available

**Workarounds:**

- **Option A**: Deploy FastAPI separately (Heroku, Render, Cloud Run, etc.)
- **Option B**: Use only the Streamlit UI features (no webhook auto-commenting)
- **Option C**: Host on a VPS/VM with both services (instead of Streamlit Cloud)

**Deploy to Streamlit Cloud:**

1. Push code to GitHub
2. Connect repository on https://share.streamlit.io
3. Set secrets in Streamlit Cloud dashboard:
   ```toml
   [secrets]
   OPENAI_API_KEY = "sk-..."
   YT_BASE_URL = "https://your-youtrack.myjetbrains.com"
   YT_TOKEN = "perm:..."
   ```
4. Deploy (webhook service will be automatically skipped)

## Architecture

### Local/Docker Architecture

```
┌─────────────────────────────────────────┐
│  Docker Container / Local Machine       │
│                                          │
│  ┌────────────────┐  ┌────────────────┐│
│  │  Streamlit UI  │  │ FastAPI Webhook││
│  │   Port 8501    │  │   Port 8010    ││
│  └────────────────┘  └────────────────┘│
│          │                    │         │
│          └────────┬───────────┘         │
│                   │                     │
│            ┌──────▼──────┐             │
│            │  ChromaDB   │             │
│            │  (Vector DB)│             │
│            └─────────────┘             │
└─────────────────────────────────────────┘
```

### Streamlit Cloud Architecture

```
┌─────────────────────────────────────────┐
│  Streamlit Cloud                        │
│                                          │
│  ┌────────────────┐                     │
│  │  Streamlit UI  │                     │
│  │   (Public URL) │                     │
│  └────────────────┘                     │
│          │                              │
│   ┌──────▼──────┐                      │
│   │  ChromaDB   │                      │
│   │  (Ephemeral)│                      │
│   └─────────────┘                      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  External Service (Optional)            │
│                                          │
│  ┌────────────────┐                     │
│  │ FastAPI Webhook│                     │
│  │  (Heroku/etc)  │                     │
│  └────────────────┘                     │
└─────────────────────────────────────────┘
```

## Environment Variables

See `.env.example` for all available variables.

### Required (Minimal Setup)

```bash
OPENAI_API_KEY=sk-...
```

### Webhook Service (Optional)

```bash
YT_BASE_URL=https://your-youtrack.myjetbrains.com
YT_TOKEN=perm:your-token-here
WEBHOOK_SECRET=your-secret-key  # Optional but recommended
```

### Advanced Configuration

```bash
# Retrieval settings
TOP_K=5
MAX_DISTANCE=0.9
PER_PARENT_DISPLAY=1
PER_PARENT_PROMPT=3
STITCH_MAX_CHARS=1500

# Memory collection
ENABLE_MEMORY=0
MEM_COLLECTION=memories
```

## Manual Start (Advanced)

If you prefer to start services manually:

**Streamlit:**
```bash
streamlit run app.py --server.port 8502
```

**FastAPI (separate terminal):**
```bash
uvicorn service_webhook:app --host 127.0.0.1 --port 8010 --log-level info
```

## Troubleshooting

### Port Already in Use

```bash
# Linux/Mac: Find process using port
lsof -ti:8502 | xargs kill -9
lsof -ti:8010 | xargs kill -9

# Windows: Find and kill process
netstat -ano | findstr :8502
taskkill /PID <pid> /F
```

### Docker: Permission Denied

```bash
chmod +x start_docker.sh
```

### Docker: ChromaDB Persistence

Data is stored in `./data_docker/chroma/` when using docker-compose.

### Webhook Not Receiving Requests

1. Check firewall settings
2. Verify YouTrack webhook URL: `http://your-server:8010/yt/issue-created`
3. Check webhook secret matches `X-Webhook-Secret` header
4. View logs: `docker-compose logs -f` or check terminal output

## Production Deployment

### Recommended Stack

- **Streamlit UI**: Streamlit Cloud (free tier)
- **FastAPI Webhook**: Cloud Run, Heroku, Render, or VPS
- **ChromaDB**: Persistent volume on webhook service

### Security Checklist

- [ ] Set `WEBHOOK_SECRET` environment variable
- [ ] Use HTTPS for webhook endpoint
- [ ] Restrict webhook IP in YouTrack (if possible)
- [ ] Use read-only YouTrack token (minimal permissions)
- [ ] Enable firewall rules for ports 8010, 8501
- [ ] Regular backups of ChromaDB data
- [ ] Monitor logs for suspicious activity

## FAQ

**Q: Can I use a different port for FastAPI?**
A: Yes, edit the port in `start_local.py`, `start_docker.sh`, and `docker-compose.yml`

**Q: Do I need both services?**
A: No. FastAPI webhook is optional. You can use just Streamlit UI.

**Q: Can I deploy FastAPI separately?**
A: Yes. Deploy `service_webhook.py` as a standalone FastAPI app with its own requirements.

**Q: Does webhook work on Streamlit Cloud?**
A: No. Deploy FastAPI separately or use a VPS/VM for both services.

**Q: How do I update after git pull?**
```bash
# Docker
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Local
# Just restart the script
python start_local.py
```
