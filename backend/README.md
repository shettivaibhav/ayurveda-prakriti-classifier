# Prakriti Classifier - Backend API Service

This folder contains the Python HTTP API server and machine learning pipeline for the **Ayurvedic Prakriti Classifier**.

---

## 🚀 Hosting on Render

### Step 1: Push Repository to GitHub
Ensure this project repository is pushed to your GitHub account.

### Step 2: Create a Web Service on Render
1. Sign in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Fill in the service configuration:
   - **Name**: `prakriti-backend-api` (or your choice)
   - **Language**: `Python`
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Instance Type**: `Free`

### Step 3: Configure Environment Variables
In the Render service settings under **Environment**, add:
- `PYTHON_VERSION`: `3.11.9`
- `CORS_ORIGIN`: `*` (or your Netlify domain, e.g. `https://your-site.netlify.app`)

*(Optional Cloud Features)*
- `SUPABASE_URL`: (your Supabase project URL)
- `SUPABASE_KEY`: (your Supabase anon or service key)
- `GEMINI_API_KEY`: (optional, for WhatsApp multimodal extraction)

### Step 4: Deploy and Note Your URL
Once deployed, Render will provide a URL like:
`https://prakriti-backend-api.onrender.com`

Verify by opening:
`https://prakriti-backend-api.onrender.com/api/status`
You should see: `{"ok": true, "model_ready": true, ...}`

Copy this URL — you will need it for the frontend configuration on Netlify!
