# Prakriti Classifier - React Frontend

This folder contains the React single-page application built with Vite and vanilla CSS.

---

## 🚀 Hosting on Netlify

### Step 1: Deploy to Netlify
1. Sign in to your [Netlify Dashboard](https://app.netlify.com/).
2. Click **Add new site** > **Import an existing project**.
3. Connect your GitHub repository.
4. Configure the build settings:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist` (or `dist` if base is set to `frontend`)
5. Click **Deploy Site**.

### Step 2: Connect Frontend to Backend (Render)
Under **Site configuration** > **Environment variables**:
1. Add an environment variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend-service.onrender.com` (replace with your Render backend URL)
2. Trigger a redeploy: **Deploys** > **Trigger deploy** > **Clear cache and deploy site**.

Your frontend will now connect directly to your backend on Render with zero CORS issues!
