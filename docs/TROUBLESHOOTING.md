# Troubleshooting Guide

Common errors and solutions for the Illustrative AI pipeline.

## Authentication Errors

### "GEMINI_API_KEY not set" or "Neither GEMINI_API_KEY nor Vertex AI configured"

**Cause:** Missing API credentials in `.env` file.

**Solution:**
```bash
# Option A: AI Studio
echo "GEMINI_API_KEY=your_key_here" > .env

# Option B: Vertex AI
cat > .env << EOF
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
EOF
```

### "Could not automatically determine credentials" (Vertex AI)

**Cause:** Application Default Credentials (ADC) not configured.

**Solution:**
```bash
gcloud auth application-default login
```

### "Credentials refresh failed" (Vertex AI)

**Cause:** ADC token expired or invalid.

**Solution:**
```bash
# Re-authenticate
gcloud auth application-default login

# Or if using service account
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### "Permission denied" or "403 Forbidden" (Vertex AI)

**Cause:** GCP account lacks required permissions.

**Solution:**
1. Ensure the Vertex AI API is enabled in your project
2. Grant your account the `Vertex AI User` role:
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT \
     --member="user:your-email@example.com" \
     --role="roles/aiplatform.user"
   ```

---

## Rate Limiting Errors

### "429 Resource Exhausted" or "Quota exceeded"

**Cause:** Too many requests per minute (RPM) or tokens per minute (TPM).

**Symptoms:**
- Error appears after several successful requests
- More common with image generation than text

**Solutions:**

1. **Wait and retry** - The `retry_with_backoff()` decorator handles this automatically with exponential backoff.

2. **Reduce concurrency** - Edit `config.py`:
   ```python
   # Lower these values
   scripting_rpm: int = 3  # was 5
   image_rpm: int = 3      # was 5
   ```

3. **Switch to Vertex AI** - Higher quotas than AI Studio free tier:
   ```bash
   # In .env
   GOOGLE_GENAI_USE_VERTEXAI=true
   ```

4. **Request quota increase** (Vertex AI only):
   - Go to GCP Console > IAM & Admin > Quotas
   - Filter by "Vertex AI" and "Generative AI"
   - Request increase for relevant quotas

### Pipeline stalls but no error

**Cause:** Rate limiter semaphore blocking requests.

**Solution:** Check if you're hitting TPM limits. The pipeline logs token usage - look for warnings about approaching limits.

---

## Model Errors

### "Model not found" or "Model is not supported"

**Cause:** Model name incorrect or not available in your region.

**Solution:**

1. **Check model availability:**
   ```python
   from google import genai
   from config import config

   client = genai.Client(
       vertexai=config.use_vertex_ai,
       project=config.gcp_project,
       location=config.gcp_location
   )
   for m in client.models.list():
       print(m.name)
   ```

2. **Use correct model names** (as of 2025):
   - Text: `gemini-2.5-flash`
   - Image: `gemini-3-pro-image-preview`, `gemini-2.5-flash-preview-image`, `gemini-2.5-flash-image`

3. **Try a different region** - Some preview models have limited availability:
   ```bash
   # In .env
   GOOGLE_CLOUD_LOCATION=us-east4  # or europe-west4
   ```

### "Image generation failed" with fallback

**Cause:** Primary image model unavailable or overloaded.

**Info:** This is expected behavior. The pipeline has 3-tier fallback:
1. `gemini-3-pro-image-preview` (highest quality)
2. `gemini-2.5-flash-preview-image` (fallback)
3. `gemini-2.5-flash-image` (last resort, 1024px max)

Check logs to see which model succeeded.

---

## Server Errors

### "500 Internal Server Error" or "503 Service Unavailable"

**Cause:** Temporary Gemini API issues.

**Solution:** The `retry_with_backoff()` decorator retries automatically up to 5 times. If persistent:

1. Check [Google Cloud Status](https://status.cloud.google.com/) for outages
2. Wait a few minutes and retry
3. Try a different model or region

### "504 Gateway Timeout"

**Cause:** Request took too long (usually image generation).

**Solution:**
1. The pipeline has built-in 180s timeout with retry
2. If persistent, try smaller/simpler prompts
3. Use a faster model tier (`gemini-2.5-flash-image`)

---

## Timeout Errors

### "Operation timed out after Xs"

**Cause:** API call exceeded timeout limit.

**Common scenarios:**
- Large context caching operations
- Complex image generation
- Network issues

**Solutions:**

1. **Check network connectivity**

2. **Increase timeout** (if you have slow network):
   ```python
   # In utils.py retry_with_backoff decorator
   timeout_seconds=300  # increase from 180
   ```

3. **Break into smaller chunks** - For very long books, process in segments

---

## Resume & State Errors

### "Manifest file corrupted" or JSON parse errors

**Cause:** Pipeline was interrupted during manifest write.

**Solution:**
```bash
# Remove corrupted manifest to start fresh
rm assets/output/production_manifest.json

# Or restore from backup if available
cp assets/output/production_manifest.json.bak assets/output/production_manifest.json
```

### Pipeline re-processes already completed pages

**Cause:** Manifest not being read correctly.

**Solution:**
1. Check that `assets/output/production_manifest.json` exists
2. Verify JSON is valid: `python -m json.tool assets/output/production_manifest.json`
3. Check file permissions

---

## Image Output Issues

### Generated images are blank or corrupted

**Cause:** Image generation returned invalid data.

**Solution:**
1. Check the `assets/output/pages/` directory for the problematic images
2. Delete corrupted images and re-run (manifest will skip completed ones)
3. Try a different image model

### Text bubbles are cut off or misaligned

**Cause:** Font file missing or incorrect path.

**Solution:**
1. Ensure font exists: `ls fonts/PatrickHand-Regular.ttf`
2. Download a TTF font if missing
3. Update `config.py` if using a different font:
   ```python
   font_path: str = "fonts/YourFont.ttf"
   ```

---

## Configuration Validation

Run this to validate your setup:
```bash
python config.py
```

Expected output:
```
✓ Using Vertex AI (project: your-project, location: us-central1)
  Rate limits: 30 RPM (text), 10 RPM (image), 4,000,000 TPM
✅ Configuration validated successfully
```

Or for AI Studio:
```
✓ Using Gemini API key (AI Studio mode)
  Rate limits: 5 RPM (text), 5 RPM (image), 1,000,000 TPM
✅ Configuration validated successfully
```

---

## Getting Help

If issues persist:
1. Check the error message in full - it often contains the solution
2. Review `CLAUDE.md` for architecture details
3. Review `VERTEX_AI_QUESTIONS.md` for quota-related issues
4. Search for the error message in Google Cloud documentation
