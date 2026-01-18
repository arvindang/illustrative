# Vertex AI Migration - Questions & Decisions Needed

This file tracks questions and decisions needed from the project owner regarding Vertex AI configuration.

## Rate Limits

### Current Settings (config.py)
```python
scripting_rpm: int = 5
character_rpm: int = 5
image_rpm: int = 5
tpm_limit: int = 1_000_000
```

### Questions

1. **What are your actual Vertex AI quotas?**
   - Check in GCP Console: IAM & Admin > Quotas
   - Filter by "Vertex AI" and "Generative AI"
   - Key quotas to check:
     - `Generate content requests per minute per region`
     - `Image generation requests per minute per region`
     - `Context caching requests per minute`

2. **Do you want different rate limits for Vertex AI vs AI Studio?**
   - Vertex AI (paid) typically allows higher throughput
   - Current conservative limits (5 RPM) may be unnecessarily slow on Vertex AI
   - Suggested Vertex AI defaults (adjust based on your quotas):
     - `scripting_rpm: 30` (text generation)
     - `character_rpm: 10` (image generation)
     - `image_rpm: 10` (image generation)
     - `tpm_limit: 4_000_000` (higher for Vertex AI)

---

## Model Selection

### Current Models (config.py)
```python
# Text Models
scripting_model_global_context: str = "gemini-2.5-flash"
scripting_model_chapter_map: str = "gemini-2.5-flash"
scripting_model_page_script: str = "gemini-2.5-flash"
layout_model: str = "gemini-2.5-flash"

# Image Models
image_model_primary: str = "gemini-3-pro-image-preview"  # FIXED from nano-banana-pro-preview
image_model_fallback: str = "gemini-3-pro-image-preview"
image_model_last_resort: str = "gemini-2.5-flash-image"

# Character Models
character_model_attributes: str = "gemini-3-flash-preview"
character_model_image: str = "gemini-3-pro-image-preview"
```

### Questions

1. **Are all these models available in your GCP region?**
   - Default region: `us-central1`
   - Some preview models may have limited regional availability
   - Run this to check:
     ```python
     from google import genai
     client = genai.Client(vertexai=True, project="YOUR_PROJECT", location="us-central1")
     for m in client.models.list():
         print(m.name)
     ```

2. **Do you want to use Imagen models instead of Gemini for image generation?**
   - Imagen 4 models are now GA on Vertex AI:
     - `imagen-4.0-generate-001` (standard)
     - `imagen-4.0-fast-generate-001` (faster, lower quality)
     - `imagen-4.0-ultra-generate-001` (highest quality)
   - Requires different API pattern (`:predict` endpoint)
   - Current implementation uses Gemini multimodal which is simpler

3. **What's your preferred image resolution?**
   - `gemini-3-pro-image-preview`: up to 4096px
   - `gemini-2.5-flash-image`: up to 1024px
   - Higher resolution = higher cost + longer generation time

---

## Authentication

### Questions

1. **How will you authenticate in production?**
   - **Option A**: Service Account JSON file
     - Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
     - Best for CI/CD and containerized deployments
   - **Option B**: Workload Identity (GKE)
     - No credentials file needed
     - Best for GKE deployments
   - **Option C**: Application Default Credentials
     - Run `gcloud auth application-default login`
     - Best for local development

2. **Do you need to support multiple GCP projects?**
   - Current implementation uses a single project from `GOOGLE_CLOUD_PROJECT`
   - If you need multi-tenancy, we'd need to modify client initialization

---

## Cost Considerations

### Estimated Costs (Vertex AI pricing as of 2025)

| Operation | Approximate Cost |
|-----------|------------------|
| Gemini 2.5 Flash (text) | $0.075 / 1M input tokens, $0.30 / 1M output tokens |
| Gemini 3 Pro Image | ~$0.04 per image |
| Context Caching | $1.00 / 1M tokens stored per hour |

### Questions

1. **Do you want to add cost tracking/estimation?**
   - Could add token usage logging
   - Could show estimated cost per run

2. **Do you want to set spending alerts in GCP?**
   - Recommended: Set budget alerts in GCP Console > Billing

---

## Region Selection

### Current Setting
```python
gcp_location: str = "us-central1"
```

### Questions

1. **Is `us-central1` the optimal region for you?**
   - Consider latency to your users/servers
   - Check model availability in other regions
   - Available regions with full Gemini support:
     - `us-central1` (Iowa)
     - `us-east4` (Virginia)
     - `europe-west4` (Netherlands)
     - `asia-northeast1` (Tokyo)

---

## Action Items After Review

- [ ] Confirm rate limits for your Vertex AI quotas
- [ ] Verify model availability in your region
- [ ] Choose authentication method for production
- [ ] Set up GCP billing alerts
- [ ] Test the pipeline end-to-end with Vertex AI enabled
