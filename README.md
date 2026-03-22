# uv_agent
A Blender extension for UV unwrap seam suggestions with optional automatic AI calls (no Copilot required).

## Automated workflow (API)
1. In Blender panel `My UV Tool`, set:
	- `Auto call AI` = enabled
	- `AI Endpoint` (OpenAI-compatible chat completions URL)
	- `Model`
	- `API Key`
	- `OpenAI Project (optional)` (recommended for `sk-proj` keys)
	- `OpenAI Org (optional)`
2. Click `Unwrap My Mesh`.
3. The addon exports data, calls the model, and writes `import_seams.json` automatically.
4. Click `Apply AI Seams`.

### Endpoint examples
- OpenAI: `https://api.openai.com/v1/chat/completions`
- OpenRouter: `https://openrouter.ai/api/v1/chat/completions`
- Local (OpenAI-compatible server): e.g. `http://localhost:1234/v1/chat/completions`

### If you get HTTP 401
- Re-paste API key (no quotes, no spaces, no line breaks).
- For OpenAI `sk-proj` keys, set `OpenAI Project (optional)` to your `proj_...` id.
- Check that your selected model is enabled for your account/project.
- Read full provider response in `uv_agent_shots/ai_error_response.txt`.

## Manual fallback workflow
If API settings are missing or call fails:
1. Click `Unwrap My Mesh`.
2. Use `export_data.json`, image files, and `prompt.txt` from `uv_agent_shots`.
3. Put the model output array into `import_seams.json`.
4. Click `Apply AI Seams`.

## Output contract
The model must return exactly one JSON array of integer edge indices.
Example: `[0, 15, 23, 42]`
