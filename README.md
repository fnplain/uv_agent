# uv_agent
A Blender extension for automatic UV seam placement

## Automated workflow (API)
1. In Blender tab `Scripting`, run:
	- `uv_tool.py`
2. In Blender side view find `My UV Tool`
3. Click `Unwrap My Mesh`.
4. The addon exports data and writes `import_seams.json` automatically.
5. Run `run_xatlas.py` `python run_xatlas.py "C:\your\path\here\Local\Temp\blender_yourNumbers\uv_agent_shots\export_data.json"`
5. Click `Apply AI Seams`.

## Small description 

Currently it uses a library 'XAtlas' combined with high-stress region indentification and sorting. 