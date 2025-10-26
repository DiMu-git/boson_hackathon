.PHONY: run_app run_api run_api_no_reload

run_app:
	PATH="$$HOME/.local/bin:$$PATH" uv run streamlit run streamlit_app.py

run_api:
	PATH="$$HOME/.local/bin:$$PATH" uv run uvicorn backend.app:app --reload

run_api_no_reload:
	PATH="$$HOME/.local/bin:$$PATH" uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000