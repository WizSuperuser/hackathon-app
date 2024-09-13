## WizlearnrAI Streamlit App

To run this project, use poetry:
```bash
pip install poetry
```

Then install the dependencies:
```bash
poetry install
```

Add the .env file.

Then run the streamlit app:
```bash
streamlit run wlapp/streamlit-app.py
```

To try the newest version, run the graph.ipynb notebook. 

To run the vertex ai llm, make sure google cloud is authenticated using superuser account:
```bash
gcloud auth application-default login
```

Roadmap:
- [ ] Add safety
- [ ] Improve solver