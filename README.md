## WizlearnrAI Streamlit App

To run this project, use poetry:
```bash
pip install poetry
```

Then install the dependencies:
```bash
poetry install
```

Activate poetry environment:
```bash
poetry shell
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

## Deploy steps

```bash
docker build --platform linux/amd64 -t wlapp:<tag> .
docker tag wlapp:latest asia-south1-docker.pkg.dev/genai-exchange-hackathon/app/wlapp:<tag>
docker push asia-south1-docker.pkg.dev/genai-exchange-hackathon/app/wlapp:<tag>
```

Then create new revision on cloud run and select the new docker image to deploy.

Roadmap:
- [x] Add safety
- [x] Add session history
- [x] Improve solver
- [x] Add response actions: rate, copy, regenerate
- [ ] Add more visualisations
- [ ] Add audio mode
- [ ] Add user auth
- [ ] Add user personalisation
