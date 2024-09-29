FROM python:3.12

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /wlapp .
COPY wizlearnr_logo.png .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit-app.py"]
