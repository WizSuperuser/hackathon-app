FROM python:3.12

RUN apt-get update && apt-get -y install
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /wlapp .
COPY wizlearnr_logo.png .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit-app.py"]

