FROM python:3.11-slim-buster
# WORKDIR /app
COPY . /app
WORKDIR /app

EXPOSE 5000

RUN pip install --upgrade pip

RUN pip install -r requirements.txt 

RUN python -m nltk.downloader punkt wordnet stopwords omw-1.4


CMD ["python", "app.py"]