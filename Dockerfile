FROM python:3.11-slim-buster

COPY . /app
WORKDIR /app

EXPOSE 5000

RUN pip install --upgrade pip

RUN pip install -r requirements.txt 

RUN python -m nltk.downloader punkt wordnet stopwords omw-1.4

CMD ["python", "src/data_preprocess.py"]
CMD ["python", "src/vectorising_bio.py"]
CMD ["python", "src/calculate_match.py"]
CMD ["python", "app.py"]
