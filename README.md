# Profile Matcher

[![Docker](https://img.shields.io/badge/Built%20With-Docker-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

This project uses NLP techniques to predict profile compatibility based on user bios, inspired by HackerEarth’s ["Love in the Time of Screens"](https://www.kaggle.com/datasets/nikhildr22/hackerearth-love-in-the-time-of-screens) challenge. The full pipeline is packaged in Docker for easy deployment and reproducibility.

---

## 📊 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/nikhildr22/hackerearth-love-in-the-time-of-screens)
- **Data**: Contains user bios and a match indicator for user pairs.
- 📁 Place all dataset files in the `data/` directory before running the project.

---

## 🗂️ Project Structure

📁 Profile-Matcher/ ├── 📁 src/ │ ├── data_preprocess.py # Step 1: Clean and format the dataset │ ├── vectorising_bio.py # Step 2: Convert bios to numeric vectors │ └── calculate_match.py # Step 3: Compute similarity and predict matches ├── 📁 data/ # Directory for dataset and output files ├── Dockerfile # Docker configuration ├── requirements.txt # Python dependencies └── README.md # Project documentation
