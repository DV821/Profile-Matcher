# Profile Matcher

A Dockerized NLP-based project to predict compatibility between user profiles from their profile features and bios. Inspired by the [HackerEarth "Love in the Time of Screens"](https://www.kaggle.com/datasets/nikhildr22/hackerearth-love-in-the-time-of-screens) challenge.

---

## 📊 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/nikhildr22/hackerearth-love-in-the-time-of-screens)
- **Files**: Contains user bios and match indicators.
- **Usage**: Place all `.csv` files inside the `data/` folder.

---

## 📁 Project Structure

Profile-Matcher/ <br>
├── 📁 data/                    # Raw and processed datasets <br>
│   └── data.csv <br>
│ <br>
├── 📁 src/                     # Source code for the pipeline <br>
│   ├── data_preprocess.py     # Step 1: Clean and preprocess bios <br>
│   ├── vectorising_bio.py     # Step 2: Convert bios to vectors (e.g., TF-IDF) <br>
│   └── calculate_match.py     # Step 3: Compute similarity and predict match <br>
│   └── utils.py              # All commonly used functions throughout the projects <br>
│ <br>
├── 📁 models/                  # All preprocessing models saved <br>
│   └── all_langs.pkl          # Dictionary for all languages <br>
│   └── count_vectorizer.pkl   # Vectorizer to vectorize bio <br>
│   └── le_job.pkl             # Label Encoder model to encode jobs <br>
│   └── ....
│ <br>
├── 📁 notebooks/               # Notebook files for rough execution of the project <br>
│   └── Feature handling.ipynb # For handling features <br>
│   └── Vectorising bio.ipynb  # To vectorise text in bio <br>
│   └── Calculate match.ipynb  # To match profiles <br>
│ <br>
├── 📁 static/                  # Css files <br>
│   └── final.css              # For final.html file <br>
│   └── index.css              # For index.html file <br>
│   └── sty;es.css             # FOr user1.html and user2.html files <br>
│ <br>
├── 📁 templates/              # HTML files for Flask Web app <br>
│   └── final.html            # Profile Match Result <br>
│   └── index.html            # Home page <br>
│   └── user1.html            # User 1 Profile Form <br>
│   └── user2.html            # User 2 Profile Form <br>
│ <br>
├── Dockerfile                 # Docker configuration file <br>
├── requirements.txt           # Python dependencies <br>
├── app.py                     # Flask App <br>
└── README.md                  # Project overview and usage instructions <br>



---

## 🐳 Docker Usage

### 🔨 Build the Docker Image

```bash
docker build -t profile-matcher .
```

### 🚀 Run the Container

docker run --rm -it profile-matcher

    This command runs the complete pipeline inside the container.

### ⚙️ Running Scripts Manually

To enter the container shell:

docker run -it --entrypoint /bin/bash profile-matcher

Run the pipeline step-by-step:

python src/data_preprocess.py <br>
python src/vectorising_bio.py <br>
python src/calculate_match.py <br>

## 💡 Features

    Label encodes user profile featueres
    
    Preprocesses and cleans user bios

    Vectorizes bios using NLP methods (Count Vectoriser)

    Uses cosine similarity to calculate match score

    Easy to extend with deep learning models

## 🚀 Future Improvements

Integrate transformer models (BERT, SBERT)

Serve predictions via FastAPI

Add frontend dashboard for interactive use

    Visualizations & evaluation metrics

🤝 Contributing

Contributions are welcome!
Please fork this repo, create a new branch, and submit a pull request.

Open an issue if you have questions or feature requests!


Let me know if you'd like the `Dockerfile` and `requirements.txt` to go along with this!

