# StudySync

**ML-Based Study Material Organiser and Previous-Year Question Paper Analyser**  
Computer Science & Design, Year 3

StudySync is a Flask web application with two learning workflows: NoteMap
classifies extracted note text by subject, while QuestionBank Mapper extracts
questions from papers and matches them to user-reviewed syllabus topics. A
separate Model Lab demonstrates supervised learning and unsupervised analysis.
The project is an educational prototype; its generated NoteMap examples and
lexical matching baseline are not validated for general student use.

## Features

- Upload PDF or TXT notes and receive a NoteMap subject prediction with
  non-probability model scores and an option to review/correct the prediction.
- Upload a text-based syllabus PDF and multiple previous-year paper PDFs;
  review extracted topics, inspect question matches, and correct mappings.
- Preserve question-paper filename, page reference, and filename-derived year
  where available.
- Compare Multinomial Naive Bayes and Linear SVM with a reproducible,
  leakage-aware held-out evaluation.
- Demonstrate chi-squared feature selection, sparse TruncatedSVD, TF-IDF
  KMeans, and an optional Linear Regression experiment.
- View Pandas summaries, Matplotlib charts, year trends when file years allow,
  and CSV exports for mapped questions and topic counts.
- Validate upload type/size/content, keep temporary uploads private, and report
  that scanned-PDF OCR is not supported.

## Technology and requirements

| Area | Technology |
|---|---|
| Runtime | Python (verified in Python 3.13.0 in the project environment) |
| Web application | Flask, Jinja2, HTML, CSS, JavaScript |
| PDF extraction | PyMuPDF |
| ML | scikit-learn: TF-IDF, MultinomialNB, LinearSVC, SelectKBest/chi2, TruncatedSVD, KMeans, LinearRegression |
| Tabular analysis and charts | Pandas, NumPy, Matplotlib |
| Tests | pytest |

Dependency minimums are listed in [requirements.txt](./requirements.txt). The
regression workflow additionally needs the optional real dataset described in
[data/datasets/README.md](./data/datasets/README.md).

## Run the application and tests

Start the Flask development server:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. To stop the development server, press
**Ctrl+C** in its terminal.

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

Run the reproducible NoteMap evaluation separately:

```powershell
.\.venv\Scripts\python.exe -m ml.train_notemap
```

The measured example results, test counts, and interpretation notes are in the
[project documentation guide](./docs/DSBD_SUBMISSION_GUIDE.md).

## Main routes

| Route | Purpose |
|---|---|
| `/` and `/dashboard` | StudySync home/dashboard |
| `/notemap` | NoteMap upload page |
| `/notemap/analyze` | POST note upload, extract, classify |
| `/notemap/correct` | POST reviewed/corrected prediction |
| `/questionbank` | QuestionBank upload page |
| `/questionbank/analyze` | POST syllabus and question-paper uploads |
| `/questionbank/map` | POST reviewed topics and corrected mappings |
| `/analytics` | Current-process analytics dashboard |
| `/analytics/export/question-mappings.csv` | Download question mapping CSV |
| `/analytics/export/topic-frequency.csv` | Download topic-frequency CSV |
| `/model-lab` | Machine learning experiments and explanations |
| `/about` | Project and concept information |
| `/source-code` | Interactive project source code browser |
| `/health` | JSON health check |

## Project structure

```text
StudySync/
├── app.py                         # Flask configuration, routes, upload workflows
├── config.py                      # Labels, limits, directories and model settings
├── requirements.txt               # Runtime and test dependencies
├── ml/
│   ├── text_extractor.py          # PyMuPDF PDF and Python TXT extraction
│   ├── classifier.py              # NoteMap TF-IDF, MNB/SVM and evaluation
│   ├── train_notemap.py           # Reproducible evaluation CLI
│   ├── similarity_mapper.py       # Topic extraction, question splitting, matching
│   ├── trend_analyzer.py          # Pandas summaries and Matplotlib charts
│   ├── feature_selection.py       # SelectKBest with chi-squared scores
│   ├── dimensionality_reduction.py # Sparse TF-IDF TruncatedSVD
│   ├── clustering.py              # KMeans summaries and diagnostics
│   └── regression.py              # Optional UCI Linear Regression baseline
├── templates/                     # Jinja2 pages and workflow results
├── static/css/style.css           # Responsive design system
├── static/js/script.js            # Navigation and upload interactions
├── data/training/                 # Generated, labelled NoteMap demo data
├── data/datasets/                 # Optional real regression dataset instructions
├── docs/DSBD_SUBMISSION_GUIDE.md   # Diagrams, concepts, results, demo and viva
└── tests/                         # pytest integration and unit tests
```

## Data, safety, and interpretation

The 60-row NoteMap CSV is newly generated **demonstration data**: 10 short
examples per subject across six subjects. It is not collected from students
and does not establish real-world model performance. Details are in
[data/training/README.md](./data/training/README.md).

Uploads are limited to 16 MB per request, extension-checked, stored under
randomized names temporarily, and removed after processing. TXT is decoded by
Python; text-based PDFs are extracted with PyMuPDF. Scanned PDFs with no
extractable text are rejected with an OCR-not-supported message. Uploads are
not executed, served as public static files, or stored in a database.

QuestionBank uses TF-IDF cosine similarity: its score indicates lexical
similarity, not a probability or exam prediction. Year trends use years inferred
from filenames, not verified dates. The QuestionBank workflow cache is
process-local and expires after 30 minutes; it is lost on server restart.

NoteMap models are fitted in memory; no serialized model artifact is loaded.
Classifier result scores are log scores or decision margins, not probabilities.
The regression page reports metrics only if the documented real UCI file is
installed and passes validation; this checkout does not include that file.

## Troubleshooting

- **`python` is not recognized:** install Python and select the installed
  interpreter, or run the commands using `.\.venv\Scripts\python.exe` after
  creating the environment.
- **Module import error:** activate `.venv` or use its interpreter, then run
  `python -m pip install -r requirements.txt` from the project root.
- **PowerShell blocks activation:** skip activation and use the explicit
  `.venv\Scripts\python.exe` command above.
- **Address/port already in use:** stop the other local development server or
  choose an available port by editing the development `app.run` settings.
- **Upload returns 413:** each request is limited to 16 MB; use fewer/smaller
  files. Multiple papers share the request limit.
- **PDF reports no text:** it may be scanned/image-only or empty. OCR is not
  implemented; use a text-based PDF or an OCR-processed copy.
- **QuestionBank review expires:** the in-memory workflow lasts up to 30
  minutes and does not survive a Flask restart; upload the files again.
- **Regression metrics are unavailable:** place the actual UCI mathematics
  course `student-mat.csv` at `data/datasets/student-mat.csv` as described in
  [data/datasets/README.md](./data/datasets/README.md). Do not replace it with
  synthetic data for a real-data claim.
- **Changing training CSV has no effect in a running server:** restart Flask;
  the NoteMap model is cached in memory after first use.

## Project documentation

See the [project documentation guide](./docs/DSBD_SUBMISSION_GUIDE.md) for Mermaid
system/data-flow diagrams, a concept-to-file mapping, dataset disclosures,
measured evaluation and test tables, known limitations, a presentation script,
and concise viva preparation answers.
