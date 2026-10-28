# data/datasets/README.md
# Datasets for StudySync

## Regression Experiment — UCI Student Performance

**File needed:** `student-mat.csv`

**Source:** https://archive.ics.uci.edu/dataset/320/student+performance

**Citation:**
> P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School
> Student Performance." In A. Brito and J. Teixeira Eds., Proceedings of
> 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12,
> Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

**How to download:**
1. Visit the URL above.
2. Download the zip file.
3. Extract `student-mat.csv` (Math course performance data).
4. Place it in this folder: `data/datasets/student-mat.csv`

**Features used:**
| Column     | Description                        |
|------------|------------------------------------|
| studytime  | Weekly study time (1–4 scale)      |
| failures   | Number of past class failures      |
| absences   | Number of school absences          |
| G1         | First period grade (0–20)          |
| G2         | Second period grade (0–20)         |
| G3         | Final grade — **prediction target**|

**Disclaimer:**
This dataset is used for demonstration purposes only.
It does not predict your personal marks.
