# NoteMap demonstration training data

`demo_notes.csv` contains 60 short labeled text examples: 10 each for Data
Structures, Database Management Systems, Operating Systems, Computer Networks,
Machine Learning, and Artificial Intelligence.

**Every row in this file is newly generated demonstration data.** These short
examples are included to make the pipeline runnable and explainable; they are
not collected from students, textbooks, or external datasets. They are not
representative enough to establish real-world classification accuracy.

## Dataset format

The UTF-8 CSV contains two columns:

- `subject`: a category listed in `SUBJECT_LABELS` in `config.py`
- `text`: one example note

The training command checks each label and rejects empty examples. Add labeled
examples carefully and keep subject categories consistent with the config.
More varied, reviewed notes per subject are needed before drawing conclusions
about real student documents.

## Run training and evaluation

From the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.train_notemap
```

The command reports the actual reproducible train/test metrics and confusion
matrices for both models. It does not save a serialized model file. The Flask
app builds the same small pipeline in memory on its first NoteMap prediction.
The development server must be restarted after editing the CSV.
