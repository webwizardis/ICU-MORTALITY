# ICU Risk Studio

ICU Risk Studio is a lightweight local web app for ICU mortality risk review. It rebuilds the original prototype into a cleaner, more maintainable project with:

- a local cohort-trained prediction engine
- engineered ICU features inspired by the supplied notebook PDF
- a polished, responsive dashboard for intake and result review
- transparent handling of blank advanced fields using cohort medians

## Run locally

1. Install Python 3.10 or newer.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open `http://127.0.0.1:8000` in your browser.

> Note: `0.0.0.0` is a bind address for the server, not a direct browser target. Use `127.0.0.1` or `localhost` instead.

## Notes

- The app uses the local CSV dataset in this folder for training at startup.
- The deployed model is a weighted logistic regression implemented directly with `numpy` for portability and interpretability.
- This project is a decision-support demo and not a replacement for clinical judgement.
