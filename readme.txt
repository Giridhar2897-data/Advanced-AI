HOW TO RUN
==========

1. Activate the virtual environment:
   source .venv/bin/activate

2. Train the models (required once before inference):
   python main.py

   Output:
   - Evaluation results printed to terminal
   - Plots saved to reports/figures/
   - Trained models saved to models/

3. Launch the inference app:
   streamlit run app.py

   Open http://localhost:8501 in your browser.

   The app has two modes:
   - Upload CSV: batch inference on a file with the same columns as creditcard.csv
   - Manual Input: enter a single transaction's values and get an instant prediction
