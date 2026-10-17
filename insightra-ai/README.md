# Insightra AI

Insightra AI is a Streamlit-based data analysis application for exploring CSV datasets, creating visualizations, generating AI-assisted insights and reports, checking data quality, and experimenting with machine-learning predictions.

## Features

- CSV upload and dataset overview
- Exploratory analysis and visualizations
- AI Analyst powered by Google Gemini
- Automatic insights and downloadable PDF reports
- Data-quality checks and cleaned-data export
- Machine-learning experiments and prediction workflows (where available in the app)

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. If prompted for a Gemini API key, use your own key. Do not paste API keys into source code or commit them to GitHub.

## Deploy

This project can be deployed with Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main file. Configure any required API key using the deployment platform's Secrets settings; never commit `secrets.toml` or a real `.env` file.

## Important notes

- Use datasets you are permitted to upload and analyze.
- The dependency list uses unpinned package versions; test the project in a clean environment before deployment and pin versions after confirming compatibility.
- AI-generated analysis may be incorrect; verify important conclusions independently.
