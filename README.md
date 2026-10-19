# Insightra AI — Intelligent Data Analysis Agent

Insightra AI is an AI-powered data analysis dashboard built with Python and Streamlit. It helps users explore datasets, generate visualizations, discover insights, and experiment with machine learning through an interactive interface.

## Features

- **CSV Upload:** Upload datasets for analysis.
- **Dataset Overview:** Explore rows, columns, data types, and summary statistics.
- **Data Visualization:** Generate charts to understand patterns and relationships.
- **AI Analyst:** Ask natural-language questions about your dataset.
- **Automatic Insights:** Discover useful observations from data.
- **PDF Reports:** Generate downloadable data analysis reports.
- **Data Quality Checker:** Identify potential data quality issues.
- **Data Cleaning:** Explore data cleaning options and export cleaned datasets.
- **Prediction Lab:** Experiment with regression and classification models.
- **Model Evaluation:** Compare model performance and review feature importance.
- **Prediction Export:** Export predictions for further analysis.

## Technology Stack

- Python
- Streamlit
- Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Google Gemini API
- ReportLab
- SQLite and Joblib

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chiiradeep/insightra-ai.git
cd insightra-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Configure your Gemini API key securely through Streamlit secrets or environment variables, according to the application's configuration.

**Never commit real API keys, passwords, or secret configuration files to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Project Structure

```text
insightra-ai/
├── app.py
├── tools.py
├── report_generator.py
├── agent.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

The exact files may vary depending on the project configuration.

## Security

- Do not upload API keys or credentials.
- Keep private datasets and local databases out of the public repository.
- Store secrets using environment variables or the deployment platform's secrets manager.

## Project Status

Insightra AI is an evolving project. Features and deployment configuration may change as development continues.

## Author

**Chiiradeep**

GitHub: https://github.com/Chiiradeep

---

*Insightra AI — Turn data into insights.*
