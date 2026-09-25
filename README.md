# AI Data Analyst

An AI-powered Data Analyst application built with **Python and Streamlit** for exploratory data analysis, visualization, feature engineering, machine learning, prediction, reporting, and natural-language dataset analysis.

The project combines traditional data analytics with an AI Data Assistant using **RAG (Retrieval-Augmented Generation)** and **LangGraph** for structured AI workflow orchestration.

---

## 🚀 Project Overview

AI Data Analyst is an interactive data analytics application that allows users to upload datasets and perform analysis through a Streamlit dashboard.

The application provides:

- Dataset upload and overview
- Data-quality analysis
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Data visualization
- Feature engineering
- Feature analysis
- Machine learning
- Model evaluation
- Prediction
- Reports
- Natural-language dataset analysis
- AI Data Assistant
- RAG-based retrieval
- LangGraph-based AI workflow

The goal is to combine practical **Data Analytics, Machine Learning, and AI-assisted data analysis** in one application.

---

# ✨ Key Features

## 1. Dataset Upload

Users can upload supported datasets and immediately begin analysis.

The application provides:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Duplicate records
- Dataset preview
- Descriptive statistics

Supported formats include CSV and Excel files.

---

## 2. Dataset Overview

The Dataset Overview section provides a quick understanding of the uploaded dataset.

### Analysis includes

- Dataset shape
- Column information
- Data types
- Missing-value analysis
- Duplicate detection
- Data preview
- Numerical statistics
- Basic data-quality information

---

## 3. Exploratory Data Analysis

The application provides an interactive EDA workflow for discovering patterns and relationships.

### EDA capabilities

- Descriptive statistics
- Numerical feature analysis
- Categorical analysis
- Distribution analysis
- Correlation analysis
- Group-based analysis
- Time-series analysis
- Missing-value analysis
- Outlier exploration

---

## 4. Data Visualization

The application provides visual analysis using:

- Histograms
- Bar charts
- Scatter plots
- Box plots
- Correlation visualizations
- Feature comparisons
- Category analysis
- Time-series visualizations

---

## 5. Feature Engineering

The application provides feature preparation capabilities for analytical and machine-learning workflows.

This includes:

- Feature transformation
- Feature creation
- Numerical transformations
- Categorical processing
- Data preparation

---

## 6. Feature Analysis

The application provides feature-level analysis to understand relationships between variables.

This includes:

- Feature relationships
- Correlation analysis
- Feature importance
- Numerical feature analysis
- Target-based analysis
- Statistical comparisons

---

## 7. Machine Learning

The application includes machine-learning workflows for supported datasets.

### Capabilities

- Classification
- Regression
- Model training
- Model evaluation
- Feature analysis
- Prediction

### Machine Learning Technologies

- Scikit-Learn
- XGBoost
- LightGBM
- NumPy
- Pandas

---

## 8. Prediction

The Prediction section allows trained machine-learning models to generate predictions.

```text
Dataset
   ↓
Data Preparation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
```

---

## 9. Reports

The application provides reporting capabilities based on the performed analysis.

Reports can include:

- Dataset summaries
- Statistical findings
- Analytical results
- Feature analysis
- Machine-learning results
- Prediction results

---

# 🧠 AI Data Assistant

The application includes an **AI Data Assistant** that allows users to ask natural-language questions about the currently uploaded dataset.

The assistant combines exact dataset analysis with the project's AI workflow.

Example questions:

```text
How many patients have diabetes?

What is the average glucose level?

Which country has the highest confirmed cases?

What percentage of transactions are fraudulent?

What is the average transaction amount?

Show the average amount by fraud class.
```

The application performs exact calculations from the uploaded dataset where applicable.

---

# 🔎 RAG - Retrieval-Augmented Generation

The project uses **RAG (Retrieval-Augmented Generation)** as part of the AI Data Assistant workflow.

RAG is used to retrieve relevant context for the AI workflow.

Simplified workflow:

```text
User Question
      ↓
Question Processing
      ↓
Relevant Context Retrieval
      ↓
RAG
      ↓
Context + Dataset Analysis
      ↓
AI Response
```

---

# 🔗 LangGraph

The project uses **LangGraph** to organize the AI workflow into structured processing steps.

Simplified workflow:

```text
User Question
      ↓
AI Data Assistant
      ↓
Question Understanding
      ↓
Dataset Analysis
      ↓
RAG Retrieval
      ↓
LangGraph Workflow
      ↓
Context + Analytical Results
      ↓
AI Explanation
      ↓
Final Response
```

LangGraph is therefore part of the project's AI Data Assistant architecture.

---

# 🧠 AI Architecture

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  AI Data Assistant  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Exact Dataset    │        │ RAG Retrieval    │
       │ Analysis         │        │                  │
       │ Pandas / Python  │        │ Relevant Context │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     LangGraph       │
                    │     Workflow        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Context + Analysis  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Natural-Language    │
                    │ AI Response         │
                    └─────────────────────┘
```

---

# 📚 Dataset-Specific Analysis

The application has been tested with multiple datasets.

## 👥 HR Analytics Dataset

Used for employee and workforce analysis.

Example analysis:

- Employee counts
- Average age
- Department analysis
- Job-role analysis
- Income analysis
- Employee attributes

---

## 🦠 COVID-19 Dataset

Used for country, region, and time-based analysis.

Fields include:

- Province/State
- Country/Region
- Latitude
- Longitude
- Date
- Confirmed
- Deaths
- Recovered
- Active
- WHO Region

Example analysis:

- Confirmed cases
- Deaths
- Recoveries
- Active cases
- Country-level analysis
- WHO region analysis
- Daily changes
- Death rates
- Latest statistics for selected countries

---

## 💳 Credit Card Fraud Detection Dataset

Used for fraud analysis and machine-learning workflows.

Example analysis:

- Fraud transaction count
- Non-fraud transaction count
- Fraud percentage
- Transaction amount analysis
- Fraud vs non-fraud comparison
- Feature associations
- Class-based analysis

Technologies used:

- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- Machine Learning

---

## 🩺 Diabetes Dataset

Used for healthcare-oriented exploratory analysis.

Example analysis:

- Total patients
- Diabetes count
- Diabetes percentage
- Age analysis
- Glucose analysis
- BMI analysis
- Blood pressure analysis
- Insulin analysis
- Outcome-based analysis

---

## 🚗 Car Price Dataset

Used for exploratory data analysis and machine-learning workflows.

Example analysis:

- Price analysis
- Feature relationships
- Numerical analysis
- Categorical analysis
- Correlation analysis
- Prediction workflows

---

## 🛒 Sales Dataset

Used for business-oriented exploratory analysis.

Example analysis:

- Sales analysis
- Product analysis
- Customer analysis
- Category analysis
- Revenue analysis
- Trend analysis

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn
- Streamlit

## Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM

## AI / RAG

- Ollama
- Llama 3.2
- RAG (Retrieval-Augmented Generation)
- LangChain
- LangGraph
- ChromaDB
- nomic-embed-text
- Natural-language dataset analysis
- Exact structured dataset calculations

## Development Tools

- Git
- GitHub
- VS Code

---

# 📁 Project Structure

```text
AI-Data-Analyst/
│
├── data/
│   ├── HR_Data.xlsx
│   ├── car_price_prediction.csv
│   ├── covid_19_clean_complete.csv
│   ├── creditcard.csv
│   ├── diabetes.csv
│   └── sales_data.csv
│
├── src/
│   └── app/
│       └── dashboard.py
│
├── tests/
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/shiva123239/AI-Data-Analyst.git
```

## 2. Navigate to the project

```bash
cd AI-Data-Analyst
```

## 3. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

## 4. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

## 5. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Run the Application

From the project root directory:

```powershell
$env:PYTHONPATH="."
streamlit run src/app/dashboard.py
```

The Streamlit application will open in the browser.

---

# 💡 How to Use

1. Launch the Streamlit application.
2. Upload a CSV or Excel dataset.
3. Review the Dataset Overview.
4. Explore EDA and statistics.
5. Create visualizations.
6. Perform feature engineering and feature analysis.
7. Train machine-learning models when applicable.
8. Generate predictions.
9. Open the AI Data Assistant.
10. Ask natural-language questions about the uploaded dataset.
11. Use the RAG and LangGraph workflow for AI-assisted analysis.

---

# 🔄 End-to-End Workflow

```text
Upload Dataset
      ↓
Dataset Overview
      ↓
EDA & Statistics
      ↓
Visualization
      ↓
Feature Engineering
      ↓
Feature Analysis
      ↓
Machine Learning
      ↓
Prediction
      ↓
Reports
      ↓
AI Data Assistant
      ↓
Exact Dataset Analysis
      +
RAG Retrieval
      ↓
LangGraph Workflow
      ↓
AI Response
```

---

# 🎯 Project Objectives

- Build an end-to-end data analytics application
- Simplify exploratory data analysis
- Provide interactive data visualization
- Automate common analytical questions
- Integrate machine-learning workflows
- Provide prediction capabilities
- Enable natural-language dataset analysis
- Integrate RAG into a data-analysis workflow
- Use LangGraph for AI workflow orchestration
- Combine traditional analytics with AI-assisted analysis
- Create a practical Data Analytics / Data Science portfolio project

---

# 🔍 What This Project Demonstrates

## Data Analytics

- Data cleaning
- EDA
- Statistical analysis
- Data quality analysis
- Data visualization
- Feature analysis

## Python

- Pandas
- NumPy
- Data processing
- Analytical automation
- Object-oriented programming concepts

## Machine Learning

- Classification
- Regression
- Feature engineering
- Model evaluation
- XGBoost
- LightGBM
- Scikit-Learn

## AI

- Natural-language data analysis
- RAG
- LangGraph
- Retrieval-based workflows
- AI-assisted analytical explanations

---

# 🔐 Data & Privacy

The application processes uploaded datasets for analysis and machine-learning workflows.

Users should avoid uploading:

- Passwords
- Confidential business information
- Personally identifiable information
- Sensitive financial information
- Other restricted data

unless appropriate security and privacy controls are in place.

---

# 📌 Future Improvements

Potential future improvements include:

- Additional machine-learning algorithms
- More advanced AI agents
- Improved RAG capabilities
- Additional LangGraph workflows
- Automated data-quality recommendations
- More visualization options
- Advanced report generation
- Additional dataset-specific analytical modules
- Expanded AI-powered insights
- Deployment enhancements

---

# 👨‍💻 Author

## Shiva Rama Krishna Durgam

**B.Com – Computer Applications**

Focus areas:

- Data Analytics
- Data Science
- Machine Learning
- Generative AI
- Python

GitHub: https://github.com/shiva123239

LinkedIn: https://www.linkedin.com/in/durgam-shiva-rama-krishna-7164a424b/

---

# ⭐ Project

If you find this project useful, consider giving the repository a star.
