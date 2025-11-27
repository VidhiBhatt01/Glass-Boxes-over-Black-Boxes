# Glass Boxes over Black Boxes
## An Empirical Study on Fairness in Criminal Justice AI

### 📊 Overview
This project conducts a comprehensive empirical analysis comparing interpretable machine learning models against black-box models in the context of criminal justice AI systems. Using the ProPublica COMPAS dataset, we evaluate both predictive performance and algorithmic fairness across different model architectures.

### ❓ Research Questions
- Do interpretable models (Decision Trees, Logistic Regression) perform comparably to black-box models (Random Forest, XGBoost) in recidivism prediction?
- How do different model types exhibit bias across racial groups?
- What are the most effective bias mitigation strategies?
- How much of the observed bias stems from data versus algorithmic amplification?

### 📁 Dataset
*Source*: [ProPublica COMPAS Dataset](https://www.kaggle.com/datasets/danofer/compass)

The dataset contains criminal justice records with demographic information, criminal history, and recidivism outcomes, originally used in ProPublica's investigation of the COMPAS risk assessment tool.

### 🔬 Methodology

#### Models Evaluated
*Interpretable Models:*
- Decision Tree (max_depth=5)
- Logistic Regression

*Black-box Models:*
- Random Forest (100 estimators)
- XGBoost (100 estimators)

#### Analysis Framework
1. *Performance Evaluation*: Accuracy, F1-score, precision, recall
2. *Fairness Assessment*: Disparate impact ratios, false positive/negative rate differences
3. *Feature Ablation*: Impact of removing individual features on fairness
4. *Bias Mitigation*: Demographic parity and equalized odds constraints
5. *Causal Analysis*: Decomposition of data bias vs. algorithmic amplification
6. *Statistical Testing*: Cross-validation and significance testing

### 🎯 Key Findings
- Interpretable models achieve comparable accuracy to black-box models (68.1% vs 67.4%)
- All models exhibit significant disparate impact (ratios > 1.9)
- Bias is primarily algorithmic (70.6%) rather than data-driven (29.4%)
- Fairness constraints effectively reduce disparate impact with minimal accuracy loss
- Feature removal (especially Number_of_Priors) can improve fairness

### 📋 Requirements

pandas>=2.0.3
numpy>=1.24.4
scikit-learn>=1.2.1
xgboost
fairlearn>=0.13.0
matplotlib
seaborn
scipy


### 🚀 Installation & Usage
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt (if available) or install packages individually
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/danofer/compass)
4. Place propublica_data_for_fairml.csv in the project directory
5. Run the Jupyter notebook: criminal_justice_ai_bias_study.ipynb

### 📂 Project Structure

├── criminal_justice_ai_bias_study.ipynb  # Main analysis notebook <br>
├── propublica_data_for_fairml.csv        # Dataset (download separately) <br>
└── README.md                             # This file


### 📈 Results Summary
The study demonstrates that interpretable models can achieve competitive performance while providing greater transparency in criminal justice applications. However, all models exhibit concerning levels of racial bias, highlighting the critical need for fairness-aware machine learning approaches in high-stakes domains.

### 👥 Contributors
*Authors*: Vidhi Bhatt, Yuri Kim, Ashvin Loghashankar  
*Course*: CS 269 Fall 2025, UCLA

### 🤝 Contribution Statement
All team members have equally contributed to this project, including research design, implementation, analysis, and documentation.
