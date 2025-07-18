# Telco Customer Churn Analysis Project

## Project Overview
This project analyzes customer churn patterns in a telecommunications company using machine learning techniques, clustering analysis, and data visualization. For the best understanding of the project methodology, findings, and business implications, we strongly recommend starting with the **Telecoms_Churn_Report.pdf** in the reports directory.

## Authors
- Sam Khanin
- Albinson Felix
- Zach Kontor
- JJ Kailash

## Project Structure
```
telco-churn-rep/
│
├── data/               # Data files
├── models/            # Saved model files
├── reports/           # Analysis reports
├── src/              # Source code
│   ├── cleaning.py
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_selection.py
│   ├── main.py
│   ├── model_building.py
│   ├── preprocessing.py
│   └── visualization.py
└── visualizations/    # Generated plots and Tableau workbooks
```

## Installation
1. Clone this repository
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Execution
You can run the main prediction pipeline using:
```bash
python src/main.py
```

The main pipeline includes:
1. Data cleaning
2. Feature engineering
3. Model training and evaluation
4. Visualization generation

**Note**: The model optimization is currently configured to maximize recall. This can be modified by changing the `SCORING` variable in `src/config.py`. Available options include:
- 'precision'
- 'recall'
- 'f1'
- 'accuracy'
- 'roc_auc'

## Key Documents
- **Telecoms_Churn_Report.pdf**: Comprehensive analysis of the entire project (Highly recommended reading)
- **Modeling_and_Clustering_Results.xlsx**: Detailed results from modeling and clustering analyses
- **visualizations/tableau/Tableau_Visualizations.twbx**: Interactive visualizations for Exploratory Data Analysis

## Results
The project includes:
- Customer segmentation through clustering
- Churn prediction models
- Feature importance analysis
- Interactive visualizations
- Comprehensive recommendations

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or collaboration opportunities, please email me at samkhaninchess@gmail.com

