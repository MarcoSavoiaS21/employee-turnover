# Employee Turnover Prediction and Feature Analysis

This project aims to predict employee turnover using machine learning models, namely Random Forest and Logistic Regression. The analysis provides insights into key factors influencing turnover, evaluates model performance, and suggests strategies to improve employee retention.

## Project Overview

Employee turnover can have significant financial implications for companies. In this project, we predict whether an employee will leave the company using HR analytics data. The dataset consists of features like job satisfaction, performance scores, and work accidents. We use Random Forest and Logistic Regression models to explore employee behavior and identify the most important factors affecting turnover.

## Files in the Project

- `HR_comma_sep.csv`: The dataset used in this project, which can be downloaded from Kaggle.
- `employee_turnover.ipynb`: The Jupyter Notebook containing the main analysis and model training.
- `turnover_prediction.py`: The Python script defining functions for loading data, preprocessing, model training, hyperparameter tuning, evaluation, and feature importance extraction.

## Instructions for Running the Project

1. **Clone the Repository**  
   First, clone the project repository to your local machine:

   ```
   git clone https://github.com/MarcoSavoiaS21/employee-turnover.git
   cd employee-turnover
   ```

2. **Install Required Packages**  
   Ensure you have the necessary Python packages installed. You can do so by running the following command:

   ```
   pip install pandas matplotlib seaborn scikit-learn
   ```

3. **Run the Analysis**  
   After installing the required packages, open the Jupyter Notebook:

   ```
   jupyter notebook employee_turnover.ipynb
   ```

   You can also run the script using the Python file:

   ```
   python turnover_prediction.py
   ```

   Ensure that all files, including `HR_comma_sep.csv`, are in the same folder as the `.ipynb` and `.py` files.

## Project Structure

- **Data Preprocessing**: Handling categorical features, splitting the data into training and test sets, and normalizing the input features.
- **Model Training**:
  - **Random Forest**: Initial model setup and hyperparameter tuning using GridSearchCV.
  - **Logistic Regression**: Simpler model used for comparison, offering insights into feature impact.
- **Model Evaluation**: The models are evaluated on their accuracy, precision, recall, and F1-score.
- **Feature Importance**: The Random Forest model's feature importance and Logistic Regression's coefficients are analyzed to understand key factors affecting employee turnover.

## Key Findings

- **Top Features**: Employee satisfaction level, number of projects, and years at the company are the most critical factors influencing turnover.
- **Model Performance**: The Random Forest model performs significantly better than Logistic Regression, achieving 99% accuracy.
- **Impact of Work Accidents**: Interestingly, work accidents appear to reduce the likelihood of employees leaving, possibly due to psychological factors like the sunk cost effect.

## Future Improvements

- **Incorporate additional features**: Qualitative data such as employee feedback and external market trends could enhance the model's performance.
- **Apply more advanced models**: Exploring more complex ensemble or hybrid models could yield better predictive accuracy.

## License

This project is licensed under the MIT License.
   
