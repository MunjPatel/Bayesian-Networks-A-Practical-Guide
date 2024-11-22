import pandas as pd
import random
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# defining the network structure
loan_model = BayesianNetwork([('Credit_Score', 'Loan_Default'),
                              ('Income_Level', 'Loan_Default')])

# dummy data for training
random.seed(42)

def generate_loan_data(size):
    data = []
    for _ in range(size):
        
        credit_score = random.choices(['High', 'Medium', 'Low'], weights=[0.4, 0.4, 0.2])[0]
        income_level = random.choices(['High', 'Medium', 'Low'], weights=[0.3, 0.5, 0.2])[0]

        if credit_score == 'Low' and income_level == 'Low':
            loan_default = random.choices(['Yes', 'No'], weights=[0.8, 0.2])[0]
        elif credit_score == 'Low' or income_level == 'Low':
            loan_default = random.choices(['Yes', 'No'], weights=[0.6, 0.4])[0]
        elif credit_score == 'Medium' and income_level == 'Medium':
            loan_default = random.choices(['Yes', 'No'], weights=[0.3, 0.7])[0]
        else:
            loan_default = random.choices(['Yes', 'No'], weights=[0.1, 0.9])[0]
        data.append({'Credit_Score': credit_score, 'Income_Level': income_level, 'Loan_Default': loan_default})

    return pd.DataFrame(data)

loan_data = generate_loan_data(10000)

# fitting the model using MLE
loan_model.fit(loan_data, estimator=MaximumLikelihoodEstimator)

# inferencing
inference = VariableElimination(loan_model)
result = inference.query(variables=['Loan_Default'], evidence={'Credit_Score': 'High', 'Income_Level': 'Low'})

print(result)
