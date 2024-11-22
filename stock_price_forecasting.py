import pandas as pd
import random
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# defining the network structure
stock_model = BayesianNetwork([('Interest_Rate', 'Stock_Return'),
                               ('Inflation', 'Stock_Return')])

# dummy data for training
random.seed(42)

def generate_stock_data(size):
    data = []
    for _ in range(size):

        interest_rate = round(random.uniform(0.5, 3.0), 1)
        inflation = round(random.uniform(1.0, 3.5), 1)

        if interest_rate < 1.5 and inflation < 2.0:
            stock_return = round(random.uniform(6.0, 8.0), 1)
        elif interest_rate > 2.5 or inflation > 3.0:
            stock_return = round(random.uniform(2.0, 4.0), 1)
        else:
            stock_return = round(random.uniform(4.0, 6.0), 1)

        data.append({
            'Interest_Rate': interest_rate,
            'Inflation': inflation,
            'Stock_Return': stock_return
        })

    return pd.DataFrame(data)

stock_data = generate_stock_data(size = 10000)

# fitting the model using MLE
stock_model.fit(stock_data, estimator=MaximumLikelihoodEstimator)

# inferencing
inference = VariableElimination(stock_model)
result = inference.query(variables=['Stock_Return'], evidence={'Interest_Rate': 1.7, 'Inflation': 2.0})

print(result)
