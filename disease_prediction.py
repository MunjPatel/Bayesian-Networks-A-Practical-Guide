import pandas as pd
import random
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# defining the network structure
disease_model = BayesianNetwork([('Symptom', 'Disease'),
                                 ('Family_History', 'Disease')])

# dummy data for training
random.seed(42)

def generate_symptom_data(size):
    data = []
    for _ in range(size):
        
        symptom = random.choices(['Yes', 'No'], weights=[0.6, 0.4])[0]
        family_history = random.choices(['Yes', 'No'], weights=[0.3, 0.7])[0]

        if symptom == 'Yes' and family_history == 'Yes':
            disease = random.choices(['Yes', 'No'], weights=[0.9, 0.1])[0]
        elif symptom == 'Yes' and family_history == 'No':
            disease = random.choices(['Yes', 'No'], weights=[0.7, 0.3])[0]
        elif symptom == 'No' and family_history == 'Yes':
            disease = random.choices(['Yes', 'No'], weights=[0.5, 0.5])[0]
            disease = random.choices(['Yes', 'No'], weights=[0.2, 0.8])[0]

        data.append({'Symptom': symptom, 'Family_History': family_history, 'Disease': disease})

    return pd.DataFrame(data)

disease_data = generate_symptom_data(size = 10000)

# fitting the model using MLE
disease_model.fit(disease_data, estimator=MaximumLikelihoodEstimator)

# inferencing
inference = VariableElimination(disease_model)
result = inference.query(variables=['Disease'], evidence={'Symptom': 'Yes', 'Family_History': 'Yes'})

print(result)
