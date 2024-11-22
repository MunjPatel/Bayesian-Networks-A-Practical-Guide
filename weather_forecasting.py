import pandas as pd
import random
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# defining the network structure
weather_model = BayesianNetwork([('Cloud_Cover', 'Rain'),
                                 ('Temperature', 'Rain')])

# dummy data for training
random.seed(42)

def generate_weather_data(size):
    data = []
    for _ in range(size):
        
        cloud_cover = random.choices(['High', 'Medium', 'Low'], weights=[0.5, 0.3, 0.2])[0]
        temperature = random.choices(['High', 'Medium', 'Low'], weights=[0.3, 0.4, 0.3])[0]

        if cloud_cover == 'High' and temperature == 'Low':
            rain = random.choices(['Yes', 'No'], weights=[0.8, 0.2])[0]
        elif cloud_cover == 'High':
            rain = random.choices(['Yes', 'No'], weights=[0.7, 0.3])[0]
        elif temperature == 'Low':
            rain = random.choices(['Yes', 'No'], weights=[0.6, 0.4])[0]
        else:
            rain = random.choices(['Yes', 'No'], weights=[0.3, 0.7])[0]

        data.append({'Cloud_Cover': cloud_cover, 'Temperature': temperature, 'Rain': rain})

    return pd.DataFrame(data)

weather_data = generate_weather_data(size = 10000)

# fitting the model using MLE
weather_model.fit(weather_data, estimator=MaximumLikelihoodEstimator)

# inferencing
inference = VariableElimination(weather_model)
result = inference.query(variables=['Rain'], evidence={'Cloud_Cover': 'High', 'Temperature': 'Low'})

print(result)
