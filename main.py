"""
***********************************************
FUZZY CONTROL SYSTEM: Insurance Risk Assessment
***********************************************
Author: Adam Gużewski

To run program install:
pip install scikit-fuzzy |
pip install matplotlib |

Fuzzy control system which calculates the risk of insurance based on drivers age, population of the city
when driver lives and power of drivers car.

**********************************************

-- Antecednets (Inputs):
    1) 'drivers_age' - values between 18 and 80 // How old is the customer
        Fuzzy set: low, medium, high
    2)  'population_of_city' - values between 0 and 500 (expressed in 10 000, e.g. 'population_of_city' = 150, it means
        1 500 000 in real life) // how big is the population in drivers city
        Fuzzy set: low, medium, high
    3)  'car_power' - values between 0 and 220 (kilowatt) // how much power drivers car has
        Fuzzy set: low, medium, high

-- Consequents (Outputs):
    1) 'risk'
    Universe: How risky is our client on a scale from 0 to 100% ?
    Fuzzy Set: low, medium, high
-- Rules:
    IF 'drivers_age' is low and 'population_of_city' is high and 'car_power' is high then the 'risk' will be high
    IF 'drivers_age' is low and 'population_of_city' is low and 'car_power' is low, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is low and 'car_power' is low, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is low and 'car_power' is high, then the 'risk' is high.
    IF 'drivers_age' is medium and 'population_of_city' is low and 'car_power' is high, then the 'risk' is medium.
    IF 'drivers_age' is medium and 'population_of_city' is low and 'car_power' is low, then the 'risk' is low.
    IF 'drivers_age' is low and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is high.
    IF 'drivers_age' is medium and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is high.
    IF 'drivers_age' is low and 'population_of_city' is high and 'car_power' is low, then the 'risk' is high.

-- Usage:
    If I check how risky to insure is 35 yo man, living in 4 500 000 people city and driving 150 kilowatt car,
    my system will print that the driver is 61.27% risky.
"""

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

'''
INPUTS/Antecedents:
'''
drivers_age = ctrl.Antecedent(np.arange(17, 81, 1), 'drivers_age')
population_of_city = ctrl.Antecedent(np.arange(0, 501, 1), 'population_of_city')
car_power = ctrl.Antecedent(np.arange(0, 221, 1), 'car_power')

'''
OUTPUT/Consequent
'''
risk = ctrl.Consequent(np.arange(0, 101, 1), 'insurance_risk')

'''
Custom Membership functions:
'''
drivers_age['low'] = fuzz.trimf(drivers_age.universe, [17, 24, 31])
drivers_age['medium'] = fuzz.trimf(drivers_age.universe, [25, 34, 50])
drivers_age['high'] = fuzz.trimf(drivers_age.universe, [31, 50, 80])

population_of_city['low'] = fuzz.trimf(population_of_city.universe, [0, 150, 250])
population_of_city['medium'] = fuzz.trimf(population_of_city.universe, [150, 250, 350])
population_of_city['high'] = fuzz.trimf(population_of_city.universe, [250, 350, 500])

car_power['low'] = fuzz.trimf(car_power.universe, [0, 70, 120])
car_power['medium'] = fuzz.trimf(car_power.universe, [70, 120, 170])
car_power['high'] = fuzz.trimf(car_power.universe, [120, 170, 220])


# drivers_age.automf(3)
# population_of_city.automf(3)
# car_power.automf(3)

risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
risk['high'] = fuzz.trimf(risk.universe, [50, 75, 100])

'''
Views of the triangles
'''
risk.view()
population_of_city.view()
car_power.view()
drivers_age.view()

'''
Rules:
    IF 'drivers_age' is low and 'population_of_city' is high and 'car_power' is high then the 'risk' will be high
    IF 'drivers_age' is low and 'population_of_city' is low and 'car_power' is low, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is low and 'car_power' is low, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is low and 'car_power' is high, then the 'risk' is high.
    IF 'drivers_age' is medium and 'population_of_city' is low and 'car_power' is high, then the 'risk' is medium.
    IF 'drivers_age' is medium and 'population_of_city' is low and 'car_power' is low, then the 'risk' is low.
    IF 'drivers_age' is low and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is high.
    IF 'drivers_age' is medium and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is medium.
    IF 'drivers_age' is high and 'population_of_city' is medium and 'car_power' is medium, then the 'risk' is high.
    IF 'drivers_age' is low and 'population_of_city' is high and 'car_power' is low, then the 'risk' is high.
'''
rule1 = ctrl.Rule(drivers_age['low'] | population_of_city['high'] | car_power['high'], risk['high'])
rule2 = ctrl.Rule(drivers_age['low'] | population_of_city['low'] | car_power['low'], risk['medium'])
rule3 = ctrl.Rule(drivers_age['high'] | population_of_city['low'] | car_power['low'], risk['medium'])
rule4 = ctrl.Rule(drivers_age['high'] | population_of_city['low'] | car_power['high'], risk['high'])
rule5 = ctrl.Rule(drivers_age['medium'] | population_of_city['low'] | car_power['high'], risk['medium'])
rule6 = ctrl.Rule(drivers_age['medium'] | population_of_city['low'] | car_power['low'], risk['low'])
rule7 = ctrl.Rule(drivers_age['low'] | population_of_city['medium'] | car_power['medium'], risk['high'])
rule8 = ctrl.Rule(drivers_age['medium'] | population_of_city['medium'] | car_power['medium'], risk['medium'])
rule9 = ctrl.Rule(drivers_age['high'] | population_of_city['medium'] | car_power['medium'], risk['high'])
rule10 = ctrl.Rule(drivers_age['low'] | population_of_city['high'] | car_power['low'], risk['high'])



'''
Creation of control system.
'''
insurance_risk_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])

insurance_risk = ctrl.ControlSystemSimulation(insurance_risk_control)

'''
Specifying the input values.
'''
insurance_risk.input['drivers_age'] = 19
insurance_risk.input['population_of_city'] = 350
insurance_risk.input['car_power'] = 220


insurance_risk.compute()

print(insurance_risk.output['insurance_risk'])

risk.view(sim=insurance_risk)

plt.show()

