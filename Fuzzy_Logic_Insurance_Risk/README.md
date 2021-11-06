FUZZY CONTROL SYSTEM: Insurance Risk Assessment

Author: Adam Gużewski

To run program install:
pip install scikit-fuzzy |
pip install matplotlib |

Fuzzy control system which calculates the risk of insurance based on drivers age, population of the city
when driver lives and power of drivers car.

##Antecednets (Inputs):

    1) 'drivers_age' - values between 18 and 80 // How old is the customer
        Fuzzy set: low, medium, high
        
    2)  'population_of_city' - values between 0 and 500 (expressed in 10 000, e.g. 'population_of_city' = 150, it means
        1 500 000 in real life) // how big is the population in drivers city
        Fuzzy set: low, medium, high
        
    3)  'car_power' - values between 0 and 220 (kilowatt) // how much power drivers car has
        Fuzzy set: low, medium, high

##Consequents (Outputs):

    1) 'risk'
    Universe: How risky is our client on a scale from 0 to 100% ?
    Fuzzy Set: low, medium, high

## Rules:
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
