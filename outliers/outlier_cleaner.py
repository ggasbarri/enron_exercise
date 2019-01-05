#!/usr/bin/python
import math

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    rows_to_delete = int(math.floor(0.1 * len(predictions)))

    errors = np.absolute(predictions - net_worths)

    for i in range(0, rows_to_delete):
        worst_index = np.argmax(errors)
        predictions = np.delete(predictions, worst_index)
        ages = np.delete(ages, worst_index)
        net_worths = np.delete(net_worths, worst_index)
        errors = np.delete(errors, worst_index)


    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], predictions[i]))

    print cleaned_data
    print len(cleaned_data)

    return cleaned_data

