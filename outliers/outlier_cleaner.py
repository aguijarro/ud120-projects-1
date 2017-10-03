#!/usr/bin/python


def getResidualError(prediction, netWorth):
    return float(prediction - netWorth)


def getList(parList):
    listTemp = []
    for x in parList.tolist():
        listTemp.append(x[0])
    return listTemp


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

    predictionsList = getList(predictions)
    agesList = getList(ages)
    netWorthsList = getList(net_worths)

    for x in range(0, len(predictionsList)):
        cleaned_data.append((agesList[x],
                             netWorthsList[x],
                             getResidualError(predictions[x],
                                              netWorthsList[x])))

    cleaned_data = sorted(cleaned_data, key=lambda error: error[2])

    cleanElement = (len(cleaned_data) * 10) / 100

    cleaned_data = cleaned_data[:len(cleaned_data) - cleanElement]

    print "Tamanio: ", len(cleaned_data)

    return cleaned_data

