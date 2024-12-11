import math
import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def preprocess_data(cpi_path, look_back, look_forward):
    """ Preprocess the data into time steps of inflation rate .. X(t-3), X(t-2), X(t-1), X(t), X(t+1), X(t+2) ..
    Inflation rate is 100*log(X(t)/X(t-1))

    :param cpi_path: string
        path to data
    :param look_back: int
        The time dimension for the GRU
    :param look_forward: int
        The forecasting horizon
    :return: pandas df
    """

    # read cpi data and sort by category and date
    cpi_data = pd.read_csv(cpi_path)
    cpi_data.sort_values(by=['Category_id', 'Date'], axis=0, ascending=[True, True], inplace=True)
    cpi_data.reset_index(inplace=True)

    # calculate inflation and shift back "look_back" times and shift forward "look_forward" times
    cols = []
    for i in range(0, look_back + 1):
        cpi_data['Inflation t-{}'.format(i)] = \
            100 * np.log(cpi_data.groupby(['Category_id'])['Price'].shift(i) / \
                         cpi_data.groupby(['Category_id'])['Price'].shift(i + 1))
        if i == 0:
            cols.append('Inflation t')
        else:
            cols.append('Inflation t-{}'.format(i))
    for i in range(1, look_forward + 1):
        cpi_data['Inflation t+{}'.format(i)] = \
            100 * np.log(cpi_data.groupby(['Category_id'])['Price'].shift(-i) / \
                         cpi_data.groupby(['Category_id'])['Price'].shift(-i + 1))
        cols.append('Inflation t+{}'.format(i))

    # Rename, and reorder the cols
    cpi_data.rename(columns={'Inflation t-0': 'Inflation t'}, inplace=True)
    cpi_data['Year'] = cpi_data['Date'].apply(lambda x: int(x[0:4]))
    order = ['Category_id', 'Category', 'Year', 'Date', 'Price'] + cols + ['Indent', 'Weight', 'Parent', 'Parent_ID']
    cpi_data = cpi_data[order]

    return cpi_data
