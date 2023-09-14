from sklearn.cross_decomposition import CCA
from scipy import stats
import numpy as np


def find_CCA_scaling_vectors(input_array_one, input_array_two):
    """
        Calculate CCA (Canonical Correlation Analysis) scaling vectors
        Args:
            input_array_one: the first input array [T,D]
            input_array_two: the second input array [T,D]

        Returns:
            cca_model: CCA model with scaling vectors for the CCA transformation

        """

    # Define CCA model which considers the first CCA coefficient only
    cca_model = CCA(n_components=1)
    # Fit CCA model to the given data
    cca_model.fit(input_array_one, input_array_two)
    # Encode the given arrays into 1D space using the CCA linear transform

    return cca_model



def calculate_CCA_score(input_array_one, input_array_two, CCA_model = None):
    """
    Calculate CCA (Canonical Correlation Analysis) coefficient
    Args:
        input_array_one: the first input array [T,D]
        input_array_two: the second input array [T,D]
        CCA_model: CCA model with scaling vectors for the CCA transformation

    Returns:
        r:  Pearson Correlation Coefficient after CCA transformation (scalar)

    """

    if CCA_model is None:
        # Define CCA model which considers the first CCA coefficient only
        CCA_model = CCA(n_components=1, max_iter=2000, scale=False)

    # identify non-constant dimensions
    std_of_array_one = np.std(input_array_one, axis = 0)
    non_contant_dims_in_array_one = np.where(std_of_array_one > 1e-8)[0]
    std_of_array_two = np.std(input_array_two, axis=0)
    non_contant_dims_in_array_two = np.where(std_of_array_two > 1e-8)[0]

    # use only non-constant dimensions
    input_array_one = input_array_one[:, non_contant_dims_in_array_one] # * 1e8
    input_array_two = input_array_two[:, non_contant_dims_in_array_two] # * 1e8

    # Fit CCA model to the given data
    CCA_model.fit(input_array_one, input_array_two)

    # Encode the given arrays into 1D space using the CCA linear transform
    encoding_one, encoding_two = CCA_model.transform(input_array_one, input_array_two)

    # Standartize arrays shape: make it np.array of floats, remove any dummy dimensions
    encoding_one = np.array(encoding_one, dtype='float64').squeeze()
    encoding_two = np.array(encoding_two, dtype='float64').squeeze()

    # Calculate Pearson Correlation Coefficient (and p-value, which we don't use)
    r, p = stats.pearsonr(encoding_one, encoding_two)

    return r


if __name__ == '__main__':

    #X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    #Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

    #X = [[1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.], [1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.]]
    #Y = [[2.5, 21], [2, 2.5], [2.01, 2], [2.8, 2.01], [0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3], [1, 2], [2, 3], [3, 4], [4, 4], [0., 1.], [1.,0.], [2.,2.], [3.,4.]]

    #X = [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]]
    #Y = [[2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01], [2], [2], [2.01], [2.01]]

    n = 10000
    X = np.random.randint(1,11,(n,42))
    Y = np.random.randint(0,15,(n,42)) + X*0.005 + 7

    CCA_model = find_CCA_scaling_vectors(X,Y)

    corr = calculate_CCA_score(X,Y)

    print("CCA is : ", corr)