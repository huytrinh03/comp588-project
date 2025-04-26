import numpy as np
import pyagrum as gum
from pgmpy.readwrite import BIFWriter
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.inference.ApproxInference import ApproxInference

def tensor_mult(a, # n_1 x n_2 x ... x n_d tensor
               b, # m_{1} x m_{2} x ... x m_{l} tensor
               a_dims, # list of dimensions of a to broadcast multiply
               b_dims, # list of dimensions of b to broadcast multiply
):
    """
    This function is written by Siamak Ravanbakhsh (c) - COMP588: probabilistic graphical models - McGill University 
    
    multiplies the two tensors along the specified dimensions. a and b should have the same size in these dimensions.
    the result is a tensor c with dimensions equal to a.ndim + b.ndim - len(a_dims).
    the shape of the first a.ndim dimensions of c is the same as a, while the remaining dimensions match the dimensions of 
    b that did not participate in multiplication.

    example: 
    a = np.ones([2,2,3,5,4])
    b = np.random.rand(7,4,3,2)
    c = tensor_mult(a, b, [1,4], [3,1])
    c.shape # [2,2,3,5,4,7,3] 
    """
    
    assert len(a_dims) == len(b_dims), "a_dims and b_dims should have the same length!"
    assert np.all([a.shape[a_dims[i]] == b.shape[b_dims[i]] for i in range(len(a_dims))]), "a_dims %s and b_dims%s dimensions do not match!" %(a_dims, b_dims)

    d_a = a.ndim
    d_b = b.ndim
    #bring the relevant dimensions to the front
    missing_a = [i for i in range(d_a) if i not in a_dims]
    new_order_a = a_dims + missing_a
    a_t = np.transpose(a, tuple(new_order_a))
    missing_b = [i for i in range(d_b) if i not in b_dims]
    new_order_b = b_dims + missing_b
    b_t = np.transpose(b, tuple(new_order_b))

    #expand the tensors to make the shapes compatible
    a_t = np.reshape(a_t, list(a_t.shape)+len(missing_b)*[1])
    b_t = np.reshape(b_t, [b.shape[i] for i in b_dims]+len(missing_a)*[1]+[b.shape[i] for i in missing_b])

    #multiply
    c_t = a_t * b_t

    #reshape the results: a_dims ; missing_a ; missing_b -> original shape of a ; missing_b
    a_t_index = np.unique(new_order_a, return_index=True)[1].tolist()
    b_t_index = np.arange(d_a, d_a+d_b-len(a_dims)).tolist()
    c = np.transpose(c_t, a_t_index+b_t_index)
    return c

def kld(base_bn, learned_bn):
    '''
    args:
    - bn1, bn2 (pgmpy.models.DiscreteBayesianNetwork): 2 Bayesian networks whose KLD is computed.

    return: computed KLD
    '''
    assert base_bn.check_model(), "bn1 is invalid"
    assert learned_bn.check_model(), "bn2 is invalid"

    BIFWriter(base_bn).write_bif("bn1.bif")
    BIFWriter(learned_bn).write_bif("bn2.bif")

    bn1_gum = gum.loadBN("bn1.bif")
    bn2_gum = gum.loadBN("bn2.bif")

    g = gum.ExactBNdistance(bn1_gum, bn2_gum)
    kld = g.compute()
    return bn1_gum, bn2_gum, kld

def kld_self(base_bn, learned_bn):
    '''
    args:
    - bn1, bn2 (pgmpy.models.DiscreteBayesianNetwork): 2 Bayesian networks whose KLD is computed.

    return: computed KLD
    '''
    assert base_bn.check_model(), "bn1 is invalid"
    assert learned_bn.check_model(), "bn2 is invalid"

    inference1 = BeliefPropagation(base_bn)
    inference2 = BeliefPropagation(learned_bn)

    marginals1 = inference1.query(variables=list(base_bn.nodes), seed=42)
    marginals2 = inference2.query(variables=list(learned_bn.nodes), seed=42)

    marginals_ratio = marginals1 / marginals2
    kl = np.sum(marginals1 * np.log(marginals_ratio))
    
    return kl

def get_elements_last_axis(array):
    '''
    args:
    - array: an ndarray
    
    returns: A 2D np array, with the i-th element containing all the i-th elements of array's last axis
    '''
    # Get the shape of the array
    shape = array.shape

    # Initialize an array to store all the i-th elements of array's last axis
    elements = []

    for i in range(shape[-1]):
        # Iterate over the array along all axes except the last one
        elements_i = []
        for index in np.ndindex(shape[:-1]):
            elements_i.append(array[index][i])
        elements.append(elements_i)

    return np.array(elements)

def marginals(data):
    '''
    args:
    - data (pandas.Dataframe): data over which the marginals is calculated

    Return the marginals over the variables
    '''
    probs = data.value_counts(normalize=True).reset_index(name='prob')

    # Match the above probabilities to a numpy array
    marginals = np.zeros([2]*len(data.columns))
    for i, row in probs.iterrows():
        index = tuple(row.iloc[:-1].astype(int))
        marginals[index] = row['prob']
    
    return marginals
