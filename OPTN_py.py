import numpy as np
import graph_tool as gt
import math

def ordinal_patterns_t(ts, d, tau):
    '''
    Creates a list of tuple ordinal patterns as they appear in the time series.
    '''
    n = len(ts)
    patterns = []
    for i in range(int((n - (d-1)*tau))):
        window = ts[i:i+d*tau:tau]
        pattern = tuple(np.argsort(window,kind='mergesort'))
        patterns.append(pattern)
    return patterns

def create_optn_gt(ts, d, tau):
    '''
    Creates a graph_tool Graph from a time series using ordinal patterns as vertices.

    Parameters:
    -----------
    ts:  np.1darray or list
         The time series to create ordinal patterns from
    d:   int
         The embedding dimension (data points per ordinal pattern) used
    tau: int
         The embedding delay used e.g. tau = 2: all data points in each ordinal pattern are separated by
         one data point
    '''
    # Check for valid values of embedding parameters
    if not isinstance(d, int):
        raise TypeError('Embedding dimension needs to be an integer')
    if d<1:
        raise ValueError('Embedding dimension needs to be greater than 0')

    if not isinstance(tau, int):
        raise TypeError('Embedding delay needs to be an integer')
    if tau<1:
        raise ValueError('Embedding delay needs to be greater than 0')

    #create ordinal pattern time series
    patterns = ordinal_patterns_t(ts, d, tau)
    
    G = gt.Graph(directed=True)

    pattern_to_vertex = {}
    
    weight = G.new_edge_property("float") 
    labels = G.new_vertex_property("string") 

    for i in range(len(patterns) - 1):
        pattern_from = patterns[i]
        pattern_to = patterns[i + 1]

        # Add start and target vertex to graph if ordinal pattern has not occured already
        if pattern_from not in pattern_to_vertex:
            v_from = G.add_vertex()
            pattern_to_vertex[pattern_from] = v_from
            labels[v_from] = str(pattern_from) 
        else:
            v_from = pattern_to_vertex[pattern_from]

        if pattern_to not in pattern_to_vertex:
            v_to = G.add_vertex()
            pattern_to_vertex[pattern_to] = v_to
            labels[v_to] = str(pattern_to) 
        else:
            v_to = pattern_to_vertex[pattern_to]

        # Add edge between start and target vertex if edge does not exist already
        e = G.edge(v_from, v_to)
        if e is None:
            if v_to != v_from: # Avoid self loops
                e = G.add_edge(v_from, v_to)
                weight[e] = 1

            else:
                continue
        else:
            weight[e] += 1 # Increase weight by 1 if edge exists already

    # Renormalize weights to total number of transitions
    G.edge_properties['weight'] = weight 
    
    weightst = np.array(G.ep['weight'].a)

    total_weight = np.sum(weightst)

    for i, e in enumerate(G.edges()):
        weight[e] = weight[e]/total_weight
        
    # Assign each vertex the label of the corresponding ordinal pattern
    G.vp["label"] = labels 
    return G

def transcripts(ts,d,tau):
    '''
    Creates a list of tuple transcripts as they appear in the time series based on the ordinal pattern series.
    '''
    op = ordinal_patterns_t(ts,d,tau)
    transcripts=[]
    for i in range(len(op)-1):
        s = op[i]
        s_inv = np.argsort(s,kind='mergesort')
        transcripts.append(np.array(op[i+1])[s_inv])
    for i in range(len(transcripts)):
        transcripts[i] = tuple(transcripts[i])
    return transcripts

def create_optn_gt_ts(ts, d, tau):
    '''
    Creates a graph_tool Graph from a time series using transcripts as vertices.

    Parameters:
    -----------
    ts:  np.1darray or list
         The time series to create ordinal patterns from
    d:   int
         The embedding dimension (data points per ordinal pattern) used
    tau: int
         The embedding delay used e.g. tau = 2: all data points in each ordinal pattern are separated by
         one data point
    '''
    # Check for valid values of embedding parameters
    if not isinstance(d, int):
        raise TypeError('Embedding dimension needs to be an integer')
    if d<1:
        raise ValueError('Embedding dimension needs to be greater than 0')

    if not isinstance(tau, int):
        raise TypeError('Embedding delay needs to be an integer')
    if tau<1:
        raise ValueError('Embedding delay needs to be greater than 0')
        
    #create transcript time series
    patterns = transcripts(ts, d, tau)
    G = gt.Graph(directed=True)

    pattern_to_vertex = {}
    weight = G.new_edge_property("float") 
    labels = G.new_vertex_property("string") 

    for i in range(len(patterns) - 1):
        pattern_from = patterns[i]
        pattern_to = patterns[i + 1]

        # Add start and target vertex to graph if transcript has not occured already
        if pattern_from not in pattern_to_vertex:
            v_from = G.add_vertex()
            pattern_to_vertex[pattern_from] = v_from
            labels[v_from] = str(pattern_from) 
        else:
            v_from = pattern_to_vertex[pattern_from]

        if pattern_to not in pattern_to_vertex:
            v_to = G.add_vertex()
            pattern_to_vertex[pattern_to] = v_to
            labels[v_to] = str(pattern_to) 
        else:
            v_to = pattern_to_vertex[pattern_to]

        # Add edge between start and target vertex if edge does not exist already
        e = G.edge(v_from, v_to)
        if e is None:
            if v_to != v_from:# Avoid self loops
                e = G.add_edge(v_from, v_to)
                weight[e] = 1
            else:
                continue
        else:

            weight[e] += 1 # Increase weight by 1 if edge exists already

    # Renormalize weights to total number of transitions
    G.edge_properties['weight'] = weight 
    
    weightst = np.array(G.ep['weight'].a)

    total_weight = np.sum(weightst)


    for i, e in enumerate(G.edges()):
        weight[e] = weight[e]/total_weight

    # Assign each vertex the label of the corresponding ordinal pattern
    G.vp["label"] = labels 
    return G


# Periodicitytest
# ----------------

    
def find_EDim(A):
    '''
    Finds the largest possible embedding dimension such that the possible number of ordinal patterns is
    smaller than 10% of available data points. This ensures that a noisy signal likely creates every 
    possible ordinal pattern.
    '''
    n = len(A)
    m = 0
    while math.factorial(m + 1) < n/10:
        m += 1

    return m

def find_maxtau(A):
    '''
    Finds the largest possible embedding delay such that at most 10% of the time series is lost by the embedding.
    '''
    n = len(A)
    tau = 2
    d = find_EDim(A)
    while (d-1)*tau<0.1*n:
        tau+=1
    return tau-1

def find_min_bins(list1, list2):
    '''
    Finds the smallest result of the periodicitytest where the difference between number of
    in and out degrees is 0 or 1.
    '''
    list1 = np.array(list1)
    list2 = np.array(list2)

    min_values = np.minimum(list1, list2)
    
    condition = np.abs(list1 - list2) < 2
    min_bins = np.where(condition, min_values, np.nan)
    
    return np.nanmin(min_bins)

def periodicitytest_adv(TS,delays = None):

    '''
    Returns int result of the periodicitytest.
    Results 0 and 1 indicate strict periodicity. Increasing values signify quasi periodicity, chaos and noise
    in that order, where ~20 is a random time series.

    Parameters:
    -----------
    TS:     np.1darray
            Time series to test for periodicity.
    delays: None or int or list
            Embedding delays for which the periodicitytests operates. 
            If None, all possible embedding delays up to the result of find_maxtau are checked and 
            the smalles result is returned.
            If int, only this value will be checked and the result returned.
            If list, every value in the list is tried and smallest result is returned.
    '''

    # A minimum embedding dimension of 4 is required. Together with the critereon in find_EDim
    # this evaluates to at least 241 datapoints needed.
    if len(TS)<241:
        raise ValueError('Time series must have at least 241 Datapoints.')
    
    embedding_dim = find_EDim(TS)
    if delays is None:
        embedding_delays = np.arange(2,find_maxtau(TS)+1)
    else:
        if isinstance(delays, int):
            if delays<2:
                raise ValueError('Embedding delay needs to be larger than 1.')
            embedding_delays=[delays]
        else:
            embedding_delays = delays

    in_degrees = []
    out_degrees = []

    for i in embedding_delays:
        
        symbols = ordinal_patterns_t(TS,embedding_dim,i)
        
        unique_symbols, indices = np.unique(symbols, axis=0, return_inverse=True)
    
        adjacency = np.zeros((len(unique_symbols), len(unique_symbols)), dtype=int)

        # Create adjacency matrix of the Graph of the time series.
        for j in range(len(indices) - 1):
            from_idx = indices[j]
            to_idx = indices[j+1]
            if from_idx!=to_idx:
                adjacency[from_idx, to_idx] = 1

        # In and out degrees of each vertex
        outdeg = np.sum(adjacency,axis=1)
        indeg = np.sum(adjacency,axis=0)

        # Number of different in and out degrees
        in_degrees.append(len(np.unique(indeg)))
        out_degrees.append(len(np.unique(outdeg)))

    return find_min_bins(in_degrees,out_degrees)
