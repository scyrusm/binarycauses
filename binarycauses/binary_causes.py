"""
The idea behind this module is to develop a simple algorithm that can detect unary causal relationships (i.e., A causes B) and binary causal relationships (A and B act together to cause C).  The justification for this is that, in practice, many biological causal effects will be of this form; while nothing about graphical causal models (see Judea Pearl's "Causality: Models, Reasoning and Inference") demands that causal effects be unary, in practice, all algorithms rely on some conditional independence tests to resolve a causal graph from data, and those tests (most commonly the Fisher Z test) rely on unary interactions with Gaussian noise.  

The following script is a partial-implementation of PC algorithm (see Spirtes, Glymour, Scheines "Prediction, Causation, and Search"), where we check for both unary interaction and binary interactions--we still assume normal variables, and binary interactions are regarded as simply being the product of those variables (which will themselves still be normal, and therefore conditional independence is testable via the Fisher Z test).  For details of the Fisher Z test, see "Causal Search in Structural Vector Autoregressive Models" (Moneta, Chlaß, Entner, Hoyer, 2011)).  Here, the only addition that is made to the PC algorithm is that, for cases where there are two individual unary interactions (causes) influencing the same effect, an additional check is made to see if there is a significant binary effect (since a binary effect, whether holding the co-causer constant or not, should register as a unary effect as well--though I have not checked the robustness of this assumption).  Furthermore, we are interested in something akin to a time series--i.e., n variables which interact causally with the same n variables at a later time.  It is forbidden that these "later time" variables affect the "earlier time" variables, but it is not forbidden that variables effect variables at the same time step.  Note also, that in this prototypical implementation, I am simply assuming three time steps--one start, one hidden, one final.  

Currently the artificial data generation has various hard-coded parameters, whose effect should be tested.  Moreover, the normal variable assumption almost certainly fails in e.g. a genetic constant (where the interesting variables of interest may be mutation i.e. mostly binary).  It might be interesting to test this model in a population dynamics context.   
"""

from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from binarycauses.partial_corr import partial_corr
from plotting.plotting import plot_adjacencies
import itertools

# TODO
# check the hidden layer correctness
# add tests
# Un-hard-code various parameters set in the artificial data generation

# Account for significance/lack thereof of the partial correlation test.  This is important.  


class Artificial:
    """
    This class generates an artificial causal graph and data therefrom.  Various parameters, for example the strength of various noise terms, are currently hardcoded.  
    Attributes
    ---------
    n_nodes:  the number of nodes in the graph at a single time step
    unary_adjaciences: an adjacency matrix connecting time steps
    unary_coefficients:  a matrix describing the time evolution of the variables.  Currently, if unary_coefficients = U, x_t+1 = U x_t + eps, where x_t, x_t+1 are column vectors, eps is a noise term
    binary_adjacencies:  a dictionary, mapping a length-two tuple (indicating a pair of causes) to one or zero (indicating adjacency or not)
    binary_coefficients:  a dictionary with the coefficients describing the time evolution of the variables according to the binary interaction, i.e. x_i,t+1 += sum(x_j,t x_k, t * V[(j, k)]) over all possible j, k
    T:  the number of data points
    X:  the "start" vectors
    Y:  the "final" vectors
    """

    def __init__(self):
        self.n_nodes = None
        self.unary_adjacencies = None
        self.unary_coefficients = None
        self.binary_adjacencies = None
        self.binary_coefficients = None
        self.T = None
        self.X = None
        self.Y = None
        # The following three parameters are hard-coded, but may be worth experimenting with
        self.noise_magnitude=0.01,
        self.noise_variance=1,
        self.hidden_layers=1

    def generate_artificial_graph(self, n_nodes):
        """
        This is a more stateful wrapper for artificial_graph_generator
        Arguments
        --------
        n_nodes:  the number of nodes/variables
        """

        self.n_nodes = n_nodes
        (self.unary_adjacencies, self.unary_coefficients), (
            self.binary_adjacencies,
            self.binary_coefficients) = self.artificial_graph_generator(
                self.n_nodes)

    def generate_artificial_data(self, T, use_binary=False):
        """
        Generates artificial data (a more stateful wrapper of artificial_data_generator); assumes that the artificial graph has already been generated
        Arguments
        --------
        T:  the number of datapoints
        """
        self.T = T
        self.X, self.Y = self.artificial_data_generator(
            self.n_nodes,
            self.T, (self.unary_adjacencies, self.unary_coefficients),
            (self.binary_adjacencies, self.binary_coefficients),
            use_binary=False,
            noise_magnitude=self.noise_magnitude,
            noise_variance=self.noise_variance,
            hidden_layers=self.hidden_layers)

    def generate(self, n_nodes, T):
        """
        A wrapper for both the graph and data generation
        Arguments
        -----
        n_nodes: number of variables to consider
        T:  number of data points to consider
        """
        self.generate_artificial_graph(n_nodes)
        self.generate_artificial_data(T)

    def artificial_graph_generator(self, n_nodes):
        """
        This function generates an artificial graph, including unary (linear) interactions
        Arguments
        -------
        n_nodes:  number of variables/nodes
        """
        # Various parameters related to the artificial adjacency are hardcoded.  This should change.
        unary_adjacencies = np.random.choice(
            [0, 1], p=[0.8, 0.2], size=(n_nodes, n_nodes))
        binary_adjacencies = {
            (x1, x2): np.random.choice([0, 1], p=[0.2, 0.8], size=n_nodes) *
            unary_adjacencies[:, x1] * unary_adjacencies[:, x2] * int(x1 != x2)
            for x1 in range(n_nodes) for x2 in range((x1 + 1), n_nodes)
        }
        for x1 in range(n_nodes):
            binary_adjacencies[(x1, x1)] = np.zeros(n_nodes)
            for x2 in range((x1 + 1), n_nodes):
                binary_adjacencies[(x2, x1)] = binary_adjacencies[(x1, x2)]

        unary_coefficients = np.multiply(
            np.random.normal(0, 1, (n_nodes, n_nodes)), unary_adjacencies)
        binary_coefficients = {
            key: np.random.normal(0, 1) * binary_adjacencies[key]
            for key in binary_adjacencies.keys()
        }
        return (unary_adjacencies, unary_coefficients), (binary_adjacencies,
                                                         binary_coefficients)

    def artificial_data_generator(self,
                                  n_nodes,
                                  T,
                                  unary,
                                  binary,
                                  use_binary=False,
                                  noise_magnitude=0.01,
                                  noise_variance=1,
                                  hidden_layers=1):
        """
        Generates artificial data for an already-generated causal graph
        Arguments
        --------
        n_nodes:  number of variables
        T:  number of data points
        unary: a tuple of the unary adjacency and coefficient matrices
        binary:  a typle of the binary adjacency and coefficient dictionaries
        """
        assert type(unary) == tuple
        if use_binary:
            assert type(binary) == tuple
        X = np.random.normal(0, 1, (T, n_nodes))
        Y = 0.0 * np.random.normal(0, 1, (T, n_nodes))

        def effect(X, Y):
#            Y = np.copy(Yin)
            for irow in range(T):
                Y[irow] += np.matmul(
                    unary[1],
                    X[irow, :].T) + noise_magnitude * np.random.normal(0, noise_variance, n_nodes)
                if use_binary:
                    for inode in range(n_nodes):
                        Y[irow] += np.sum(
                            np.matmul(0.5 * np.outer(X[irow, :], X[irow, :]),
                                      binary[1][:, :, inode]))
            return Y

        for hl in range(hidden_layers+1):
            Y = effect(X, Y)
        return X, Y


def fisher_z(rho, K, T):
    """
    Fisher Z correlation test.  See "Causal Search in Structural Vector Autoregressive Models" (Moneta, Chlaß, Entner, Hoyer, 2011)
    Arguments
    -------
    rho:  correlation, partial or otherwise
    K:  number of variables which are being controlled for
    T:  number of datapoints

    Returns
    ------
    fisher z score (approximately normally distributed according to null hypothesis that variables in question are uncorrelated)
    """
    return 0.5 * np.sqrt(T - K - 3) * np.log(np.abs((1 + rho) / (1 - rho)))


def double_sided_normal_p(s):
    """
    Arguments:
    s:  fisher z score
    Returns
    -----
    Double-sided p-value test for a given fisher z
    """
    return 2 - 2 * stats.norm.cdf(np.abs(s))


def normal_p_threshold(s, threshold=0.05):
    """
    wrapper function; takes in a fisher z score and return 1 or 0 according to whether than score is significant, according the given significance threshold (defaults to 0.05
    """
    return int(double_sided_normal_p(s) < threshold)


def z_score_to_adjacencies(z_scores, threshold=0.05):
    """
    wrapper function; translates an array of fisher z scores to an array of 1s or 0s according to whether those z scores are significant according to the set threshold, which is 0.05 by default
    """

    return np.vectorize(
        normal_p_threshold, excluded=['threshold'])(
            z_scores, threshold=threshold)


def partial_correlation(X1, X2, *controls):
    """
    wrapper for partial_corr, written by Fabian Pedregosa
    Arguments
    -----
    X1, X2:  arrays whose partial correlation (conditional on some other arrays) we wish to compute)
    *controls:  any number of other control arrays
    """
    return partial_corr(np.vstack((X1, ) + (X2, ) + (*controls, )).T)


class EmpiricalCausalAdjacencySolver:
    """
    This class is the "inverse" of the artificial data generator--it takes in data, and generates a causal graph (according to the various assumptions described above)
    Attributes
    ---------
    T:  Number of data vectors (i.e. "time series")
    X:  "start" data (with shape (T, n_nodes))
    Y:  "end" data (with shape (T, n_nodes))
    empirical unary_adjaciences: an empirically generated  adjacency matrix connecting time steps
    empirical_binary_adjacencies:  an empirically generated dictionary, mapping a length-two tuple (indicating a pair of causes) to one or zero (indicating adjacency or not)
    """

    def __init__(self):
        self.n_nodes = None
        self.T = None
        self.X = None
        self.Y = None
        self.empirical_unary_adjacencies = None
        self.empirical_binary_adjacencies = None

    def solve(self, X, Y):
        """
        a stateful wrapper for generate_empirical_adjacencies
        Arguments
        -------
        X:  "start" data
        Y:  "final" data
        """
        assert X.shape == Y.shape
        self.T = X.shape[0]
        self.n_nodes = X.shape[1]
        self.X = X
        self.Y = Y
        self.empirical_unary_adjacencies, self.empirical_binary_adjacencies = self.generate_empirical_adjacencies(
            X, Y)

    def generate_empirical_adjacencies(self, X, Y):
        """
        generates empirical adjacencies from start and final data, according to the assumptions detailed above.  This function likely deserves more testing.

        Arguments
        -------
        X:  "start" data
        Y:  "final" data
        """
        assert X.shape == Y.shape
        n_nodes = X.shape[1]
        T = X.shape[0]
        unary_z_scores = np.zeros((n_nodes, n_nodes))
        empirical_unary_adjacencies = np.ones((n_nodes, n_nodes))
        ##  This section is the "meat" of the algorithm.  The algorithm seeks to first eliminate all non-significant connections (where nothing is controlled for) between causal node inode and effect node jnode.  Then it will control for one other causal node, two nodes, etc., to check whether any causation can be explained indirectly through another causal node.  If it can, the adjacency is eliminated.  This may be overly conservative.  
        for n_controls in range(n_nodes):
            for inode in range(n_nodes):
               for jnode in range(n_nodes):
                   if empirical_unary_adjacencies[inode, jnode] == 1:
                       possible_controls = [x for x in range(n_nodes) if x!= inode]
                       controlled_z_scores = []
                       for control_set in itertools.combinations(possible_controls, n_controls):
                           controls = tuple(X[:,cnode] for cnode in control_set)
                           rho = partial_correlation(X[:, jnode], Y[:, inode], *controls)[0, 1]
                           controlled_z_scores.append(fisher_z(rho, len(controls), T))
                       unary_z_scores[inode, jnode] = np.min(np.abs(controlled_z_scores))  #Picks the least-significant z-score
            empirical_unary_adjacencies = z_score_to_adjacencies(unary_z_scores)
        binary_z_scores = {(x1, x2): np.ones(n_nodes)
                           for x1 in range(n_nodes)
                           for x2 in range((x1 + 1), n_nodes)}
        for binary_causal_pair in binary_z_scores.keys():
            c1 = binary_causal_pair[0]
            c2 = binary_causal_pair[1]

            for ieffect in range(n_nodes):
                if empirical_unary_adjacencies[ieffect,
                                               c1] == 1 and empirical_unary_adjacencies[ieffect,
                                                                                        c2] == 1:
                    rho = np.corrcoef(X[:, c1] * X[:, c2], Y[:, ieffect])[0, 1]
                    binary_z_scores[binary_causal_pair][ieffect] = fisher_z(
                        rho, 0, T)
        empirical_binary_adjacencies = {
            key: z_score_to_adjacencies(binary_z_scores[key])
            for key in binary_z_scores.keys()
        }
        for x1 in range(n_nodes):
            empirical_binary_adjacencies[(x1, x1)] = np.zeros(n_nodes)
            for x2 in range((x1 + 1), n_nodes):
                empirical_binary_adjacencies[(
                    x2, x1)] = empirical_binary_adjacencies[(x1, x2)]
        return empirical_unary_adjacencies, empirical_binary_adjacencies


if __name__ == '__main__':
    ### A basic test
    n_nodes = 4
    T = 10000
    art = Artificial()
    art.generate(n_nodes, T)
    empir = EmpiricalCausalAdjacencySolver()
    empir.solve(art.X, art.Y)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0] = plot_adjacencies(
        ax[0],
        art.unary_adjacencies,
        binary_adjacencies=art.binary_adjacencies,
        plot_title='True Causal Graph')
    ax[0] = plot_adjacencies(
        ax[1],
        empir.empirical_unary_adjacencies,
        binary_adjacencies=empir.empirical_binary_adjacencies,
        plot_title='Empirical Causal Graph')
    plt.show()
