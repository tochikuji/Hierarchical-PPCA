"""
An example of Hierarchical Probabilistic Principal Component Analysis (HPPCA),
proposed in [1].

[1] Aiga Suzuki, Hayaru Shouno, "Generative Model of Textures Using Hierarchical Probabilistic Principal Component Analysis",
    Proc. of the 2017 Intl. Conference on Parallel and Distributed Processing Techniques and Applications (PDPTA’17),
    CSREA Press, pp.333-338, USA, Jul. 2017.
"""

import numpy
from sklearn.datasets import make_classification

from hpca import HPPCA
import hpca.reduction_rule as rrule

def main():
    # declare the model which has 3 intermediate PCA
    hppca = HPPCA(3)

    # set an group mask to the model
    # the mask means the first 3 components belong to group-1, following 2 components belong to group-2,
    # and the last 5 components belong to group-3
    mask = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    hppca.set_blockmask(mask)

    # set names to each group w.r.t. the indecies
    hppca.set_blockname(['first_three', 'next_two', 'others'])

    # set reduction rules for each PCA
    # decide dimensionality with the fixed contribution rate 0.9 for all intermediate PCAs
    hppca.set_intermediate_reducer([rrule.ContributionRateReduction(ccr=0.9)] * 3)

    # gate PCA outputs the fixed dimensionality = 2
    hppca.set_gate_reducer(rrule.FixedDimensionalityReduction(dim=2))

    # generate data
    N = 3000
    class1 = numpy.random.multivariate_normal(numpy.zeros(3), numpy.diag([1, 2, 3]), N)
    class2 = numpy.random.multivariate_normal(numpy.zeros(2), numpy.diag([0.2, 3]), N)
    class3 = numpy.random.multivariate_normal(numpy.zeros(5), numpy.diag([0.3, 4, 3, 2, 5]), N)

    X = numpy.hstack([class1, class2, class3])

    # run analysis
    hppca.fit(X)

    print(hppca.intermediates_[0].explained_variance_ratio_)
    print(hppca.intermediates_[1].explained_variance_ratio_)
    print(hppca.intermediates_[2].explained_variance_ratio_)

    print(hppca.gate_.explained_variance_ratio_)


if __name__ == '__main__':
    main()
