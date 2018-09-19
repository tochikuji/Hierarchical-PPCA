import sklearn.decomposition as dec
from typing import Iterable, Union, List, Optional


class ReductionRule:
    def effective_components(self, pca: dec.PCA) -> slice:
        raise NotImplementedError()

    def effective_dim(self, pca: dec.PCA) -> int:
        """
        Calculate an effective dimensionality

        Args:
            pca: dec.PCA

        Returns:
            number of dimesion
        """

        dim_slice = self.effective_components(pca)
        return dim_slice.stop - dim_slice.start

    def effective_ccr(self, pca: dec.PCA) -> float:
        """
        Calculate a cumulative contribution rate of the effective dimensionality

        Args:
            pca: pca engine

        Returns:
            ccr <- [0, 1]
        """

        dim = self.effective_dim(pca)
        ccr = 0.

        for lamb in pca.explained_variance_ratio_[:dim]:
            ccr += lamb

        return ccr


class FixedDimensionalityReduction(ReductionRule):
    def __init__(self, dim: int):
        """
        Specify fixed dimensionality; None means no reduction.
        Args:
            dim(int): output dimensionality
        """
        self.dim = dim

    def effective_components(self, pca: dec.PCA):
        if pca.n_components_ < self.dim:
            raise ValueError(f'Output dimensionality must be smaller than n_components_, '
                             f'but dim={self.dim} > {pca.n_components_} has specified')

        return slice(0, self.dim)

    def calc_ccr(self, pca):
        return pca.explained_variance_ratio_[0:self.dim].sum()


class ContributionRateReduction(ReductionRule):
    def __init__(self, ccr=1.0):
        """
        Handle reduction with the CCR(Cumulative contribution rate);
        Args:
            ccr((0-1]): CCR of PCA, i.e., a sum of eigenvalues
        """
        self.ccr = ccr

    def effective_components(self, pca):
        if self.ccr <= 0. or self.ccr > 1.:
            raise ValueError(f'CCR must be in (0, 1] but the CCR = {self.ccr}')

        effective_dim = 0
        current_ccr = 0.

        while current_ccr < self.ccr:
            try:
                current_ccr += pca.explained_variance_ratio_[effective_dim]
                effective_dim += 1
            except IndexError:
                effective_dim = pca.n_components_
                break

        return slice(0, effective_dim)
