import numpy
import sklearn.decomposition as dec
from typing import Iterable, Union, List, Optional
from hpca.reduction_rule import ReductionRule, ContributionRateReduction, FixedDimensionalityReduction


class HPPCA:
    def __init__(self, n_blocks: int, **kargs):
        self.intermediates_ = [dec.PCA(**kargs) for _ in range(n_blocks)]
        self.gate_ = dec.PCA(**kargs)

        self.n_blocks_ = n_blocks

        self.block_masks_ = None
        self.block_names_ = [str(id) for id in range(self.n_blocks_)]

        self.intermediate_reducers_ = [ContributionRateReduction()] * n_blocks
        self.gate_reducer_ = ContributionRateReduction()

    def set_blockmask(self, mask: Iterable):
        """

        Args:
            mask:

        Returns:

        """

        block_ids = sorted(set(mask))

        if len(block_ids) != self.n_blocks_:
            raise IndexError(f'size mismatch between n_block: {self.n_blocks_} and '
                             f'a number of specified blockmask: n={len(block_ids)}')

        # verify whether the mask is correct sequence of indecies
        for i, id in enumerate(block_ids):
            if i != id:
                raise ValueError(f'mask must be a sequence of indeies, i.e., [1,2,...,N]')

        self.block_masks_ = numpy.asarray(mask, dtype=int)

    def set_blockname(self, names: Iterable):
        """

        Args:
            names:

        Returns:

        """

        if len(names) != self.n_blocks_:
            raise TypeError(f'length mismatch [self.n_blocks_: {self.n_blocks_}, names(given): {len(names)}]')

        self.block_names_ = names

    def set_intermediate_reducer(self, reducers: Union[ReductionRule, List[ReductionRule]],
                                 id: Optional[int] = None):
        """

        Args:
            reducers:
            id:

        Returns:

        """

        if isinstance(reducers, list):
            if len(reducers) != len(self.intermediate_reducers_):
                raise IndexError(f'size mismatch between specified reducer: {len(reducers)} '
                                 f'and n_blocks: {self.n_blocks_}')
            self.intermediate_reducers_ = reducers

        if isinstance(reducers, ReductionRule):
            if id is None:
                raise IndexError('block id must be specified to set to single block')

            self.intermediate_reducers_[id] = reducers

    def set_gate_reducer(self, reducer: ReductionRule):
        self.gate_reducer_ = reducer

    @property
    def blocks_(self):
        return self.block_masks_

    def __decompose_blocks(self, X: numpy.ndarray) -> List[numpy.ndarray]:
        if self.blocks_ is None:
            raise LookupError('block mask is not set. Call `set_blockmask` in advance to set masks.')

        block_ids = sorted(set(self.block_masks_))

        blocks = []

        for id in block_ids:
            # pick dimensions masked components with the id
            block = X.T[numpy.where(self.block_masks_ == id)].T
            blocks.append(block)

        return blocks

    def __zerofill_reduced_repr(self, Y: numpy.ndarray, pca: dec.PCA):
        N, dim_reduced = Y.shape
        raw_dim = pca.n_components_

        zeros_to_fill = numpy.zeros((N, raw_dim - dim_reduced))

        return numpy.hstack((Y, zeros_to_fill))

    def list2dic(self, X):
        """
        convert listed feature into named hash

        Args:
            X (Iterable): feature in input space

        Returns:
            dict or list of dictionary
        """
        return {name: block for name, block in zip(self.__decompose_blocks(X), self.block_names_)}

    def dic2list(self, X):
        """
        convert feature dictionary into flattened list

        Args:
            X (dict): feature in input space

        Returns:
            numpy.array
        """
        pass

    def fit_intermediates(self, X):
        decomposed_X = self.__decompose_blocks(X)

        for block_X, block_pca in zip(decomposed_X, self.intermediates_):
            block_pca.fit(block_X)

    def fit(self, X):
        self.fit_intermediates(X)
        intermediate_repr = self.intermediate_transform(X)
        self.gate_.fit(intermediate_repr)

    def reconstruct(self, X):
        reduced_repr = self.transform(X)
        return self.inverse_transform(reduced_repr)

    def reconstruct_intermediate(self, X):
        reduced_repr_intr = self.intermediate_transform(X)
        return self.intermediate_inverse_transform(reduced_repr_intr)

    def inverse_transform(self, Y):
        Y_filled = self.__zerofill_reduced_repr(Y, self.gate_)
        intermediate_reduced_repr = self.gate_.inverse_transform(Y_filled)

        Y_intr = self.__decompose_intermediate_repr(intermediate_reduced_repr)

        return self.intermediate_inverse_transform(Y_intr)

    def intermediate_inverse_transform(self, Y_intr):
        dims = [reduction.effective_dim(pca)
                for reduction, pca in zip(self.intermediate_reducers_, self.intermediates_)]

        buffer = []
        current_index = 0

        for d, pca in zip(dims, self.intermediates_):
            block_repr = Y_intr[current_index:current_index + d]
            block_repr_filled = self.__zerofill_reduced_repr(block_repr, pca)

            recon_X = pca.inverse_transform(block_repr_filled)
            buffer.append(recon_X)

        return numpy.hstack(buffer)

    def transform(self, X):
        intermediate_repr = self.intermediate_transform(X)
        return self.gate_.transform(intermediate_repr)

    def intermediate_transform(self, X: numpy.ndarray, vectify=True) -> Union[numpy.ndarray, List[numpy.ndarray]]:
        """
        Convert features into intermediate reduced representations

        Args:
            X (numpy.ndarray): 2D-array (N, d)
            vectify: if vectify is False, it returns a list of each block representations

        Returns:
            numpy.ndarray or list of numpy.ndarray
        """

        decomposed_X = self.__decompose_blocks(X)
        buffer = []

        for block_X, block_pca, block_reduct in zip(decomposed_X, self.intermediates_, self.intermediate_reducers_):
            effective_dim = block_reduct.effective_dim(block_pca)
            block_feature = block_pca.transform(block_X)[:, :effective_dim]
            buffer.append(block_feature)

        if vectify:
            return numpy.hstack(buffer)
        else:
            return buffer


if __name__ == '__main__':
    hpca = HPPCA(n_blocks=3)
    X = numpy.random.rand(1000, 10)
    blockmask = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    hpca.set_blockmask(blockmask)
    hpca.set_blockname([f'Block{i + 1}' for i in range(3)])
    hpca.fit(X)
    pass
