import cooler
import numpy as np


def distribution_at_dist(arr, d):
    n = arr.shape[0]
    return np.array([arr[i, j] for i, j in zip(range(0, n - d), range(d, n))])


def normalize_intra(arr):
    n = arr.shape[0]
    averages_at_dist = [np.nanmean(distribution_at_dist(arr, d)) for d in range(0, n)]
    ans = np.zeros_like(arr, dtype='float64')
    for i in range(n):
        for j in range(n):
            ans[i, j] = arr[i, j] / averages_at_dist[abs(i - j)]
    return ans


class CoolerExtended(cooler.Cooler):
    __threshold = 0.8

    def __init__(self, filepath):
        super().__init__(filepath)
        self.hic_matrices_normalized = {}
        for current_chr in self.chromnames:
            mat = self.matrix(balance=False).fetch(current_chr)
            mat_nan = self.__zeros_to_nan(mat)
            mat_norm = normalize_intra(mat_nan)
            self.hic_matrices_normalized[current_chr] = mat_norm

    def __zeros_to_nan(self, arr):
        arr = arr.astype(float)
        n = arr.shape[0]
        for i in range(len(arr)):
            if ((arr[i] == 0).sum(0) / n) >= self.__threshold:
                arr[i] = np.nan
                arr[:, i] = np.nan
        return arr


