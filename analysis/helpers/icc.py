import numpy as np


class ReliabilityAnalysis:
    @staticmethod
    def compute_icc(data: np.ndarray) -> float:
        """
        Two-way mixed ICC(2,1) — absolute agreement.
        data: numpy array of shape (n_samples, k_runs)
        Returns the ICC coefficient.
        Reference: Koo and Li (2016).
        """
        n, k = data.shape

        grand_mean = data.mean()
        ms_rows = k * np.sum((data.mean(axis=1) - grand_mean) ** 2) / (n - 1)
        ss_err = np.sum(
            (
                data
                - data.mean(axis=1, keepdims=True)
                - data.mean(axis=0, keepdims=True)
                + grand_mean
            )
            ** 2
        )
        ms_err = ss_err / ((n - 1) * (k - 1))

        return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
