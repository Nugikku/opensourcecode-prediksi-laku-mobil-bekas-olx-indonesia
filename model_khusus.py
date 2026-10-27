"""
AutoLaku: model khusus yang perlu bisa di-pickle & dimuat ulang oleh mesin_prediksi.py / app.py.
(Class harus berada di modul terpisah, bukan di skrip training, agar pickle tidak merujuk ke __main__.)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer


class NaiveBayesDiskret(BaseEstimator, ClassifierMixin):
    """Naive Bayes untuk campuran fitur kategorik dan numerik (materi P9).

    - Kolom biner (hasil one-hot) dipakai apa adanya.
    - Kolom numerik didiskretkan ke n_bins (kuantil) lalu di-one-hot.
    - BernoulliNB dengan Laplace smoothing (alpha) menghitung prior & likelihood.
    Dipakai karena GaussianNB tidak cocok untuk fitur one-hot.
    """

    def __init__(self, n_bins=8, alpha=1.0):
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        biner = np.array([set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(X.shape[1])])
        self.kol_biner_ = np.where(biner)[0]
        self.kol_numerik_ = np.where(~biner)[0]
        self.kbins_ = None
        if len(self.kol_numerik_):
            self.kbins_ = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='quantile')
            self.kbins_.fit(X[:, self.kol_numerik_])
        self.nb_ = BernoulliNB(alpha=self.alpha).fit(self._ubah(X), y)
        self.classes_ = self.nb_.classes_
        return self

    def _ubah(self, X):
        X = np.asarray(X, dtype=float)
        bagian = [X[:, self.kol_biner_]]
        if self.kbins_ is not None:
            bagian.append(self.kbins_.transform(X[:, self.kol_numerik_]))
        return np.hstack(bagian)

    def predict(self, X):
        return self.nb_.predict(self._ubah(X))

    def predict_proba(self, X):
        return self.nb_.predict_proba(self._ubah(X))