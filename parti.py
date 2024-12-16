import numpy as np
import pandas as pd
import tqdm
import logging
from sklearn.decomposition import PCA

try:
    from py_pcha import PCHA
except:
    PCHA = None
from scipy.spatial import ConvexHull
from unmixing import SISAL
from stat_utils import fdr_bh
from scipy.stats import hypergeom, mannwhitneyu, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances


ALG_PCHA = "ALG_PCHA"
ALG_SISAL_PY = "ALG_SISAL_PY"
ALG_PCA = "ALG_PCA"
MIN_VOL = "MIN_VOLUME"
MAX_VOL = "MAX_VOL"
VAR_EXP_IND = 4
LOW_REP_COUNT = 5
HIGH_REP_COUNT = 20
RERUN_THRESHOLD = 10


def sisal_py(data, n_archs, rerun=0):
    n = data.shape[0]
    data_original = data
    data = np.concatenate([data, np.ones((n, 1))], axis=1).T
    # data = data + np.random.randn(*data.shape) * 0.1
    res = SISAL(data, n_archs)[0]
    if res is None:
        logging.info("invalid result - rerunning")
        if rerun < RERUN_THRESHOLD:
            return sisal_py(data_original, n_archs, rerun=rerun + 1)
        else:
            raise ValueError("Python SISAL calculation failed")
    return res[:-1, :].T


def get_simplex_alg(alg_name):
    if alg_name == ALG_PCHA:
        def simplex_finder(data, n_archetypes):
            return np.array(PCHA(data.T, n_archetypes)[0].T)
    elif alg_name == ALG_SISAL_PY:
        simplex_finder = sisal_py
    else:
        raise NotImplementedError("Invalid algorithm type  - {}".format(alg_name))
    return simplex_finder


def repeated_simplex_finder(simplex_finder, n_repeats, criteria=MIN_VOL):
    def repeated_finder(data, n_archetypes):
        best_archs = None
        min_cost = np.inf
        for _ in range(n_repeats):
            try:
                archs = simplex_finder(data, n_archetypes)
            except:
                continue
            vol = calculate_volume(archs)
            if criteria == MAX_VOL:
                cost = - vol
            elif criteria == MIN_VOL:
                cost = vol
            else:
                raise NotImplementedError("Invalid simplex criteria: {}".format(criteria))
            if cost < min_cost:
                min_cost = cost
                best_archs = archs
        return best_archs

    return repeated_finder


def elbow_chooser(vars: np.ndarray):
    orig = np.array([0, vars[0]], dtype=float)
    end = np.array([vars.size - 1, vars[-1]], dtype=float)
    slope_vec = end - orig
    slope_vec = slope_vec / np.sqrt(np.inner(slope_vec, slope_vec))
    distances = []

    for i in range(vars.size):
        p = np.array([i, vars[i]], dtype=float)
        proj = slope_vec * np.inner(slope_vec, p - orig)
        perp = (p - orig) - proj
        distances.append(np.linalg.norm(perp))
    return np.argmax(distances) + 1


def calculate_volume(points):
    chull = ConvexHull(points)
    return chull.volume


def compute_t_ratio(data, simplex):
    chull_vol = calculate_volume(data)
    simplex_vol = calculate_volume(simplex)
    return simplex_vol / chull_vol


class ArchetypeFinder(object):
    def __init__(self,
                 n_archetypes: int,
                 alg: str,
                 n_perm: int = 20,
                 n_boot: int = 20,
                 max_boot_samples: int = -1):

        self.n_boot = n_boot
        self.max_boot_samples = max_boot_samples if max_boot_samples > 0 else np.inf
        self.alg = repeated_simplex_finder(get_simplex_alg(alg),
                                           LOW_REP_COUNT if alg == ALG_PCHA else HIGH_REP_COUNT,
                                           MAX_VOL if alg == ALG_PCHA else MIN_VOL)
        self.t_ratio_maximizing = alg == ALG_PCHA
        self.n_archetypes = n_archetypes
        self.n_perm = n_perm
        self.archetypes_ = None
        self.err_covs_ = None
        self.err_means_ = None
        self.p_val_ = None
        self.best_sample_indices_ = None
        self.null_t_ratio_distribution_ = None
        self.real_t_ratio_ = None

    def _check_null(self, datapoints):
        n, d = datapoints.shape
        alg = self.alg
        n_archetypes = self.n_archetypes

        def compute_t(datapoints, seed):
            if seed is not None:
                np.random.seed(seed)
            data_permuted = np.empty((n, d))
            for feat_ind in range(d):
                data_permuted[:, feat_ind] = datapoints[np.random.permutation(n), feat_ind]
            return compute_t_ratio(data_permuted, alg(data_permuted, n_archetypes))

        tratios = []
        for _ in tqdm.trange(self.n_perm, desc="Running t-ratios permutation test for simplex volume"):
            tratios.append(compute_t(datapoints, None))
        if self.n_perm == 0:
            tratios = [-np.inf]  # make sure that the default result is p-val == 1
        return np.array(tratios)

    def _compute_errors(self, datapoints: np.ndarray):
        n, d = datapoints.shape
        n_samp = int(np.minimum(n, self.max_boot_samples))
        all_archs = np.empty((self.n_boot, self.n_archetypes, d), dtype=float)
        all_samp_indices = np.empty((self.n_boot, n_samp), int)
        orig_archetypes = self.archetypes_
        n_archetypes = self.n_archetypes

        def get_archs(datapoints, seed):
            if seed is not None:
                np.random.seed(seed)
            samp_indices = np.random.choice(np.arange(n), (n_samp,))
            data_perm = datapoints[samp_indices]
            curr_archs = self.alg(data_perm, n_archetypes)
            cost_mat = euclidean_distances(orig_archetypes, curr_archs) ** 2
            _, ind_col = linear_sum_assignment(cost_mat)
            return curr_archs[ind_col], samp_indices

        for i_boot in tqdm.trange(self.n_boot, desc="Running bootstrap for archetype error estimation"):
            curr_archs, samp_indices = get_archs(datapoints, None)
            all_archs[i_boot] = curr_archs
            all_samp_indices[i_boot, :] = samp_indices

        self.err_means_ = all_archs.mean(axis=0)
        self.err_covs_ = np.zeros((self.n_archetypes, d, d))
        self.all_boot_archs_ = all_archs

        for i in range(self.n_archetypes):
            archs_i = all_archs[:, i, :]
            errs_i = archs_i - self.err_means_[[i]]
            cov_i = (errs_i.T @ errs_i) / self.n_boot
            self.err_covs_[i] = cov_i

        costs = np.sum((all_archs - self.err_means_[None]) ** 2, axis=(1, 2))
        best_ind = np.argmin(costs)
        return all_samp_indices[best_ind], self.all_boot_archs_[best_ind]

    def fit(self, datapoints: np.ndarray):
        data_low_dim = datapoints[:, :self.n_archetypes - 1]
        self.archetypes_ = self.alg(data_low_dim, self.n_archetypes)  # [n_archetypes, data_dim]

        best_samp, best_archs = self._compute_errors(data_low_dim)
        self.best_sample_indices_ = best_samp

        if (self.max_boot_samples == -1) or (self.max_boot_samples > datapoints.shape[0]):
            # Run standard t-ratio test
            data_for_t = data_low_dim
        else:
            # Run t-ratio test only on the best bootstrap sample in terms of distances from the mean archs distribution
            data_for_t = data_low_dim[best_samp]

        real_t_ratio = compute_t_ratio(data_for_t, self.archetypes_)
        null_dist = self._check_null(data_for_t)
        # check if the real t-ratio is closer to 1 than expected by chance
        self.p_val_ = np.mean(real_t_ratio < null_dist) if self.t_ratio_maximizing else np.mean(real_t_ratio > null_dist)
        self.real_t_ratio_ = real_t_ratio
        self.null_t_ratio_distribution_ = null_dist


class EnrichmentCalculator(object):
    def __init__(self,
                 bin_size: float = None,
                 eval_p_max: bool = False,
                 fdr_threshold: float = 0.1):
        self.bin_size = bin_size
        self.eval_p_max = eval_p_max
        self.fdr_threshold = fdr_threshold
        self.disc_p_vals_ = None
        self.disc_rejections_ = None
        self.cont_p_vals_ = None
        self.cont_rejections_ = None
        self.distance_to_archetypes_ = None
        self.datapoints_ = None
        self.archetypes_ = None

    def _default_bin_size(self, n_points: int):
        default_n = round(n_points / 10)
        default_n = max(default_n, 2)
        bin_size = 1 / default_n
        return bin_size

    def fit(self,
            datapoints: np.ndarray,
            archetypes: np.ndarray,
            discrete_enrichments: np.ndarray,
            continuous_enrichments: np.ndarray):
        self.datapoints_ = datapoints
        self.archetypes_ = archetypes
        n_disc_features = discrete_enrichments.shape[1]
        pts2archs_dists = euclidean_distances(datapoints, archetypes)  # [n_points, n_archs]
        self.distance_to_archetypes_ = pts2archs_dists
        n_archetypes = archetypes.shape[0]
        self.disc_p_vals_ = np.zeros((n_archetypes, n_disc_features))
        self.disc_corr_ = np.zeros((n_archetypes, n_disc_features))
        self.disc_corr_p_vals_ = np.zeros((n_archetypes, n_disc_features))
        for feat_index, feat_enrichment in enumerate(discrete_enrichments.T):
            valid = np.logical_not(np.isnan(feat_enrichment))
            n_points = int(np.sum(valid))
            bin_size = self.bin_size or self._default_bin_size(n_points=n_points)
            n_bins = int(round(1 / bin_size))
            points_per_bin = np.round(n_points * bin_size).astype(int)
            valid_dists = pts2archs_dists[valid]
            ind_sorted = np.argsort(valid_dists, axis=0)
            valid_feats = feat_enrichment[valid]
            valid_feats_bool = valid_feats.astype(bool)
            for arch_index in range(n_archetypes):
                feats_sorted = valid_feats[ind_sorted[:, arch_index]]
                hyg = hypergeom(n_points, np.sum(feats_sorted), points_per_bin)
                p_val = 1 - hyg.cdf(np.sum(feats_sorted[:points_per_bin]) - 1)
                self.disc_p_vals_[arch_index, feat_index] = p_val
                u, p = mannwhitneyu(valid_dists[valid_feats_bool, arch_index],
                                    valid_dists[~valid_feats_bool, arch_index])
                n1, n2 = np.sum(valid_feats_bool), np.sum(~valid_feats_bool)
                corr = (2 * u) / (n1 * n2) - 1  # rank biserial correlation
                self.disc_corr_[arch_index, feat_index] = corr
                self.disc_corr_p_vals_[arch_index, feat_index] = p
        if n_disc_features > 0:
            self.disc_rejections_ = fdr_bh(self.disc_p_vals_.flatten(),
                                           self.fdr_threshold).reshape(self.disc_p_vals_.shape)
        else:
            self.disc_rejections_ = np.zeros_like(self.disc_p_vals_)

        n_cont_features = continuous_enrichments.shape[1]
        self.cont_p_vals_ = np.ones((n_archetypes, n_cont_features))
        self.cont_median_diffs_ = np.zeros((n_archetypes, n_cont_features))
        self.cont_mean_diffs_ = np.zeros((n_archetypes, n_cont_features))
        self.cont_corr_rho_ = np.zeros((n_archetypes, n_cont_features))
        self.cont_corr_p_vals_ = np.zeros((n_archetypes, n_cont_features))
        for feat_index, feat_enrichment in enumerate(continuous_enrichments.T):
            valid = np.logical_not(np.isnan(feat_enrichment))
            feat_enrichment = feat_enrichment[valid]
            n_points = int(np.sum(valid))
            bin_size = self.bin_size or self._default_bin_size(n_points=n_points)
            valid_dists = pts2archs_dists[valid]
            ind_sorted = np.argsort(valid_dists, axis=0)
            points_per_bin = np.round(n_points * bin_size).astype(int)
            for arch_index in range(n_archetypes):
                if np.all(feat_enrichment == np.mean(feat_enrichment)):
                    continue  # mann-whitney will fail in the case that all values are the same
                first_bin = feat_enrichment[ind_sorted[:, arch_index]][:points_per_bin]
                rest = feat_enrichment[ind_sorted[:, arch_index]][points_per_bin:]
                _, p_val = mannwhitneyu(first_bin, rest)
                self.cont_p_vals_[arch_index, feat_index] = p_val
                self.cont_median_diffs_[arch_index, feat_index] = np.median(first_bin) - np.median(rest)
                self.cont_mean_diffs_[arch_index, feat_index] = np.mean(first_bin) - np.mean(rest)
                rho, corr_pval = spearmanr(valid_dists[:, arch_index], feat_enrichment)
                self.cont_corr_rho_[arch_index, feat_index] = rho
                self.cont_corr_p_vals_[arch_index, feat_index] = corr_pval
        if n_cont_features > 0:
            self.cont_rejections_ = fdr_bh(self.cont_p_vals_.flatten(),
                                           self.fdr_threshold).reshape(self.cont_p_vals_.shape)
        else:
            self.cont_rejections_ = np.zeros_like(self.cont_p_vals_)


class ParTI(object):
    def __init__(self,
                 max_dim: int = 20,
                 simplex_alg: str = ALG_PCHA,
                 n_archetypes: int = None,
                 bin_size: float = None,
                 fdr_threshold: float = 0.1,
                 verbose: bool = True,
                 n_permutations: int = 1000,
                 choose_dim_by: str = ALG_PCA,
                 bind_archetypes: str = None,
                 n_bootstraps: int = 1000,
                 max_boot_samples: int = -1,
                 project_by: str = None):
        self.max_boot_samples = max_boot_samples
        self.project_by = project_by
        self.n_bootstraps = n_bootstraps
        self.max_dim = max_dim
        self.simplex_alg = simplex_alg
        self.n_archetypes = n_archetypes
        self.bin_size = bin_size
        self.fdr_threshold = fdr_threshold
        self.verbose = verbose
        self.is_fitted = False
        self.n_permutations = n_permutations
        self.bind_archetypes = bind_archetypes
        self.enrichment_calculator = EnrichmentCalculator(bin_size=bin_size,
                                                          eval_p_max=False,
                                                          fdr_threshold=fdr_threshold)
        self.choose_dim_by = choose_dim_by
        # estimated attributes
        self.chosen_dim_ = None
        self.explained_vars_ = None
        self.archetypes_ = None
        self.pca_components_ = None
        self.archetypes_cov_ = None
        self.archetypes_p_val_ = None
        self.discrete_enrichment_p_vals_ = None
        self.discrete_enrichment_rejections_ = None
        self.continuous_enrichment_p_vals_ = None
        self.continuous_enrichment_rejections_ = None
        self.continuous_enrichment_median_diffs_ = None
        self.continuous_enrichment_mean_diffs_ = None

    def choose_dim(self, data_points: np.ndarray):
        pca = PCA(data_points.shape[1])
        data_pca = pca.fit_transform(data_points)

        self.explained_vars_ = np.cumsum(pca.explained_variance_ratio_)[:self.max_dim]
        dims = list(range(1, self.max_dim + 1))
        if PCHA is not None:
            msg = "calculating PCHA explained variance for elbow method"
            def safe_repeated_call(data, noc):
                for _ in range(10):
                    try:
                        res = PCHA(data.T, noc=noc)
                        return res
                    except Exception as e:
                        print(e)
                        continue
                raise ValueError("PCHA failed to converge")

            self.pcha_explained_vars_ = np.array(
                [safe_repeated_call(data_pca.T, noc=dim + 1)[VAR_EXP_IND] for dim in tqdm.tqdm(dims, desc=msg)])
        else:
            self.pcha_explained_vars_ = np.zeros((0,))
        if self.n_archetypes is not None:
            dim = self.n_archetypes - 1
        else:
            if self.choose_dim_by == ALG_PCA:
                dim = elbow_chooser(self.explained_vars_)
                self.n_archetypes = dim + 1
            elif self.choose_dim_by == ALG_PCHA:
                dim = elbow_chooser(self.pcha_explained_vars_)
                self.n_archetypes = dim + 1
            else:
                raise ValueError("Invalid algorithm for choosing dimension: '{}'".format(self.choose_dim_by))
        return dim


    def _project_data(self, datapoints: np.ndarray):
        self.chosen_dim_ = self.choose_dim(datapoints)
        pca_input_data = datapoints
        if self.project_by is not None:
            pca_input_data = pd.read_csv(self.project_by)
            dim_diff = pca_input_data.shape[1] - datapoints.shape[1]
            if dim_diff == 1:
                pca_input_data = pca_input_data.iloc[:, 1:].values
            elif dim_diff == 0:
                pca_input_data = pca_input_data.values
            else:
                raise ValueError(
                    "project_by input data has different number of dimensions compared to ParTI features input")

        self.pca = PCA(n_components=self.chosen_dim_)
        self.pca.fit(pca_input_data)
        self.data_pca_ = self.pca.transform(datapoints)
        self.pca_components_ = self.pca.components_

    def _find_archetypes(self):
        logging.info("Running archetype analysis for chosen dimension - {}".format(self.chosen_dim_))
        self.archetype_finder = ArchetypeFinder(n_archetypes=self.chosen_dim_ + 1,
                                                alg=self.simplex_alg,
                                                n_perm=self.n_permutations,
                                                n_boot=self.n_bootstraps,
                                                max_boot_samples=self.max_boot_samples)
        self.archetype_finder.fit(self.data_pca_)
        if self.bind_archetypes is not None:
            bind_archetypes = pd.read_csv(self.bind_archetypes)
            arch_cols = [col for col in bind_archetypes.columns if col.startswith("arch")]
            bind_archetypes = bind_archetypes[arch_cols].values.T

            dists = euclidean_distances(bind_archetypes, self.pca.inverse_transform(self.archetype_finder.archetypes_))
            bind_indices, matched_indices = linear_sum_assignment(dists)
            remaining_indices = [ind for ind in range(self.chosen_dim_ + 1) if ind not in matched_indices]
            arch_indices = np.concatenate([matched_indices, remaining_indices]).astype(int)
        else:
            arch_indices = np.arange(self.chosen_dim_ + 1, dtype=int)
        self.archetypes_ = self.archetype_finder.archetypes_[arch_indices]
        self.archetypes_cov_ = self.archetype_finder.err_covs_[arch_indices]
        self.archetypes_boot_mean_ = self.archetype_finder.err_means_[arch_indices]
        self.all_boot_archs_ = self.archetype_finder.all_boot_archs_[:, arch_indices]

        self.archetypes_original_ = self.pca.inverse_transform(self.archetypes_)
        self.archetypes_boot_mean_original_ = self.pca.inverse_transform(self.archetypes_boot_mean_)
        self.all_boot_archs_orig_ = self.pca.inverse_transform(
            self.all_boot_archs_.reshape(self.n_bootstraps * self.n_archetypes, -1)).reshape(
            (self.n_bootstraps, self.n_archetypes, -1))
        self.archetypes_p_val_ = self.archetype_finder.p_val_
        self.best_sample_indices_ = self.archetype_finder.best_sample_indices_
        self.real_t_ratio_ = self.archetype_finder.real_t_ratio_
        self.t_ratio_null_dist_ = self.archetype_finder.null_t_ratio_distribution_
        return

    def _calculate_enrichment(self, datapoints, archs, discrete_enrichment, continuous_enrichment):
        self.enrichment_calculator.fit(datapoints=datapoints,
                                       archetypes=archs,
                                       discrete_enrichments=discrete_enrichment,
                                       continuous_enrichments=continuous_enrichment)
        self.continuous_enrichment_p_vals_ = self.enrichment_calculator.cont_p_vals_
        self.continuous_enrichment_rejections_ = self.enrichment_calculator.cont_rejections_
        self.discrete_enrichment_p_vals_ = self.enrichment_calculator.disc_p_vals_
        self.discrete_enrichment_rejections_ = self.enrichment_calculator.disc_rejections_
        self.continuous_enrichment_median_diffs_ = self.enrichment_calculator.cont_median_diffs_
        self.continuous_enrichment_mean_diffs_ = self.enrichment_calculator.cont_mean_diffs_
        return

    def fit(self, datapoints: np.ndarray, continuous_enrichments: np.ndarray, discrete_enrichments: np.ndarray):
        self._project_data(datapoints)
        self._find_archetypes()
        self._calculate_enrichment(self.data_pca_[:, :self.chosen_dim_], self.archetypes_, discrete_enrichments,
                                   continuous_enrichments)
        return

    def compute_distances_to_archetype(self, datapoints: np.ndarray, archetype: int):
        if datapoints is None:
            data_low_dim = self.data_pca_
        else:
            data_low_dim = self.pca.transform(datapoints)
        arch_low_dim = self.archetypes_[[archetype]]
        dists = np.sum((data_low_dim - arch_low_dim) ** 2, axis=-1)  # Sum over features to obtain square distances
        return dists

    def save_evaluation(self, path):
        fitted_params = {k: v for k, v in self.__dict__.items() if k.endswith("_") and (v is not None)}
        np.savez(path, **fitted_params)
