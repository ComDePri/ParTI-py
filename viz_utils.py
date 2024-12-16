import os
from os.path import join as opj

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import rgb2hex

from parti import ParTI, EnrichmentCalculator
from stat_utils import confidence_ellipse_by_cov, confidence_ellipse_3d_by_cov
import seaborn as sns
from matplotlib.colors import hex2color
from sklearn.metrics.pairwise import euclidean_distances


def choose_scatter_size_and_alpha(x, y, ax):
    try:
        # compute the total range of values in the plot
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        # compute the typical closest point distance
        dists = euclidean_distances(np.array([x, y]).T)
        dists[np.arange(dists.shape[0]), np.arange(dists.shape[0])] = np.inf
        min_dist = np.min(dists, axis=0)
        # compute the typical distance between points
        typical_dist = np.median(min_dist)
        # choose radius to be 1/2 of typical distance in relative units
        radius = typical_dist / 2
        relative_size = (radius ** 2 / (x_range * y_range))
        relative_size = np.clip(relative_size, 0.0001, np.inf)
        rad_corrected = np.sqrt(relative_size * x_range * y_range)
        # expected overlap
        overlap = np.quantile(np.sum(dists < rad_corrected, axis=1), 0.95) + 1
        # compute alpha
        alpha = np.clip(1 / overlap, 0.05, 1)
        s = relative_size * ax.bbox.width * ax.bbox.height
        s = np.clip(s, 4, 100)
    except:
        s = 10
        alpha = 0.5
    return s, alpha


def get_palette(palette_name):
    if palette_name in ["", "default"]:
        palette = [plt.get_cmap('tab10')(i) for i in range(10)]
    else:
        try:
            palette = palette_name.split(",")
            palette = [hex2color(hexc) for hexc in palette]
            if len(palette) < 10:
                palette = palette + [plt.get_cmap('tab10')(i) for i in range(10 - len(palette))]
        except:
            raise ValueError(
                "Couldn't extract hex color values from palette string. Please provide a comma-separated list of hex colors")
    return tuple(palette)


def save_rotating_gif(out_path, fig, ax):
    def init():
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i * 2)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    anim.save(out_path, fps=15, dpi=200, writer='imagemagick')


def plot_archetypes_3d(parti_estimator: ParTI):
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(parti_estimator.data_pca_[:, 0], parti_estimator.data_pca_[:, 1], parti_estimator.data_pca_[:, 2], s=1.5,
               c='k')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    for i, arch in enumerate(parti_estimator.archetypes_):
        ax.scatter(arch[0], arch[1], arch[2], label="archetype #{}".format(i + 1))
        confidence_ellipse_3d_by_cov(parti_estimator.archetypes_cov_[i], arch, ax, n_std=1.0,
                                     facecolor=rgb2hex(PALETTE(i)))
    return ax


def plot_archetypes_2d(parti_estimator: ParTI, archetypes_palette=get_palette("default")):
    ax = plt.gca()
    ax.scatter(parti_estimator.data_pca_[:, 0], parti_estimator.data_pca_[:, 1], s=1.5, c='k', alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    for i, arch in enumerate(parti_estimator.archetypes_):
        ax.scatter(arch[0], arch[1], label="archetype #{}".format(i + 1), color=archetypes_palette[i])
        confidence_ellipse_by_cov(parti_estimator.archetypes_cov_[i], arch, ax, n_std=1.0,
                                  edgecolor=rgb2hex(archetypes_palette[i]))
    plt.legend()


def savefig_all(fig, out_path):
    fig.savefig(out_path.format("png"), dpi=400)
    fig.savefig(out_path.format("pdf"))
    fig.savefig(out_path.format("svg"))


def get_discrete_ticks(values, n_ticks=8):
    range_size = max(values) - min(values)
    step = int(np.ceil((range_size / n_ticks)))
    return np.arange(min(values), max(values) + step, step)[:n_ticks]


def plot_elbow_results(elbow_choices, dims, results_dir, run_desc):
    plt.figure()
    plt.plot(dims, elbow_choices, "ok")
    plt.xlabel("Maximal Dimension")
    plt.xticks(get_discrete_ticks(dims))
    plt.ylabel("Elbow-based Choice")
    savefig_all(plt.gcf(), opj(results_dir, "{}elbow_plot.{}").format(run_desc, "{}"))
    plt.close()


def save_explained_var_results(results_dir: str,
                               parti_estimator: ParTI,
                               run_desc=""):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.figure()
    esv = parti_estimator.explained_vars_
    esv = np.concatenate(([0], esv))
    #  plot explained var with line and markers
    plt.plot(np.arange(esv.size), esv, "-ok")
    plt.xticks(get_discrete_ticks(np.arange(esv.size), n_ticks=8))
    plt.xlabel("dimension")
    plt.ylabel("PCA explained variance")
    savefig_all(plt.gcf(), opj(results_dir, "{}pca_ev_plot.{}".format(run_desc, "{}")))
    plt.close()

    plt.figure()
    # plt.title("PCHA cumulative explained variance")
    esv = parti_estimator.pcha_explained_vars_
    esv = np.concatenate(([0], esv))
    plt.plot(np.arange(esv.size), esv, "-ok")
    plt.xlabel("dimension")
    plt.ylabel("PCHA explained variance")
    plt.xticks(get_discrete_ticks(np.arange(esv.size), n_ticks=8))
    savefig_all(plt.gcf(), opj(results_dir, "{}esv.{}".format(run_desc, "{}")))
    plt.close()


def save_results_viz(results_dir: str,
                     parti_estimator: ParTI,
                     disc_df,
                     cont_df,
                     archetypes_palette=tuple([plt.get_cmap('tab10')(i) for i in range(10)]),
                     run_desc=""):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_explained_var_results(results_dir=results_dir,
                               parti_estimator=parti_estimator,
                               run_desc=run_desc)

    save_enrichment_results_viz(results_dir=results_dir,
                                enrichment_estimator=parti_estimator.enrichment_calculator,
                                disc_df=disc_df,
                                cont_df=cont_df,
                                archetypes_palette=archetypes_palette,
                                run_desc=run_desc)

    #  Pareto front plots

    d = parti_estimator.archetypes_.shape[1]
    if d >= 3:
        try:
            fig = plt.figure()
            ax = plot_archetypes_3d(parti_estimator)
            savefig_all(fig, opj(results_dir, "{}archs_scatter3D.{}").format(run_desc, "{}"))
            plt.close(fig)
        except:
            print("Failed creating 3D scatter plot of the simplex")

    if d >= 2:
        fig = plt.figure()
        plot_archetypes_2d(parti_estimator, archetypes_palette=archetypes_palette)
        savefig_all(fig, opj(results_dir, "{}archs_scatter2D.{}").format(run_desc, "{}"))
        plt.close(fig)
    else:
        pass


def save_results_data(results_dir: str,
                      parti_estimator: ParTI,
                      disc_features_descs,
                      cont_features_descs,
                      data_features_descs,
                      samples_index,
                      run_desc=""):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    archs_pca = parti_estimator.archetypes_
    archs_original = np.dot(archs_pca, parti_estimator.pca.components_[:archs_pca.shape[1]]) + parti_estimator.pca.mean_
    df_dict = {"feature_name": data_features_descs}
    for i, arch in enumerate(archs_original):
        df_dict["arch_{}".format(i + 1)] = arch.tolist()
    df = pd.DataFrame(df_dict)
    df.to_csv(opj(results_dir, "{}archs_original.csv".format(run_desc)))
    df_dict = {"feature_name": ["PC-{}".format(i + 1) for i in range(archs_pca.shape[1])]}
    for i, arch in enumerate(archs_pca):
        df_dict["arch_{}".format(i + 1)] = arch.tolist()

    df = pd.DataFrame(df_dict)
    df.to_csv(opj(results_dir, "{}archs_pca.csv".format(run_desc)))

    save_enrichment_results_data(results_dir=results_dir,
                                 enrichment_estimator=parti_estimator.enrichment_calculator,
                                 disc_features_descs=disc_features_descs,
                                 cont_features_descs=cont_features_descs,
                                 samples_index=samples_index,
                                 run_desc=run_desc)


def save_enrichment_results_data(results_dir: str,
                                 enrichment_estimator: EnrichmentCalculator,
                                 disc_features_descs,
                                 cont_features_descs,
                                 samples_index=None,
                                 run_desc=""):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    col_keys = ["feature_name", "archetype", "p-value", "is_significant", "median_difference", "mean_difference",
                "Spearman_rho"]
    n_archs = enrichment_estimator.archetypes_.shape[0]
    n_features = len(cont_features_descs)
    archs_desc = []

    for i in range(1, n_archs + 1):
        archs_desc.extend([i] * n_features)

    data_rows = list(zip(cont_features_descs * n_archs,
                         archs_desc,
                         enrichment_estimator.cont_p_vals_.flatten().tolist(),
                         enrichment_estimator.cont_rejections_.flatten().tolist(),
                         enrichment_estimator.cont_median_diffs_.flatten().tolist(),
                         enrichment_estimator.cont_mean_diffs_.flatten().tolist(),
                         enrichment_estimator.cont_corr_rho_.flatten().tolist()))
    df = pd.DataFrame(data_rows, columns=col_keys)
    df.to_csv(opj(results_dir, "{}continuous_enrichment_results.csv".format(run_desc)), index=False)

    col_keys = ["feature_name", "archetype", "p-value", "is_significant", "rank_biserial_correlation"]
    n_features = len(disc_features_descs)
    archs_desc = []
    for i in range(1, n_archs + 1):
        archs_desc.extend([i] * n_features)
    data_rows = list(zip(disc_features_descs * n_archs,
                         archs_desc,
                         enrichment_estimator.disc_p_vals_.flatten(),
                         enrichment_estimator.disc_rejections_.flatten(),
                         enrichment_estimator.disc_corr_.flatten()))
    df = pd.DataFrame(data_rows, columns=col_keys)
    df.to_csv(opj(results_dir, "{}discrete_enrichment_results.csv".format(run_desc)), index=False)

    dists = enrichment_estimator.distance_to_archetypes_
    dists_indices = np.argsort(np.argsort(dists, axis=0), axis=0)
    df_dists = pd.DataFrame(dists, columns=[f"arch{i}_distance" for i in range(1, dists.shape[1] + 1)],
                            index=samples_index)
    df_dists.to_csv(opj(results_dir, "{}arch_dists.csv".format(run_desc)), index=True)
    df_dist_order = pd.DataFrame(dists_indices,
                                 columns=[f"arch{i}_distance_order" for i in range(1, dists.shape[1] + 1)],
                                 index=samples_index)
    df_dist_order.to_csv(opj(results_dir, "{}arch_dists_order.csv".format(run_desc)), index=True)


def save_enrichment_results_viz(results_dir,
                                enrichment_estimator: EnrichmentCalculator,
                                disc_df,
                                cont_df,
                                run_desc,
                                archetypes_palette=tuple([plt.get_cmap('tab10')(i) for i in range(10)]),
                                add_legend=True,
                                archetype_names=None):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if archetype_names is None:
        archetype_names = [f"arch{i}" for i in range(1, enrichment_estimator.archetypes_.shape[0] + 1)]
    sns.set()

    # Enrichment plots
    binned_disc_enrichment = {dfd: {} for dfd in disc_df.columns}
    binned_cont_enrichment = {cfd: {} for cfd in cont_df.columns}

    for d_feat, d_feat_name in zip(disc_df.values.T,
                                   disc_df.columns):
        d_feat_name = d_feat_name.replace("/", "_")
        plt.figure()
        all_archs_feat_dfs = []
        for arch in range(enrichment_estimator.archetypes_.shape[0]):
            dists = enrichment_estimator.distance_to_archetypes_[:, arch]
            nans = np.isnan(d_feat)
            valid_dists = dists[~nans]  # ensure we consider only valid points in the binning process
            valid_feats = d_feat[~nans]
            n_per_bin = enrichment_estimator.bin_size * valid_dists.size
            bin_assign = (np.argsort(np.argsort(valid_dists)).astype(float) // n_per_bin).astype(
                int)  # use argsort twice to get ranking and then round to get binning
            sums = np.bincount(bin_assign, weights=valid_feats)
            means = sums / n_per_bin
            binned_disc_enrichment[d_feat_name]["arch#{}".format(arch + 1)] = means
            feat_df = pd.DataFrame({"distance to archetype": valid_dists,
                                    "bin number": bin_assign,
                                    d_feat_name: valid_feats.astype(float) / np.mean(valid_feats)})
            feat_df["Archetype"] = archetype_names[arch]
            all_archs_feat_dfs.append(feat_df)
        all_archs_feat_dfs = pd.concat(all_archs_feat_dfs, axis=0)
        sns.pointplot(x='bin number', y=d_feat_name, data=all_archs_feat_dfs, hue="Archetype",
                      palette=archetypes_palette, capsize=0.1, legend=add_legend)
        # if add_legend:
        #     plt.legend(title="archetype")
        plt.xlabel("distance from archetype quantile")
        plt.ylabel(f"{d_feat_name} fold enrichment")
        plt.savefig(
            opj(results_dir, "{}_enr_disc-{}_mean_est.png".format(run_desc, d_feat_name)))
        plt.close()

    for c_feat, c_feat_name in zip(cont_df.values.T,
                                   cont_df.columns):
        c_feat_name = c_feat_name.replace("/", "_")
        for arch, color in zip(range(enrichment_estimator.archetypes_.shape[0]), archetypes_palette):
            dists = enrichment_estimator.distance_to_archetypes_[:, arch]
            nans = np.isnan(c_feat)
            valid_dists = dists[~nans]  # ensure we consider only valid points in the binning process
            valid_feats = c_feat[~nans]
            plt.figure()
            s, a = choose_scatter_size_and_alpha(valid_dists, valid_feats, plt.gca())
            plt.scatter(x=valid_dists, y=valid_feats, color=color, s=s, alpha=a, edgecolors='none')
            plt.xlabel(f"distance from archetype #{arch + 1}")
            plt.ylabel(c_feat_name)
            plt.savefig(
                opj(results_dir, "{}_enr_cont-{}_scatter_arch-{}.png".format(run_desc, c_feat_name, arch + 1)))
            plt.close()

        plt.figure()
        all_archs_feat_dfs = []
        for arch in range(enrichment_estimator.archetypes_.shape[0]):
            dists = enrichment_estimator.distance_to_archetypes_[:, arch]
            nans = np.isnan(c_feat)
            valid_dists = dists[~nans]  # ensure we consider only valid points in the binning process
            valid_feats = c_feat[~nans]
            n_per_bin = enrichment_estimator.bin_size * valid_dists.size
            bin_assign = (np.argsort(np.argsort(valid_dists)).astype(float) // n_per_bin).astype(
                int)  # use argsort twice to get ranking and then round to get binning
            sums = np.bincount(bin_assign, weights=valid_feats)
            means = sums / n_per_bin
            binned_cont_enrichment[c_feat_name]["arch#{}".format(arch + 1)] = means
            feat_df = pd.DataFrame({"distance to archetype": valid_dists,
                                    "bin number": bin_assign,
                                    c_feat_name: valid_feats.astype(float)})
            feat_df["Archetype"] = archetype_names[arch]
            all_archs_feat_dfs.append(feat_df)
        all_archs_feat_dfs = pd.concat(all_archs_feat_dfs, axis=0)
        sns.pointplot(x='bin number', y=c_feat_name, data=all_archs_feat_dfs, hue="Archetype",
                      palette=archetypes_palette, estimator=np.median, capsize=0.1)
        plt.xlabel("distance from archetype quantile")
        plt.ylabel(f"median {c_feat_name}")
        plt.savefig(
            opj(results_dir, "{}_enr_cont-{}_median_est.png".format(run_desc, c_feat_name)))
        plt.close()

        sns.pointplot(x='bin number', y=c_feat_name, data=all_archs_feat_dfs, hue="Archetype",
                      palette=archetypes_palette, capsize=0.1)
        plt.xlabel("distance from archetype quantile")
        plt.ylabel(f"mean {c_feat_name}")
        plt.savefig(
            opj(results_dir, "{}_enr_cont-{}_mean_est.png".format(run_desc, c_feat_name)))
        plt.close()
