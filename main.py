import json
import os
from os.path import join as opj
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from parti import ParTI, ALG_PCA, ALG_SISAL_PY, EnrichmentCalculator, ALG_PCHA, elbow_chooser
from viz_utils import save_results_viz, save_results_data, save_enrichment_results_viz, save_enrichment_results_data, \
    get_palette, save_explained_var_results, plot_elbow_results

FEATURES_NAME = "features.csv"
DISCRETE_NAME = "discrete_enrichment.csv"
CONTINUOUS_NAME = "continuous_enrichment.csv"


def is_boolean(unique_vals):
    if len(unique_vals) != 2:
        return False
    else:
        if bool(unique_vals[0]) ^ bool(unique_vals[1]):
            return True
    return False


def load_data(data_dir: str, no_enrichment: bool = False, discrete_two_sided: bool = False):
    assert os.path.exists(data_dir), "Invalid data directory path"
    features_path = opj(data_dir, FEATURES_NAME)
    assert os.path.exists(features_path), "Features file does not exist"
    features_df = pd.read_csv(features_path, delimiter=",")
    assert not features_df.isnull().values.any(), "Can't run ParTI with NaN feature values"


    indices = None
    if features_df.columns[0].lower() == "id":
        logging.info("Interpreting first column of {} as ID column".format(FEATURES_NAME))
        features_df = features_df.set_index(features_df.columns[0])
        indices = features_df.index.to_numpy()
    feature_names = list(features_df.keys())
    features_df = features_df.astype(float)
    logging.info("Using the following {} columns as features for simplex finding: \n {}".format(len(feature_names),
                                                                                                ", ".join(
                                                                                                    feature_names)))

    if no_enrichment:
        return features_df
    discrete_path = opj(data_dir, DISCRETE_NAME)
    if os.path.exists(discrete_path):
        discrete_df = pd.read_csv(discrete_path)
        if discrete_df.columns[0].lower() == "id" and (indices is not None):
            logging.info("Interpreting first column of {} as ID column".format(DISCRETE_NAME))
            discrete_df = discrete_df.set_index(discrete_df.columns[0])
            discrete_df = discrete_df.reindex(indices)
            discrete_df = discrete_df.loc[indices]
        # make all discrete variables binary
        all_columns = list(discrete_df.keys())
        non_binary_columns = [col for col in all_columns if not is_boolean(discrete_df[col].unique())]
        discrete_df = pd.get_dummies(discrete_df, columns=non_binary_columns)
        if discrete_two_sided:
            # Add negated variables to test for both enrichment and depletion
            discrete_negated = ~discrete_df.astype(bool)
            discrete_negated.columns = discrete_negated.columns.map(lambda n: "not-{}".format(n))
            discrete_df = pd.concat([discrete_df, discrete_negated], axis=1)
        discrete_df = discrete_df.astype(float)
        logging.info(
            "Using the following columns as features for discrete enrichment: \n {}".format(", ".join(discrete_df.columns)))

    else:
        logging.info("No discrete enrichment table was found in the data directory.")
        discrete_df = pd.DataFrame(index=features_df.index, columns=[], dtype=float)


    continuous_path = opj(data_dir, CONTINUOUS_NAME)
    if os.path.exists(continuous_path):
        continuous_df = pd.read_csv(continuous_path)
        if continuous_df.columns[0].lower() == "id" and (indices is not None):
            logging.info("Interpreting first column of {} as ID column".format(CONTINUOUS_NAME))
            continuous_df = continuous_df.set_index(continuous_df.columns[0])
            continuous_df = continuous_df.reindex(indices)
            continuous_df = continuous_df.loc[indices]

        logging.info(
            "Using the following columns as features for continuous enrichment: \n {}".format(
                ", ".join(continuous_df.columns)))
        continuous_df = continuous_df.astype(float)
    else:
        logging.info("No continuous enrichment table was found in the data directory.")
        continuous_df = pd.DataFrame(index=features_df.index, columns=[], dtype=float)
    logging.info(f"Found {features_df.shape[0]} samples and {features_df.shape[1]} features, {discrete_df.shape[1]} discrete features and {continuous_df.shape[1]} continuous features.")
    return features_df, discrete_df, continuous_df


class Main():
    def _set_out_dir(self, out_dir):
        self._out_dir = out_dir
        assert out_dir != "", "Output dir should be specified explicitly!"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def _set_run_desc(self, run_desc):
        if run_desc != "":
            run_desc = run_desc + "_"
        self._run_desc = run_desc

    def _set_logging(self):
        print("Setting up logging...")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(opj(self._out_dir, "{}log.log".format(self._run_desc))),
                logging.StreamHandler()
            ]
        )

    def parti(self,
              out_dir,
              data_dir,
              max_dim=20,
              n_archetypes=-1,
              fdr_threshold=0.1,
              bin_size=0.1,
              n_permutations=1000,
              n_bootstraps=1000,
              max_boot_samples=-1,
              simplex_alg=ALG_SISAL_PY,
              choose_dim_by=ALG_PCA,
              run_desc="",
              seed=-1,
              bind_archetypes="",
              project_by="",
              discrete_two_sided=False,
              archetypes_palette=""
              ):
        """
       @param out_dir: directory where ParTI results will be saved
       @param data_dir: directory with the data files to run ParTI on
       @param max_dim: maximal number of possible data dimensions to analyze
       @param n_archetypes: predetermined number of archetypes. If not specified, elbow method will be used to choose it.
       @param fdr_threshold: False discovery rate threshold for the enrichment analysis. Used separately for continuous and
       discrete features.
       @param bin_size: portion of the data to hold in each bin for enrichment analysis.
       @param n_permutations: number of permutation test trials to use for archetype significance test.
       @param n_bootstraps: number of bootstrap trials to use for archetype confidence interval estimation.
       @param max_boot_samples: maximal number of points to resample for each bootstrap sample. If it is -1 the 
       bootstrap sample size will be of the same size as the original sample. 
       @param simplex_alg: name of algorithm to use for archetype analysis. currently supports only "ALG_SISAL".
       @param choose_dim_by: method to use for choosing the dimensionality of the data. One of "pca" or "pcha".
       @param run_desc: prefix to add for all output files (in case that multiple results are saved to the same directory)
       @param bind_archetypes: a path to a csv with archetypes in the original space in the same format as the output
       of this code. If provided, output archetypes indices will be matched to the archetypes in the file (with minimal
       euclidean distance cost). If the number of archetypes k in the file is less than n - the number of archetypes in
       this run, k out of the n archetypes will be matched such that the total cost will be minimal and the rest of n-k
       indices will be arbitrary.
       @param project_by: a path to another table of samples with the same features as features.csv to use for computing the pca instead of pca of the features.csv data.
       @param discrete_two_sided: whether to test for both enrichment and depletion for discrete features
       @param archetypes_palette: a string with comma-separated hex colors to use for the archetypes visualization
       @return/:
        """
        self._set_out_dir(out_dir=out_dir)
        if bin_size == -1.0:
            bin_size = None
        if n_archetypes == -1:
            n_archetypes = None
        if seed != -1:
            np.random.seed(seed)
        self._set_run_desc(run_desc=run_desc)

        with open(os.path.join(self._out_dir, f"{self._run_desc}run-params.json"), "w") as f:
            json.dump({
                "out_dir": out_dir,
                "data_dir": data_dir,
                "n_archetypes": n_archetypes,
                "fdr_threshold": fdr_threshold,
                "bin_size": bin_size,
                "n_permutations": n_permutations,
                "n_bootstraps": n_bootstraps,
                "max_boot_samples": max_boot_samples,
                "simplex_alg": simplex_alg,
                "seed": seed,
                "run_desc": run_desc,
                "discrete_two_sided": discrete_two_sided,
                "project_by": project_by,
                "archetypes_palette": archetypes_palette,
                "bind_archetypes": bind_archetypes,
                "choose_dim_by": choose_dim_by
            }, f, indent=1)

        self._set_logging()
        logging.info("Loading data...")
        features_df, discrete_df, continuous_df = load_data(
            data_dir, discrete_two_sided=discrete_two_sided)

        partii = ParTI(max_dim=max_dim,
                       simplex_alg=simplex_alg,
                       n_archetypes=n_archetypes,
                       bin_size=bin_size,
                       fdr_threshold=fdr_threshold,
                       verbose=True,
                       n_permutations=n_permutations,
                       n_bootstraps=n_bootstraps,
                       max_boot_samples=max_boot_samples,
                       choose_dim_by=choose_dim_by,
                       bind_archetypes=None if bind_archetypes == "" else bind_archetypes,
                       project_by=None if project_by == "" else project_by)

        logging.info("Running ParTI pipeline...")
        partii.fit(features_df.values, continuous_df.values, discrete_df.values)
        logging.info("P value for simplex = {}".format(partii.archetypes_p_val_))
        logging.info("Saving results to directory: {}".format(self._out_dir))
        partii.save_evaluation(opj(self._out_dir, "{}results.npz".format(self._run_desc)))

        save_results_data(results_dir=self._out_dir,
                          parti_estimator=partii,
                          disc_features_descs=list(discrete_df.columns),
                          cont_features_descs=list(continuous_df.columns),
                          data_features_descs=list(features_df.columns),
                          samples_index=list(features_df.index),
                          run_desc=self._run_desc)

        save_results_viz(results_dir=self._out_dir,
                         parti_estimator=partii,
                         disc_df=discrete_df,
                         cont_df=continuous_df,
                         archetypes_palette=get_palette(archetypes_palette),
                         run_desc=self._run_desc)

    def enrichment(self,
                   out_dir,
                   data_dir,
                   archetypes_path,
                   fdr_threshold=0.1,
                   bin_size=0.1,
                   seed=-1,
                   run_desc="",
                   discrete_two_sided=False,
                   project_by="",
                   archetypes_palette="",
                   archetype_names=""):
        """
       @param out_dir: path to the directory where ParTI results will be saved
       @param data_dir: path to the directory with the data files to run ParTI enrichment on. Those files should be in
       the same format as for running the full ParTI algorithm.
       @param archetypes_path: path to the csv with the archetypes in the original space (same as the dimensions of
       features.csv) in the format in which it is saved by the ParTI script.
       @param fdr_threshold: False discovery rate threshold for the enrichment analysis. Used separately for continuous
       and discrete features.
       @param bin_size: portion of the data to hold in each bin for enrichment analysis.
       @param run_desc: prefix to add for all output files (in case that multiple results are saved to the same
       directory)
       @param seed: random seed for reproducibility
       @param discrete_two_sided: whether to test for both enrichment and depletion for discrete features
       @param project_by: a path to another table of samples with the same features as features.csv to use for computing the pca instead of pca of the features.csv data.
       @param archetypes_palette: a string with the name of the palette to use for the archetypes visualization
       @param archetype_names: a comma-separated string with the names of the archetypes
       @return:
        """
        self._set_out_dir(out_dir=out_dir)
        self._set_run_desc(run_desc=run_desc)

        with open(os.path.join(out_dir, f"{run_desc}run-params.json"), "w") as f:
            json.dump({
                "out_dir": self._out_dir,
                "data_dir": data_dir,
                "archetypes_path": archetypes_path,
                "fdr_threshold": fdr_threshold,
                "bin_size": bin_size,
                "seed": seed,
                "run_desc": self._run_desc,
                "discrete_two_sided": discrete_two_sided,
                "project_by": project_by,
                "archetypes_palette": archetypes_palette,
                "archetype_names": archetype_names
            }, f, indent=1)

        if bin_size == -1.0:
            bin_size = None

        if seed != -1:
            np.random.seed(seed)

        self._set_logging()

        logging.info("Loading data...")
        features_df, discrete_df, continuous_df = load_data(
            data_dir, discrete_two_sided=discrete_two_sided)
        archetypes = pd.read_csv(archetypes_path)
        arch_columns = [col for col in archetypes.columns if col.startswith("arch_")]
        n_archetypes = len(arch_columns)
        archetypes_features = archetypes[arch_columns].values.T
        enrich_calc = EnrichmentCalculator(bin_size=bin_size, fdr_threshold=fdr_threshold)
        pca = PCA(n_components=n_archetypes - 1)
        if project_by != "":
            pca_fit_data = pd.read_csv(project_by)
            start_column = 1 if (pca_fit_data.shape[1] > features_df.shape[1]) else 0
            pca_fit_data = pca_fit_data.iloc[:, start_column:].values
        else:
            pca_fit_data = features_df.values
        if archetype_names != "":
            archetype_names = archetype_names.split(",")
            assert len(archetype_names) == n_archetypes
        else:
            archetype_names = ["archetype #{}".format(i) for i in range(1, n_archetypes + 1)]
        pca.fit(pca_fit_data)
        features_pca = pca.transform(features_df.values)
        archs_pca = pca.transform(archetypes_features)
        enrich_calc.fit(features_pca, archs_pca, discrete_df.values, continuous_df.values)
        save_enrichment_results_viz(results_dir=self._out_dir,
                                    enrichment_estimator=enrich_calc,
                                    disc_df=discrete_df,
                                    cont_df=continuous_df,
                                    run_desc=self._run_desc,
                                    archetypes_palette=get_palette(archetypes_palette),
                                    archetype_names=archetype_names)

        save_enrichment_results_data(results_dir=self._out_dir,
                                     enrichment_estimator=enrich_calc,
                                     disc_features_descs=list(discrete_df.columns),
                                     cont_features_descs=list(continuous_df.columns),
                                     run_desc=self._run_desc)

    def choose_dimension(self,
                         out_dir,
                         data_dir,
                         max_dim=-1,
                         by="pca",
                         run_desc="",
                         seed=-1
                         ):
        self._set_out_dir(out_dir=out_dir)
        if seed != -1:
            np.random.seed(seed)
        self._set_run_desc(run_desc=run_desc)

        with open(os.path.join(self._out_dir, f"{self._run_desc}run-params.json"), "w") as f:
            json.dump({
                "out_dir": out_dir,
                "data_dir": data_dir,
                "max_dim": max_dim,
                "by": by,
                "seed": seed,
                "run_desc": run_desc,
            }, f, indent=1)

        self._set_logging()
        logging.info("Loading data...")
        features_df = load_data(
            data_dir, no_enrichment=True)
        if max_dim != -1:
            assert max_dim <= features_df.shape[1], "Maximal dimension should be less than the number of features"
            assert max_dim >= 3, "Maximal dimension should be at least 3"
        max_dims = [max_dim] if max_dim != -1 else range(3, 21)
        if by.lower() == "pca":
            by = ALG_PCA
        elif by.lower() == "pcha":
            by = ALG_PCHA
        else:
            raise ValueError("Unknown dimensionality reduction method: {}".format(by))
        partii = ParTI(max_dim=max_dims[-1],
                       simplex_alg=ALG_SISAL_PY,
                       n_archetypes=None,
                       bin_size=0.1,
                       fdr_threshold=0.1,
                       verbose=True,
                       n_permutations=10,
                       n_bootstraps=10,
                       max_boot_samples=10,
                       choose_dim_by=by,
                       bind_archetypes=None,
                       project_by=None)
        partii.choose_dim(features_df.values)
        all_dim_choices = []
        for max_dim in max_dims:
            ev = partii.explained_vars_ if by == ALG_PCA else partii.pcha_explained_vars_
            dim = elbow_chooser(ev[:max_dim])
            all_dim_choices.append(dim)
            logging.info(f"Chosen dimension (for maximal dimension of {max_dim}): {dim}")
        if len(all_dim_choices) > 1:
            plot_elbow_results(all_dim_choices, max_dims, self._out_dir, self._run_desc)

        logging.info("Saving results to directory: {}".format(self._out_dir))
        partii.save_evaluation(opj(self._out_dir, "{}results.npz".format(self._run_desc)))
        save_explained_var_results(results_dir=self._out_dir,
                                   parti_estimator=partii,
                                   run_desc=self._run_desc)


if __name__ == '__main__':
    main = Main()
    argparser = argparse.ArgumentParser()
    # define subcommands for the different main methods:
    subparsers = argparser.add_subparsers(dest="subcommand")
    # define the arguments for each subcommand:
    # parti
    parti_parser = subparsers.add_parser("parti")
    parti_parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save results in")
    parti_parser.add_argument("--data_dir", type=str, required=True,
                              help="Directory with the data files to run ParTI on")
    parti_parser.add_argument("--max_dim", type=int, default=20, help="Maximal dimension to check for choosing the number of archetypes")
    parti_parser.add_argument("--n_archetypes", type=int, default=-1, help="Number of archetypes to use. If not specified, will be chosen by elbow method")
    parti_parser.add_argument("--fdr_threshold", type=float, default=0.1, help="FDR threshold for enrichment analysis")
    parti_parser.add_argument("--bin_size", type=float, default=0.1, help="Fraction of datapoint per bin for enrichment analysis")
    parti_parser.add_argument("--n_permutations", type=int, default=1000, help="Number of permutations for the t-ratio significance test")
    parti_parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstraps for the archetypes confidence ellipse estimation")
    parti_parser.add_argument("--max_boot_samples", type=int, default=-1, help="Maximal number of samples to resample for each bootstrap. If -1, will be the same as the original sample size")
    parti_parser.add_argument("--simplex_alg", type=str, default=ALG_SISAL_PY, help=f"Algorithm to use for simplex analysis. One of {ALG_SISAL_PY}, {ALG_PCHA}")
    parti_parser.add_argument("--choose_dim_by", type=str, default=ALG_PCA, help="method to use for choosing the dimensionality of the data. One of 'pca' or 'pcha'.")
    parti_parser.add_argument("--run_desc", type=str, default="", help="Prefix to add to the name of all output files")
    parti_parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility")
    parti_parser.add_argument("--bind_archetypes", type=str, default="", help="Path to a csv with archetypes in the original space to match the output archetypes to")
    parti_parser.add_argument("--project_by", type=str, default="", help="Path to a csv with samples to use for PCA projection instead of the features.csv data")
    parti_parser.add_argument("--discrete_two_sided", action="store_true", help="Whether to test for both enrichment and depletion for discrete features")
    parti_parser.add_argument("--archetypes_palette", type=str, default="", help="Comma-separated string with hex colors to use for the archetypes visualization")
    parti_parser.set_defaults(func=main.parti)

    # enrichment
    enrichment_parser = subparsers.add_parser("enrichment")
    enrichment_parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save results in")
    enrichment_parser.add_argument("--data_dir", type=str, required=True,
                                   help="Directory with the data files to run ParTI on")
    enrichment_parser.add_argument("--archetypes_path", type=str, required=True, help="Path to the csv with the archetypes in the original space")
    enrichment_parser.add_argument("--fdr_threshold", type=float, default=0.1, help="FDR threshold for enrichment analysis")
    enrichment_parser.add_argument("--bin_size", type=float, default=0.1, help="Fraction of datapoint per bin for enrichment analysis")
    enrichment_parser.add_argument("--run_desc", type=str, default="", help="Prefix to add to the name of all output files")
    enrichment_parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility")
    enrichment_parser.add_argument("--discrete_two_sided", action="store_true", help="Whether to test for both enrichment and depletion for discrete features")
    enrichment_parser.add_argument("--project_by", type=str, default="", help="Path to a csv with samples to use for PCA projection instead of the features.csv data")
    enrichment_parser.add_argument("--archetypes_palette", type=str, default="", help="Comma-separated string with hex colors to use for the archetypes visualization")
    enrichment_parser.add_argument("--archetype_names", type=str, default="", help="Comma-separated string with the names of the archetypes")
    enrichment_parser.set_defaults(func=main.enrichment)

    # choose_dimension
    choose_dim_parser = subparsers.add_parser("choose_dimension")
    choose_dim_parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save results in")
    choose_dim_parser.add_argument("--data_dir", type=str, required=True,
                                   help="Directory with the data files to run ParTI on")
    choose_dim_parser.add_argument("--max_dim", type=int, default=-1, help="Maximal dimension to check for choosing the number of archetypes. If -1, will check up to 20 and will report all choices")
    choose_dim_parser.add_argument("--by", type=str, default="pca", help="Method to use for choosing the dimension. One of 'pca' or 'pcha'")
    choose_dim_parser.add_argument("--run_desc", type=str, default="", help="Prefix to add to the name of all output files")
    choose_dim_parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility")
    choose_dim_parser.set_defaults(func=main.choose_dimension)

    # parse and call relevant method
    args = argparser.parse_args()
    kwargs = {k:v for k,v in vars(args).items() if k not in ["subcommand", "func"]}
    print(args)
    args.func(**kwargs)
