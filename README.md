# ParTI-py
python implementation of the Pareto task inference (ParTI) algorithm.

Based on the original Matlab implementation of the Pareto Task Inference algorithm by [Hart et al.](https://www.nature.com/articles/nmeth.3254).
The original code is available [here](https://www.weizmann.ac.il/mcb/alon/download/pareto-task-inference-parti-method).


## Setup

## Running ParTI as a standalone script

You can use the `main.py` to run ParTI on your own data.
To do that all input files should be located in a single directory in the following files:

 - _features.csv_ - a csv file where each row is a datapoint and columns indicate data features 
used for the archetype analysis (with feature names as column headers). 
This file should not contain any NaN values or non numerical values! If the name of the first column is 'id'/ 'ID' it will be used as the index of the data.
 - _discrete_enrichment.csv_ - a csv file where each row corresponds to the discrete enrichment features of the same 
 data point in features.csv. All features with more than two values are replaced with binary indicator features. 
 Data should contain feature names as column headers. If `features.csv` contains a column named 'id'/'ID' it should also be present in this file (first column). Only samples that appear in the `features.csv` file will be used.
 - _continuous_enrichment.csv_ - a csv file where each row corresponds to the continuous enrichment features of the same 
 data point in features.csv. Data should contain feature names as column headers. If `features.csv` contains a column named 'id'/'ID' it should also be present in this file (first column). Only samples that appear in the `features.csv` file will be used.

The script has three run configurations:
 - `parti` - runs the full ParTI algorithm
 - `enrichment` - runs the enrichment analysis only
 - `choose_dimension` - runs the dimensionality reduction analysis only

To run the script, use the following command:
`Python3 main.py parti|choose_dimension|enrichment --output_dir <output_dir> --data_dir <data_dir>`'

*See code documentation (or run `python3 main.py --help`) for full description of script arguments.
 

 