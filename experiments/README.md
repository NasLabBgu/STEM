# Experiments

The experiments were performed on the IAC dataset containing data from 3 sources:
- FourForums
- ConvinceMe
- CreateDebate

The IAC dataset can be downloaded from [here](https://nlds.soe.ucsc.edu/iac2) (after submitting the required details).
The data is provided in 2 formats:
- mysql files (to be used with _mysql_)
- raw format (alternative) - `txt` files containing tab-delimited content, and the headers can be inferred from the accompanying `sql` file containing the scheme of the database.

### Prepare the datasets
In the experiments, The raw data is used to prepare the datasets for the experiments in a uniform convention. Then the same experiment can be performed on any of the sources.
Use the script `scripts/prepare-datasets.py` to prepare each of the datasets separately. It can be executed as follows:

```shell

$ python -m experiments.scripts.prepare_dataset -h

usage: prepare_dataset.py [-h] dataset path out

positional arguments:
  dataset     name of the dataset to prepare
  path        Path to the IAC directory containing all dataset as downloaded and extracted
  out         Output path to store the dataset in the new format (similar to VAST)

optional arguments:
  -h, --help  show this help message and exit
```

For example, executing the following command:
```shell
$ python -m experiments.scripts.prepare-dataset 4forums experiments/data/fourforums experiments/data/fourforums/4forum.csv
```

will be resulted with the prepared dataset at `experiments/data/fourforums/4forum.csv` of _fourforums_ for the experiments

## Running experiment 1
### Conversation structure only

put some descriptiong \
... \
and some commands for example

## Running experiment 2
### Combining (the original) Zero-Shot stance predictions (put reference)

1. Run the `prepare2zs.py` script to prepare the dataset for the zer-shot predictions. \
2. he run te zero-shot predictions. (dedicated script)\
3. Then take the results and merge with the corresponding dataset (that was prepared for experiment 1) (dedicted script)\
4. run experiment 2 using the merged dataset






