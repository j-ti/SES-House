# SES-House

This is the source code of a project we developed in the course Smart Energy
Systems at the TU Berlin, look [here](https://www.aot.tu-berlin.de/index.php?id=2895)
for course description.

## Requirements
* Python
* [Anaconda](https://docs.anaconda.com/anaconda)
* [Gurobi](https://www.gurobi.com/)

## Data
* ./sample: should contain the sample files for test functions
* ./data: should contain the dataset files for the forecasting and optimization

We use data from [PecanStreet](https://dataport.pecanstreet.org/) in order to do
the forecasting and optimization.
We use PecanStreet for PV power generation, house loads and price data.

## Programming

[![Build Status](https://api.travis-ci.com/j-ti/SES-House.svg?token=9GudSoJGkvnBmiR1HWN7&branch=master)](https://travis-ci.com/j-ti/SES-House)

### Run code

In order to run the code you need to have the files mentioned in the
configuration, e.g. `configs/default.ini`.

Create our conda environment and activate it:
```
conda env create -f conda.yml
conda activate ses-house
```

Run our default example:
```
python code/simple_model.py configs/default.ini
```

Run our Vermont example:
```
python code/simple_model.py configs/Vermont.ini
```

Run our Forecasting PV example:
```
python code/forecast_pv.py
```

Run our Forecasting using the saved model example:
```
python code/forecast_pv.py 1
```

Run our Forecasting Load example:
```
python code/forecast_load.py
```
Check the `code/forecast_load_conf.py` and `code/forecast_conf.py` to adapt the
configuration.

### Get started

The following instructions should work Linux environments.
To get started, install anaconda. Therefore follow the installation instruction
on [this website](https://docs.anaconda.com/anaconda/install/linux/).
```bash
make install
conda activate ses-house
```

To be sure that the code is working do:
```bash
make test
```

When editing the code do this command to format the code:
```bash
make black
```

When something weird happens and for cleanup:
```bash
make clean
```

When you add or remove packages from our `ses-house` conda environment, update
the environment file `conda.yml` with this command:
```bash
make export
```

## Help

If you would like to try this out and have some questions, comments or
improvements, feel free to write us or open an issue.
