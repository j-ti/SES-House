# SES-House

## Goal
Get good result to publish a paper!

## Assignment 3

Do optimization for the following 3 scenarios:
* minimize electricity costs while participating in the wholesale market
* minimize electricity costs without participating in the wholesale market
* minimize green house gas emissions


## Programming

TODO add travis build status

### Get started

To get started, install anaconda. Therefore follow the installation instruction
on [this website](https://docs.anaconda.com/anaconda/install/linux/).
```bash
make install
conda activate ses-house
. env
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
