set -e

python -m unittest discover -s code
python code/simple_model.py configs/default.ini
python -m flake8 code
