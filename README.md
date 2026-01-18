# qpu-design-automation-toolkit
This repository contains the implementation of a design automation pipeline for QPUs (based on quantum dots and spin qubits).

To install the environment necessary to run the project on any machine, run the following commands:
conda env create -f environment.yml
conda activate thesis-nn
python -m ipykernel install --user --name thesis-nn --display-name "Python (thesis-nn)"

Put simulation templates/configs in configs/
Put batch sweep runners in scripts/
Keep analysis + plots in notebooks/
Write reusable helpers in src/ (parsing outputs, plotting routines, etc.)
