# qpu-design-automation-toolkit
This repository contains the implementation of a design automation pipeline for QPUs (based on quantum dots and spin qubits).

To create the Conda environment, install the project in editable mode, and run the tests:
```bash
conda env create -f environment.yml
conda activate rj_thesis_project
python -m pip install -e .
python -m pytest -q
```

Put simulation templates/configs in configs/
Put batch sweep runners in scripts/
Keep analysis + plots in notebooks/
Write reusable helpers in src/ (parsing outputs, plotting routines, etc.)
