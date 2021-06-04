# rev_adv

# Required packages

- torch
- torchvision
- foolbox
- art
- jive
- explore

To install explore and jive:
```
git clone https://github.com/idc9/py_jive.git
python setup.py install
```
```
git clone https://github.com/idc9/explore.git
python setup.py install
```

# Setup

- Set `download=True` in utils.py under setup folder to download the data, then set back to `download=False`
- Set `self.work_dir` to be the local working directory in constant.py under setup folder.

# Code

AJIVE classificationo:
- Train deep classifier: train_model.py
- Generate adversarial examples with foolbox and art: generate_adv_foolbox.py
- Generate adversarial features: generate_feature.py
- Perform ajive analysis on adversarial features from different attack methods: ajive_analysis.py
- Classify adversarial attack based on individual components: cla_ajive.py

Naive classifier:
- Combine adversarial examples to make a new dataset: getcomdata.py
- Classify attack methods based on adversarial examples: cla_naive.py
