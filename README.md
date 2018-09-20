## Description
Python implementation of Hierarchical Probabilistic Principal Component Analysis proposed in [1].

HPPCA significantly improves dimesionality reduction performance by absorbing our prior knowledge about 
the group structure of the features and by decreasing the number of parameter from individual components.

![vis](https://github.com/tochikuji/GitHub-Assets/blob/master/images/hppca.png?raw=true)
_originated from [1]._


## Installation
```
python setup.py install
```

or

```
pip install .
```

in the root of this repository.

## Requirements

- Python >3.5
- scikit-learn
- numpy

## Usage
See `examples`

## References

[1] Aiga Suzuki, Hayaru Shouno, "Generative Model of Textures Using Hierarchical Probabilistic Principal Component Analysis", Proc. of PDPTA’17, CSREA Press, pp.333-338, 2017.
