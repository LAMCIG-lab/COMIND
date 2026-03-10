# COMIND: Connectome-based Monotonic Inference of Neurodegenerative Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-2508.10343-b31b1b.svg)](https://arxiv.org/abs/2508.10343)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

COMIND is a mechanistic, connectome-constrained disease progression model for neurodegeneration. It models regional biomarker accumulation as a system of coupled logistic-diffusion trajectories over a structural brain connectome, enabling scalable and interpretable inference and disease subtype discovery from longitudinal neuroimaging data.

## Method

COMIND models neurodegeneration as a $p$-dimensional ODE system:

$$\frac{dx}{dt} = [I - D(x)][Kx + f]$$

where $x(t) \in [0,1]^p$ are regional biomarker values, $K = s_t K^*$ is the structural connectome scaled by a global timescale, $D(x)$ enforces capacity-limited logistic growth, and $f$ is a sparse forcing term representing external sources of pathology. Each subject is assigned a disease-time offset $\beta_i$ mapping them onto the canonical trajectory. The full model has only $2p + 1$ parameters, making it tractable on small longitudinal cohorts.

COMIND supports automated subtype discovery by inferring $Z$ subtype-specific forcing terms $\{f_1, \ldots, f_Z\}$ via a generalized EM algorithm, with BIC-based selection of the number of subtypes. Validated on PPMI, the model recovers three morphological subtypes from cortical thickness trajectories alone that align with established Parkinson's motor phenotypes (TD/PIGD), without any clinical priors.

## Installation

```bash
git clone https://github.com/LAMCIG-lab/COMIND.git
cd COMIND
pip install -e .
```

## Citation

If you use COMIND, please cite:
```bibtex
@article{semchin2025comind,
  title={Scalable Modeling of Nonlinear Network Dynamics in Neurodegenerative Disease},
  author={Semchin, Daniel and d'Angremont, Emile and Lorenzi, Marco and Gutman, Boris A.},
  journal={arXiv preprint arXiv:2508.10343},
  year={2025}
}
```

## Contact

Daniel Semchin — `dsemchin@hawk.illinoistech.edu`  
Illinois Institute of Technology, Department of Biomedical Engineering  
Lab: [LAMCIG](https://github.com/LAMCIG-lab)
