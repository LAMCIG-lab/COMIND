# COMIND: Connectome-based Monotonic Inference of Neurodegenerative Dynamics

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2508.10343-b31b1b.svg)](https://arxiv.org/abs/2508.10343)
[![MICCAI 2025 Workshop](https://img.shields.io/badge/MICCAI%202025-Workshop-blue)](#citation)
[![MICCAI 2026](https://img.shields.io/badge/MICCAI%202026-Coming%20Soon-lightgrey)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) -->

COMIND is a mechanistic, connectome-constrained disease progression model for neurodegeneration. It models regional biomarker accumulation as a system of coupled logistic-diffusion trajectories over a structural brain connectome, enabling scalable and interpretable inference from longitudinal neuroimaging data.

---

## Method

At its core, COMIND models neurodegeneration as a $p$-dimensional ODE system:

$$\frac{dx}{dt} = [I - D(x)][Kx + f]$$

where $x(t) \in [0,1]^p$ are regional biomarker values, $K = s_t K^*$ is the structural connectome scaled by a global timescale, $D(x)$ enforces capacity-limited logistic growth, and $f$ is a sparse forcing term representing external sources of pathology. Each subject is assigned a disease-time offset $\beta_i$ mapping them onto the canonical trajectory. The full model has only $2p + 1$ parameters, making it tractable on small longitudinal cohorts.

**Coming soon — v1.1 (MICCAI 2026):** Automated disease subtype discovery via EM clustering of subtype-specific forcing terms $\{f_1, \ldots, f_Z\}$, with BIC-based selection of the number of subtypes. Validated on PPMI, recovering three morphological subtypes that align with established Parkinson's motor phenotypes (TD/PIGD) from cortical thickness trajectories alone.

---

## Installation

```bash
git clone https://github.com/LAMCIG-lab/COMIND.git
cd COMIND
pip install -e .
```

---

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

<!-- ---

## Funding

Supported by the Michael J. Fox Foundation grant MJFF-021683, *Multimodal Dynamic Modeling and Prediction of Parkinsonian Symptom Progression*. -->

---

## Contact

Daniel Semchin — `dsemchin@hawk.illinoistech.edu`  
Illinois Institute of Technology, Department of Biomedical Engineering  
Lab: [LAMCIG](https://github.com/LAMCIG-lab)
