
# Modular-DCM: Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference

Sound and complete algorithms have been proposed to compute identifiable causal queries using the causal structure and data. However, most of these algorithms assume accurate estimation of the data distribution, which is impractical for high-dimensional variables such as images. On the other hand, modern deep generative architectures can be trained to sample from high-dimensional distributions. However, training these networks is typically very costly. Thus, it is desirable to leverage pre-trained models to answer causal queries using such high-dimensional data. 

To address this, we propose modular training of deep causal generative models that not only makes learning more efficient but also allows us to utilize large, pre-trained conditional generative models. To the best of our knowledge, our algorithm, **Modular-DCM**, is the first algorithm that, given the causal structure, uses adversarial training to learn the network weights and can make use of pre-trained models to provably sample from any identifiable causal query in the presence of latent confounders. With extensive experiments on the Colored-MNIST dataset, we demonstrate that our algorithm outperforms the baselines. We also show our algorithm’s convergence on the COVIDx dataset and its utility with a causal invariant prediction problem on CelebA-HQ.

---
## Important Links

- **Paper**: [Modular-DCM on OpenReview](https://openreview.net/forum?id=bOhzU7NpTB)  
- **Podcast**: [Listen to our paper on Illuminate](https://illuminate.google.com/library?play=9733Qf_PyJhB)

---

## Key Features
- **Modular Training**: Efficiently trains deep causal models by modularizing the learning process.
- **Latent Confounders**: Handles high-dimensional causal queries even in the presence of latent confounders.
- **Pre-trained Models**: Integrates pre-trained conditional generative models to leverage state-of-the-art architectures without retraining.
- **High-dimensional Data**: Demonstrates efficacy with image data, addressing challenges in causal inference with high-dimensional variables.
- **Theoretical Guarantees**: Guarantees identifiability of causal queries using adversarial training.

---

## Experiments

### 1. Colored-MNIST (Semi-synthetic)
- **Objective**: Demonstrate Modular-DCM’s ability to handle high-dimensional causal queries with image mediators.
- **Graph**: Front-door graph (`D → Image → A` with `D ↔ A`).
- **Results**: Achieved superior convergence and fidelity compared to baselines; consistent with causal queries.

### 2. COVIDx CXR-3 (Real-world Medical Imaging)
- **Objective**: Validate Modular-DCM’s convergence on high-dimensional medical imaging datasets.
- **Graph**: `C → Xray → N` with latent confounders (`C ↔ N`).
- **Results**: Correctly matched observational and interventional distributions using pre-trained generative models.

### 3. CelebA-HQ (Invariant Prediction)
- **Objective**: Train causal invariant classifiers for robust attribute prediction under domain shifts.
- **Graph**: `Sex ↔ Eyeglass → Image; Sex → Eyeglass`.
- **Results**: Improved prediction accuracy under domain shifts by leveraging generated interventional datasets.

### 4. MNIST Diamond Graph (Complex Semi-synthetic)
- **Objective**: Test Modular-DCM’s performance with multiple image nodes and confounders.
- **Graph**: Diamond structure with `I1 → Digit → I2 → Color` and confounding.
- **Results**: Produced consistent interventional samples with better fidelity scores.

### 5. Asia/Lung Cancer Dataset (Low-dimensional Benchmark)
- **Objective**: Test Modular-DCM on classic causal inference datasets with latent confounders.
- **Results**: Achieved low total variation distance and KL divergence, proving correct interventional sampling.

### 6. Sachs Protein Dataset (Real-world Low-dimensional)
- **Objective**: Validate transportability and multi-dataset training.
- **Graph**: Semi-Markovian protein signaling network.
- **Results**: Correctly sampled interventional distributions using both observational and interventional datasets.



---


## Citation

If you use Modular-DCM in your work, please cite:

```bibtex
@inproceedings{
rahman2024modular,
title={Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference},
author={Md Musfiqur Rahman and Murat Kocaoglu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=bOhzU7NpTB}
}
```

## Contact

For questions or discussions about **Modular-DCM**, please reach out:

- **Md Musfiqur Rahman**  
  - Email: [rahman89@purdue.edu](mailto:rahman89@purdue.edu)  
  - Twitter: [@Musfiqshohan](https://twitter.com/Musfiqshohan)

- **Murat Kocaoglu (Advisor)**  
  - Twitter: [@murat_kocaoglu_](https://twitter.com/murat_kocaoglu_)

