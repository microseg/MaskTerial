# MaskTerial: A Foundation Model for Automated 2D Material Flake Detection

![arXiv](https://img.shields.io/badge/arXiv-soon-b31b1b.svg)
[![Database Demo Website](https://img.shields.io/badge/DataGen-Demo-blue)](https://datagen.uslu.tech)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BibTeX](https://img.shields.io/badge/BibTeX-gray)](#Citing2DMatGMM)

After the paper is published this repository will host the code and related resources for the MaskTerial project, a robust Deep-Learning based model for real-time detection and classification of 2D material flakes.

## Abstract

The detection and classification of exfoliated two-dimensional (2D) material flakes from optical microscope images can be automated using computer vision algorithms.
This has the potential to increase the accuracy and objectivity of classification and the efficiency of sample fabrication, and it allows for large-scale data collection.
Existing algorithms often exhibit challenges in identifying low-contrast materials and typically require large amounts of training data.
Here, we present a deep learning model, called MaskTerial, that uses an instance segmentation network to reliably identify 2D material flakes.
The model is extensively pre-trained using a synthetic data generator, that generates realistic microscopy images from unlabeled data.
This results in a model that can to quickly adapt to new materials with as little as 5 to 10 images.
Furthermore, an uncertainty estimation model is used to finally classify the predictions based on optical contrast.
We evaluate our method on eight different datasets comprising five different 2D materials and demonstrate significant improvements over existing techniques in the detection of low-contrast materials such as hexagonal boron nitride.

## <a name="Citing2DMatGMM"></a>Citing MaskTerial

If you use our work or dataset in your research or find the code helpful, we would appreciate a citation to the original paper:  

```bibtex
Coming soon
```

## Contact

If you encounter any issues or have questions about the project, feel free to open an issue on our GitHub repository.
This repository is currently maintained by [Jan-Lucas Uslu](mailto:jan-lucas.uslu@rwth-aachen.de).
