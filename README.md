# Détection d’hologrammes à partir de vidéos

## Objectifs
This project aims to be able to detect the presence of a hologram on a passport and determine whether the document is original or fraudulent. To conduct this study, we used the MIDV-Holo dataset, which contains passport videos recorded using smartphones. 

<img src="images/holo_in_passport.png" alt="Pred" width="500"/>

Example of a hologram in a French passport

## Method : Random walks approach

From a video frame, we try to create a random path of pixels (information spatial). Over time, we have different frames of the passport, we will take the information of these pixels and combine them to create a 2D Spatio-Temporal Representation.

A hologram pixel is characterized by high saturation, which fluctuates over time. If an STR image of a region contains a hologram, there will be color variation between pixels over time. In contrast, a non-hologram STR will have no or tiny fluctuations over time.

<img src="images/diagram.png" alt="Pred" width="1000"/>

## Dataset

The dataset used to create the STRs (Spatio-Temporal Representation) for training is MIDV-Holo - [Link of dataset](https://github.com/SmartEngines/midv-holo)

The structure of dataset : 

<img src="images/dataset_structure.png" alt="Pred" width="500"/>

The dataset has:

- 150 videos of "origin" documents (with hologram).

- 150 videos of "fraud" documents. (50 copies of a document without hologram, 50 copies of a document with drawn in an image editor hologram pattern, and 50 printed photos of “original” document).


## Bibliographie

LI Koliaskina et al. “MIDV-Holo : A Dataset for ID Document Hologram Detection in a Video Stream”. In : International Conference on Document Analysis and Recognition. Springer. 2023, p. 486-503. doi : https://doi.org/10.1007/978-3-031-41682-8_30.

Harshal Chaudhari, Rishikesh Kulkarni et M.K. Bhuyan. “Weakly Supervised Learning based Reconstruction of Planktons in Digital In-line Holography”. In : Digital Holography and 3-D Imaging 2022. Optica Publishing Group, 2022, W5A.6. doi : 10.1364/DH.2022.W5A.6. url : https://opg.optica.org/abstract.cfm?URI=DH-2022-W5A.6.