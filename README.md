# UGML

This repository provides the code for the UGML in Uncertainty-Guided Model Learning for Trustworthy Medical Image Segmentation.

# Introduction
The segmentation of medical images plays a crucial role in aiding clinical diagnosis and disease screening. Although recent AI algorithms have greatly improved accuracy, there is still skepticism among clinicians about the reliability and trustworthiness of these algorithms when used in real-life situations. To address this, uncertainty estimation is a useful method for enhancing the credibility of models. It allows us to measure the reliability of network outputs and identify instances where the network may perform poorly. In our research, we introduce a new method for trustworthy medical image segmentation. This method aims to produce dependable segmentation results and reliable uncertainty estimations without imposing excessive computational burden. One key feature of our approach is the extraction of feature maps from each decoder and their fusion based on voxel-level uncertainty to fully leverage multi-scale semantic information. We also model the probability and uncertainty of medical image segmentation problems using subjective logic theory, which quantifies the uncertainty of the backbone by parameterizing class probabilities as a Dirichlet distribution and calculating the distribution strength. Moreover, our framework learns to gather reliable evidence from the feature, which leads to the final segmentation results. To further improve model performance using voxel-level uncertainty, we introduce an uncertainty-based adaptive threshold strategy. This strategy dynamically adjusts the threshold based on the model's learning state, guiding the model to focus on regions with high uncertainty to optimize segmentation results. Extensive experiments on multiple public datasets have validated the effectiveness of our approach, and we plan to make the code publicly available. 
![Image](https://github.com/user-attachments/assets/abe22355-25fb-4203-b2a3-f0e725ec057d)

# Data Availability
LiTS 2017: http://www.lits-challenge.com  
Flare 2022: https://flare22.grand-challenge.org/

# Notice
The complete code will be released when the paper is published.
