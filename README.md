#  Knowledge-data fusion dominated vehicle platoon dynamics modeling and analysis: A physics-encoded deep learning approach

## Overview
The code repository for the study titled **"Knowledge-data fusion dominated vehicle platoon dynamics modeling and analysis: A physics-encoded deep learning approach"** will be made available upon acceptance of the manuscript.



## Main Contributions

- A novel vehicle platoons dynamics modeling framework (PeDL) with interpretability and accuracy is developed. It focuses on directly learning and encoding the parameters of the generalized platoon model, aiming to avoid excessive dependence on prior knowledge, and to maximize the utility of the framework. Moreover, PeDL can be scaled to model platoons with different numbers of vehicles and highly interactive scenarios. 
- IA multi-scale feature learning network (MTFLN) is designed to facilitate the end-to-end learning of the parameters necessary for PeDL. The network initially extracts vehicle-level features from naturalisti driving platoon trajectories. With these features as the center, the causal attention mechanism is employed to learn comprehensive platoon-level features and implements multi-step prediction.
- A platoon trajectory dataset is extracted from [HIGH-SIM](https://github.com/CATS-Lab/Filed-Experiment-Data-HIGH_Sim) dataset and applied to PeDL. The proposed PeDL can reproduce the ground-truth of platoon following behavior with high accuracy. In particular, PeDL succeeded in replicating both the stability and safety evolution of the platoon, highlighting its exceptional capacity in physical analyzability.



## Abstract

Nonlinear platoon dynamics modeling plays a crucial role in predicting and optimizing the interactions between vehicles, and the importance of its accuracy and physical analyzability is self-evident. At present, platoon-level modeling often relies on the extension of vehicle-level car following, lacking the extraction and capture of vehicle behavior interaction features at the platoon scale. More importantly, theoretical knowledge within the physics-informed deep learning (PIDL)-based car following models often plays a guiding role rather than being directly encoded as the learning object. However, these physical parameters are precisely the prerequisites for achieving analyzable and reproducible traffic characteristics. In order to fully utilize the analytical power of physical parameters, this paper proposed a novel physics-encoded deep learning (PeDL) framework to model the nonlinear platoon dynamics. In this framework, an analyzable parameters encoder computational graph (APeCG) is designed to encode physical parameters and ensure local stability. Beside, a multi-scale trajectory feature learning network (MTFLN) is constructed to learn and infer convoy following patterns from trajectory data. By that, the seemingly contradictory physical analyzability and inherent high precision of deep learning can be resolved, providing deductive insights. To test the performance of our proposed PeDL, experiments based on human-driven vehicle trajectory datasets (HIGHSIM) are conducted. The results indicate that PeDL is effective in platoon dynamics modeling in response to the speed of the leading vehicle and outperforms the baselines. Significantly, PeDL can reproduce the real-world platoon driving behaviors with statistical realism, including stability and safety, which is expected. Code is public: [PeMTFLN](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN).



## Framework

Overall **“parameters encoder multi-scale trajectory feature learning network (PeMTFLN)”** 
![framework](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/framework.png)




## Environment

- **Operating System**: Ubuntu 20.04
- **CUDA Version**: 11.4



## Train



## Evaluation

## Quantitative results

 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/box.png)
 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/accuracy-index.png)
 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/accuracywholevis.png)

## Qualitative results
The reproduced platoon trajectory examples of PeMTFLN
 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/accuracylocalvis.png)
The platoon stability evolution replicated by PeMTFLN
 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/stability.png)
The SSDD and PET of platoon replicated by PeMTFLN
 ![image](https://github.com/SpaceTrafficSafetyTeam/PeMTFLN/blob/main/safety.png)





## Citation

 


