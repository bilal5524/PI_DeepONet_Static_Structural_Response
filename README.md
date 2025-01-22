# PI_DeepONet_Static_Structural_Response

This repository contains configurations and code for implementing structural analysis using [**DeepXDE**](https://deepxde.readthedocs.io/) with both PaddlePaddle and TensorFlow backends. The configurations are tailored for the following structures:  
1. **2D Beam Structure**  
2. **KW51 Structure**

## Installation

1. Install the DeepXDE library along with the required backend:  
   - **PaddlePaddle** for the 2D Beam Structure.  
   - **TensorFlow** for the KW51 Structure.  

---

## 2D Beam Structure

### Configuration
- **Backend**: PaddlePaddle  
- **Strategy**: Split Branch and Trunk Strategy  

### Data Folder
The `Data` folder contains:
- Input files  
- Output files  
- Stiffness matrix  
- Load location information  

### Required Modifications

1. **DeepONet Implementation**  
   Modify the `deeponet.py` file in the DeepXDE Paddle backend folder:
   
<Python_Installation_Directory>\Lib\site-packages\deepxde\nn\paddle\deeponet.py

based on the `deeponet.py` file provided in this repository.

2. **Loss Function**  
Update the loss functions:

<Python_Installation_Directory>\Lib\site-packages\deepxde\losses.py

based on `losses.py` file in this repository to implement the (DD + EC) loss function.

3. **Model Configuration**  
Modify the model file to select the required loss function (either DD or DD + EC):  

<Python_Installation_Directory>\Lib\site-packages\deepxde\model.py


---

## KW51 Structure

### Configuration
- **Backend**: TensorFlow  
- **Strategy**: "Independent" DeepONet Strategy  

### Data Folder
The `Data` folder contains:
- Stiffness matrix  
- Inputs and outputs associated with the following loss functions:  
- DD Loss  
- DD + EC Loss  
- DD + Schur Loss

### Abaqus Model
The Abaqus model for the **KW51 structure** is also included in this repository.  
- This model has been developed and validated.  
- The data provided in the `Data` folder (stiffness matrix, inputs, outputs, etc.) has been generated based on this validated Abaqus model.

### Required Modifications

1. **DeepONet Implementation**  
Update the DeepONet configuration for TensorFlow by modifying:  

<Python_Installation_Directory>\Lib\site-packages\deepxde\nn\tensorflow\deeponet.py

Use the `deeponet.py` file in this repository to implement the Independent DeepONet strategy.

2. **Loss Function Implementation**  
Update the loss function file to apply custom PI loss functions:  

<Python_Installation_Directory>\Lib\site-packages\deepxde\losses.py

based on the `losses.py` file in this repository.

---

## Notes
- Ensure that all file paths and configurations match your local Python environment.  
- Refer to the [DeepXDE Documentation](https://deepxde.readthedocs.io/) for more details.  
