# 🧬 Stage I: DISCA-Inspired Classical Pre-Processing (4YKI)

## 🎯 Project Overview

This script, `disca_mock.py`, simulates **Stage I** of a hybrid quantum chemistry workflow. The goal of this stage is to process noisy structural data (mimicking Cryo-EM subtomograms) to identify a **homogeneous conformational fragment** suitable for subsequent quantum computation.

The process involves a simulated version of the **Deep Invariant Subtomogram Clustering and Alignment (DISCA)** pipeline, combining advanced feature extraction with Gaussian Mixture Model (GMM) clustering.

### Key Computational Steps Simulated:
1.  **Data Simulation:** Simulate an ensemble of noisy structural conformations for a molecular fragment (Glycine from PDB ID 4YKI).
2.  **YOPO Feature Extraction:** Convert 3D atomic coordinates into a rotationally/translationally invariant feature vector.
3.  **GMM Clustering:** Use Gaussian Mixture Model (GMM) to probabilistically cluster the feature vectors into conformational states ($K=3$).
4.  **Fragment Selection:** Select the fragment source from the cluster exhibiting the highest "purity" (highest mean max posterior probability) as the optimal geometry for quantum encoding.

## ⚙️ Requirements

This script requires the following standard scientific Python libraries:

* `Python 3.x`
* `numpy`
* `matplotlib` (for visualization)
* `scikit-learn` (for GMM and PCA)
* `biopython` (for reading mmCIF files)

Install the required libraries:
```bash
pip install numpy matplotlib scikit-learn biopython

📂 File Setup
For the script to run successfully, you must have the target molecular structure file in the same directory:

Structure File: 4YKI.cif (mmCIF/PDBx format)

🚀 Execution
Run the script directly from your terminal:

python disca_mock.py

Expected Output
Upon successful execution, the script will:

Print the results of the feature extraction and clustering process, including the selected fragment ID.

Display a 3D Scatter Plot of the clustered conformations (reduced via Principal Component Analysis - PCA), with cluster centers marked and the final selected fragment source highlighted.

📋 Core Simulation Details
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **PDB Source** | 4YKI | The structure file used for the template fragment. |
| **Target Fragment** | GLY (Glycine) | The residue used to generate the conformational ensemble. |
| **N_CONFORMATIONS** | 100 | The number of simulated noisy structures (subtomograms). |
| **K_CLUSTERS** | 3 | The target number of clusters for the GMM. |

YOPO Feature Vector (P=6)
The simulated YOPO Feature Extraction generates a 6-dimensional rotationally invariant feature vector for each conformation, composed of simple geometric metrics:Max Inter-atomic DistanceMin Inter-atomic DistanceMean Inter-atomic DistanceStandard Deviation of DistancesRadius of Gyration Squared ($R_g^2$)Aspect Ratio (Variance X / Variance Y)

Final Output
The script selects a representative geometry from the "purest" cluster. This final fragment is then considered READY for Hamiltonian Encoding (Stage II), where it would be converted into a Qubit Hamiltonian for quantum computation.