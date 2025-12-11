import numpy as np
from Bio.PDB import MMCIFParser
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # NEW: For visualization
import matplotlib.pyplot as plt # NEW: For plotting
import os
import random
import sys
from pathlib import Path

# --- 1. CONFIGURATION AND FILE SETUP ---

# Path to the uploaded mmCIF file (PDBx format)
# FIX: Use pathlib to construct the path relative to the script location
# The file is expected to be named '4YKI.cif' and placed in the same directory.
CIF_FILEPATH = Path(__file__).parent / "4YKI.cif"

# The target ligand we are extracting coordinates for (Glycine)
TARGET_LIGAND_ID = "GLY"

# The number of simulated structural conformations (necessary for clustering)
N_CONFORMATIONS = 100 

# The desired number of structural clusters (K for GMM/DISCA)
K_CLUSTERS = 3 

# --- 2. CORE UTILITY FUNCTIONS ---

def get_glycine_coordinates(filepath, ligand_id="GLY"):
    """
    Parses the mmCIF file and returns the coordinates of the target ligand (Glycine)
    and surrounding atoms.
    """
    # Convert Path object to string for os.path.exists check
    filepath_str = str(filepath) 
    
    if not os.path.exists(filepath_str):
        print(f"\nFATAL ERROR: File '{filepath_str}' not found.")
        print("Please ensure the unzipped file '4YKI.cif' is in the exact same directory as the Python script.")
        sys.exit(1)
        
    try:
        # FIX: Open the file explicitly with UTF-8 encoding to prevent 'charmap' codec error.
        with open(filepath_str, 'r', encoding='utf-8') as handle:
            parser = MMCIFParser()
            # Pass the open file handle to the parser.
            structure = parser.get_structure("GLU_IGLUR", handle)
        
        ligand_coords = []
        is_ligand_present = False
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check for the specified ligand ID (GLY in the HETATM section)
                    if residue.get_resname() == ligand_id and residue.get_full_id()[3][0] != ' ':
                        is_ligand_present = True
                        for atom in residue:
                            ligand_coords.append(atom.get_coord().tolist())

        if not ligand_coords:
            # Fallback: Extract the first few Glycine residues from the protein backbone
            print(f"Warning: Ligand '{ligand_id}' not found in HETATM records. Falling back to protein backbone GLY residue 1.")
            
            # Re-read the structure to find the GLY backbone residue (assuming GLY is residue 1 of chain A)
            with open(filepath_str, 'r', encoding='utf-8') as handle:
                structure = parser.get_structure("GLU_IGLUR", handle)
                
            for model in structure:
                chain_A = model['A'] # Assuming chain A
                # Try to find the first GLY backbone residue (Seq ID 1 in chain A)
                try:
                    residue = chain_A[(' ', 1, ' ')] 
                    if residue.get_resname() == "GLY":
                        for atom in residue:
                            ligand_coords.append(atom.get_coord().tolist())
                        print(f"  - Successfully extracted backbone GLY (ResID 1) as fragment source.")
                        break
                except KeyError:
                    # If residue 1 doesn't exist or isn't GLY, continue to search
                    for residue in chain_A:
                        if residue.get_resname() == "GLY":
                            for atom in residue:
                                ligand_coords.append(atom.get_coord().tolist())
                            print(f"  - Successfully extracted first backbone GLY ({residue.id[1]}) as fragment source.")
                            break
                    
        if not ligand_coords:
            raise ValueError(f"No suitable '{ligand_id}' coordinates found in ligand or backbone records.")
            
        return np.array(ligand_coords)

    except Exception as e:
        print(f"Error during CIF file parsing with BioPython: {e}")
        # Returning an empty array signals a failure
        return np.array([]) 

def simulate_cryoem_ensemble(coords_template, N):
    """
    Simulates N noisy conformational states (subtomograms) based on a high-res template.
    This mimics the input data for DISCA.
    """
    ensemble = []
    # Add random noise (simulating conformational flexibility and Cryo-EM noise)
    noise_level = 0.5 
    for _ in range(N):
        # Apply slight random translation (will be filtered by YOPO)
        translation = np.random.uniform(-1, 1, 3) * 0.1
        # Apply rotational noise
        rotation_matrix = _get_random_rotation_matrix(0.1)
        
        noisy_coords = (coords_template + translation) @ rotation_matrix + \
                       np.random.normal(0, noise_level, coords_template.shape)
        ensemble.append(noisy_coords)
    return ensemble

def _get_random_rotation_matrix(angle_scale):
    """Generates a small random rotation matrix."""
    theta = np.random.uniform(-angle_scale, angle_scale)
    c, s = np.cos(theta), np.sin(theta)
    # Use a full 3D rotation for better simulation
    # Simple axis-angle rotation for the prototype:
    a = np.random.rand(3)
    a = a / np.linalg.norm(a)
    R = np.identity(3) + np.sin(theta) * np.array([[0, -a[2], a[1]], 
                                                  [a[2], 0, -a[0]], 
                                                  [-a[1], a[0], 0]]) + \
          (1 - np.cos(theta)) * np.array([
              [a[0]**2 - 1, a[0]*a[1], a[0]*a[2]],
              [a[1]*a[0], a[1]**2 - 1, a[1]*a[2]],
              [a[2]*a[0], a[2]*a[1], a[2]**2 - 1]
          ])
    return R

# --- 3. YOPO-INSPIRED FEATURE EXTRACTION (The ML Component) ---

def yopo_feature_extraction(conformation):
    """
    Conceptual implementation of the YOPO feature extraction pipeline (Steps 2 & 3).
    """
    
    if conformation.size == 0 or conformation.shape[0] < 2:
        return np.zeros(6)

    # 1. Inter-atomic Distances (Inherent Translation/Rotation Invariance)
    diff = conformation[:, np.newaxis, :] - conformation[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    # Extract Max, Min, Mean distances (Simulating the y_g Global Max-Pooling layer)
    max_dist = np.max(distances)
    min_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 0
    mean_dist = np.mean(distances)

    # 2. Distance Variance (Simulating local complexity/yc convolutions)
    std_dist = np.std(distances)
    
    # 3. Simple shape metric (Simulating y_f fully connected layer output)
    center_of_mass = np.mean(conformation, axis=0)
    R_g_sq = np.mean(np.sum((conformation - center_of_mass)**2, axis=1))

    # 4. Aspect Ratio (Simulating global shape)
    # Calculate moments of inertia (not true tensor, just simple variance)
    var_x = np.var(conformation[:, 0])
    var_y = np.var(conformation[:, 1])
    aspect_ratio = var_x / (var_y + 1e-6)
    
    # The final P-dimensional feature vector
    F = np.array([max_dist, min_dist, mean_dist, std_dist, R_g_sq, aspect_ratio])
    
    return F

# --- 4. DISCA CLUSTERING (GMM - The Statistical Model) ---

def disca_clustering_gmm(feature_matrix, K):
    """
    Performs DISCA's core statistical modeling using a Gaussian Mixture Model (GMM).
    This implements the probabilistic clustering of Step 4.
    """
    
    # Check if there are enough samples and features to run GMM
    if feature_matrix.shape[0] < K or feature_matrix.shape[1] == 0:
        print("Warning: Insufficient data to run GMM. Returning None.")
        return None, None
        
    # 1. Normalize features (Crucial for GMM convergence)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # 2. Initialize and fit GMM
    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(features_scaled)
    
    # 3. Get Posterior Probabilities (rho_k(x_n))
    posterior_probs = gmm.predict_proba(features_scaled)
    
    # 4. Assign Cluster Labels (hat{k} = argmax(rho_k))
    labels = np.argmax(posterior_probs, axis=1)
    
    # 5. Calculate Cluster Purity (Simulating the output of the iterative labeling)
    cluster_indices = np.unique(labels)
    cluster_purity = {}
    
    for k in cluster_indices:
        # Purity is represented by the mean max posterior probability within the cluster
        indices = np.where(labels == k)[0]
        if len(indices) > 0:
            mean_max_prob = np.mean(posterior_probs[indices, k])
            cluster_purity[k] = mean_max_prob
    
    # Select the "purest" cluster for quantum simulation (Fragment Selection)
    if cluster_purity:
        best_cluster = max(cluster_purity, key=cluster_purity.get)
        print(f"\n[DISCA/GMM Result] Selected Cluster {best_cluster} for Fragmentation.")
        print(f"--- Cluster Purity (Mean Max Posterior): {cluster_purity[best_cluster]:.4f}")
        return best_cluster, labels, features_scaled # Pass features_scaled for plotting
    
    return None, labels, features_scaled # Pass features_scaled for plotting


# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    
    print(f"--- STAGE I: CLASSICAL PRE-PROCESSING ---")
    print(f"Target: PDB ID 4YKI (mmCIF) - Glycine Fragment")
    
    # Step 1: Data Input
    # Check for local file existence before proceeding
    if not Path(CIF_FILEPATH).exists():
        print(f"\nFATAL ERROR: File '{CIF_FILEPATH}' not found.")
        print("Please ensure the unzipped file '4YKI.cif' is in the exact same directory as the Python script.")
        sys.exit(1)

    template_coords = get_glycine_coordinates(CIF_FILEPATH, TARGET_LIGAND_ID)
    
    if template_coords.size == 0:
        print("FATAL ERROR: Could not extract useful coordinates from the CIF file.")
        sys.exit(1)

    print(f"[Step 1] Successfully extracted Glycine coordinates ({template_coords.shape[0]} atoms).")

    # Simulate ensemble (mimicking Cryo-EM subtomograms)
    cryoem_ensemble = simulate_cryoem_ensemble(template_coords, N_CONFORMATIONS)
    print(f"[Input] Simulated {N_CONFORMATIONS} noisy conformations for clustering.")

    # Initialize the Feature Matrix
    Feature_Matrix = []

    # Steps 2 & 3: Deep Feature Extraction (YOPO) and Rotation Invariance
    print(f"\n[Steps 2 & 3] Running YOPO Feature Extraction...")
    for i, coords in enumerate(cryoem_ensemble):
        features = yopo_feature_extraction(coords)
        Feature_Matrix.append(features)

    Feature_Matrix = np.array(Feature_Matrix)
    print(f"[YOPO Output] Generated feature matrix of shape: {Feature_Matrix.shape}")

    # Steps 4 & 5: DISCA Clustering (GMM) and Iterative Dynamic Labeling (Simulated by Purity)
    print(f"\n[Step 4] Running DISCA Clustering (GMM) with K={K_CLUSTERS}...")
    best_cluster_id, cluster_labels, features_scaled = disca_clustering_gmm(Feature_Matrix, K_CLUSTERS)

    
    # --- VISUALIZATION: Plotting the Clusters ---
    if cluster_labels is not None:
        print("\n--- VISUALIZATION: Generating Cluster Plot ---")
        
        # 1. Reduce features (which are 6-dimensional) to 2D using PCA for plotting
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            marker='o',
            alpha=0.7,
            edgecolors='k'
        )
        
        # Add Cluster Centers (Mean of each Gaussian)
        if best_cluster_id is not None:
            gmm = GaussianMixture(n_components=K_CLUSTERS, covariance_type='full', random_state=42)
            gmm.fit(features_scaled) # Re-fit to get the center locations
            centers_2d = pca.transform(gmm.means_)
            
            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                marker='X',
                s=200,
                color='red',
                label='Cluster Centers',
                edgecolors='k'
            )
            
            # Highlight the selected cluster (best_cluster_id)
            selected_center_2d = centers_2d[best_cluster_id]
            plt.scatter(
                selected_center_2d[0],
                selected_center_2d[1],
                marker='*',
                s=500,
                color='yellow',
                label=f'Selected Fragment Source (Cluster {best_cluster_id})',
                edgecolors='k',
                zorder=10
            )

        plt.title('DISCA/GMM Clustering of Conformations (Reduced to 2D by PCA)', fontsize=14)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.colorbar(scatter, label='Cluster ID')
        plt.show()

    # Step 6: Fragmentation (Selecting the Homogeneous Fragment)
    if best_cluster_id is not None and cluster_labels is not None:
        # Get all indices belonging to the best cluster
        homogeneous_indices = np.where(cluster_labels == best_cluster_id)[0]
        
        if len(homogeneous_indices) > 0:
            # Select a representative conformation from this cluster
            representative_index = homogeneous_indices[random.randint(0, len(homogeneous_indices) - 1)]
            
            final_fragment_coords = cryoem_ensemble[representative_index]
            
            print("\n--- FRAGMENTATION COMPLETE ---")
            print(f"[Step 6 Output] Final Homogeneous Fragment Selected:")
            print(f"  * Source Conformation Index: {representative_index}")
            print(f"  * Atoms in Fragment: {final_fragment_coords.shape[0]}")
            print(f"  * Status: READY for Hamiltonian Encoding (Stage II).")
        else:
            print("\n--- FRAGMENTATION FAILED ---")
            print("The selected cluster was empty. Cannot proceed.")
    else:
        print("\n--- FRAGMENTATION FAILED ---")
        print("Clustering did not return a valid result.")