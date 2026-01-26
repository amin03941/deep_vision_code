import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "bio_memory_dump.pkl"
OUTPUT_FILE = "health_direction_vector.npy"
METADATA_FILE = "vector_metadata.pkl"

def generate_enhanced_health_vector():
    """
    Enhanced version with:
    1. Multi-cytokine analysis (not just TNFA)
    2. PCA-based direction finding
    3. Validation metrics
    4. Metadata export for dashboard
    """
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    # --- MULTI-BIOMARKER APPROACH ---
    # Use composite inflammatory score (TNFA + IL22 - EGF)
    # Higher score = more inflamed
    
    biomarker_scores = []
    vectors = []
    
    for item in data:
        payload = item['payload']
        # Composite inflammatory index
        score = (
            payload.get('TNFA', 0) + 
            payload.get('IL22', 0) * 0.5 -  # IL22 also inflammatory
            payload.get('EGF', 0) * 0.3      # EGF is protective
        )
        biomarker_scores.append(score)
        vectors.append(item['vector']['dense'])
    
    scores_array = np.array(biomarker_scores)
    vectors_array = np.array(vectors)
    
    # Calculate thresholds (top/bottom 20% for cleaner separation)
    high_threshold = np.percentile(scores_array, 80)
    low_threshold = np.percentile(scores_array, 20)
    
    print(f"📊 Inflammatory Index Thresholds: Low < {low_threshold:.2f} | High > {high_threshold:.2f}")

    # Stratify samples
    healthy_mask = scores_array <= low_threshold
    disease_mask = scores_array >= high_threshold
    
    healthy_vecs = vectors_array[healthy_mask]
    disease_vecs = vectors_array[disease_mask]
    
    print(f"   - Identified {len(healthy_vecs)} 'Healthy' samples")
    print(f"   - Identified {len(disease_vecs)} 'Disease' samples")

    if len(healthy_vecs) < 5 or len(disease_vecs) < 5:
        print("⚠️ Warning: Insufficient data. Using random vector.")
        diff_vector = np.random.rand(768).astype(np.float32)
        metadata = {"method": "random_fallback", "n_healthy": len(healthy_vecs), "n_disease": len(disease_vecs)}
    else:
        # --- METHOD 1: Simple Centroid (baseline) ---
        centroid_healthy = np.mean(healthy_vecs, axis=0)
        centroid_disease = np.mean(disease_vecs, axis=0)
        diff_centroid = centroid_healthy - centroid_disease
        
        # --- METHOD 2: PCA-Enhanced (captures variance direction) ---
        # Combine groups and apply PCA to find the discriminant axis
        combined = np.vstack([healthy_vecs, disease_vecs])
        labels = np.concatenate([np.zeros(len(healthy_vecs)), np.ones(len(disease_vecs))])
        
        # Standardize
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        
        # Find principal component that best separates groups
        pca = PCA(n_components=10)
        pca_transformed = pca.fit_transform(combined_scaled)
        
        # Test each PC for separation power
        best_pc_idx = 0
        best_separation = 0
        
        for i in range(min(5, pca.n_components_)):
            pc_vals = pca_transformed[:, i]
            mean_healthy = pc_vals[labels == 0].mean()
            mean_disease = pc_vals[labels == 1].mean()
            separation = abs(mean_healthy - mean_disease)
            
            if separation > best_separation:
                best_separation = separation
                best_pc_idx = i
        
        print(f"   - Best discriminant PC: {best_pc_idx} (separation: {best_separation:.2f})")
        
        # Get direction vector (in original space)
        diff_pca = pca.components_[best_pc_idx]
        diff_pca_unscaled = scaler.inverse_transform(diff_pca.reshape(1, -1)).flatten()
        
        # Ensure direction points from disease to healthy
        if np.dot(diff_pca_unscaled, diff_centroid) < 0:
            diff_pca_unscaled = -diff_pca_unscaled
        
        # --- BLEND BOTH METHODS (70% PCA, 30% Centroid) ---
        diff_vector = 0.7 * diff_pca_unscaled + 0.3 * diff_centroid
        
        # Normalize to unit vector (consistent magnitude)
        diff_vector = diff_vector / (np.linalg.norm(diff_vector) + 1e-8)
        
        print("✅ Calculated enhanced 'Health Direction' vector (PCA + Centroid blend).")
        
        # Validation: Calculate effectiveness score
        # Project all vectors onto this direction and check separation
        projections = vectors_array @ diff_vector
        proj_healthy = projections[healthy_mask]
        proj_disease = projections[disease_mask]
        
        effectiveness = abs(proj_healthy.mean() - proj_disease.mean()) / (proj_healthy.std() + proj_disease.std() + 1e-8)
        
        metadata = {
            "method": "pca_centroid_blend",
            "n_healthy": len(healthy_vecs),
            "n_disease": len(disease_vecs),
            "best_pc": best_pc_idx,
            "separation_score": best_separation,
            "effectiveness": effectiveness,
            "explained_variance": pca.explained_variance_ratio_[:3].tolist()
        }
        
        print(f"   - Effectiveness Score: {effectiveness:.3f} (higher = better separation)")

    # Save vector
    np.save(OUTPUT_FILE, diff_vector.astype(np.float32))
    
    # Save metadata for dashboard
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"💾 Saved steering vector to: {OUTPUT_FILE}")
    print(f"💾 Saved metadata to: {METADATA_FILE}")
    
    return diff_vector, metadata

if __name__ == "__main__":
    generate_enhanced_health_vector()