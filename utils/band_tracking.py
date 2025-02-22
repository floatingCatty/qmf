import numpy as np
from scipy.optimize import linear_sum_assignment

def track_bands_3d(eigvecs, eigvals, tol=1e-8, degenerate_tol=1e-8):
    """
    Tracks band indices over a 3D k-grid by matching eigenvector overlaps,
    including handling degenerate (or nearly degenerate) bands.

    Parameters:
        eigvecs: np.ndarray
            Eigenvectors with shape (nkx, nky, nkz, nband, norb).
        eigvals: np.ndarray
            Eigenvalues with shape (nkx, nky, nkz, nband).
        tol: float
            Tolerance for phase adjustment.
        degenerate_tol: float
            Energy difference threshold for identifying degenerate bands.
    
    Returns:
        tracked_eigvecs: np.ndarray
            Eigenvectors with band indices reordered to ensure continuity,
            same shape as eigvecs.
        tracked_eigvals: np.ndarray
            Corresponding eigenvalues with shape (nkx, nky, nkz, nband).
    """
    nkx, nky, nkz, nband, norb = eigvecs.shape
    tracked_eigvecs = np.empty_like(eigvecs)
    tracked_eigvals = np.empty_like(eigvals)
    
    # --- Initialization at the first k-point ---
    # We choose (0,0,0) as the starting point and sort its bands by energy.
    order0 = np.argsort(eigvals[0, 0, 0])
    tracked_eigvecs[0, 0, 0] = eigvecs[0, 0, 0][order0]
    tracked_eigvals[0, 0, 0] = eigvals[0, 0, 0][order0]
    
    # --- Traverse the 3D grid in lexicographical order ---
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                # Skip the starting point.
                if (ix, iy, iz) == (0, 0, 0):
                    continue

                # --- Choose a reference k-point ---
                # Prefer the immediate neighbor in the negative z direction,
                # if not available then negative y (using the last z point),
                # or negative x (using the last y and z points).
                if iz > 0:
                    ref = (ix, iy, iz - 1)
                elif iy > 0:
                    ref = (ix, iy - 1, nkz - 1)
                elif ix > 0:
                    ref = (ix - 1, nky - 1, nkz - 1)
                else:
                    ref = (ix, iy, iz)  # should not occur
                
                prev_vecs = tracked_eigvecs[ref]      # shape: (nband, norb)
                current_vecs = eigvecs[ix, iy, iz]      # shape: (nband, norb)
                
                # --- Compute the overlap matrix ---
                # S[m,n] = |<prev_vec_m | current_vec_n>|
                S = np.abs(np.dot(prev_vecs, current_vecs.conj().T))
                
                # --- Determine the optimal matching ---
                cost = -S  # maximize total overlap
                row_ind, col_ind = linear_sum_assignment(cost)
                # Reorder current eigenvectors and eigenvalues accordingly.
                reordered_vecs = current_vecs[col_ind].copy()
                reordered_vals = eigvals[ix, iy, iz, col_ind].copy()
                
                # --- Identify degenerate blocks in the reference ---
                # Group indices whose eigenvalue differences are below degenerate_tol.
                prev_vals = tracked_eigvals[ref]  # shape: (nband,)
                blocks = []    # list of lists for indices forming a degenerate block
                current_block = [0]
                for i in range(1, nband):
                    if np.abs(prev_vals[i] - prev_vals[i-1]) < degenerate_tol:
                        current_block.append(i)
                    else:
                        blocks.append(current_block)
                        current_block = [i]
                blocks.append(current_block)
                
                # --- Adjust degenerate blocks using subspace alignment ---
                for block in blocks:
                    if len(block) > 1:
                        b = np.array(block)
                        prev_block = prev_vecs[b]          # shape: (d, norb)
                        current_block = reordered_vecs[b]  # shape: (d, norb)
                        # Overlap within the degenerate subspace.
                        S_block = np.dot(prev_block, current_block.conj().T)  # (d, d)
                        U, sigma, Vh = np.linalg.svd(S_block)
                        # Compute the rotation R = Vh^† U^† that aligns the block.
                        R = np.dot(Vh.conj().T, U.conj().T)
                        reordered_vecs[b] = np.dot(R, current_block)
                        # The eigenvalues remain unchanged.
                
                # --- Adjust phases for all bands ---
                for band in range(nband):
                    phase = np.vdot(prev_vecs[band], reordered_vecs[band])
                    if np.abs(phase) > tol:
                        reordered_vecs[band] *= np.conj(phase) / np.abs(phase)
                
                # --- Store the results ---
                tracked_eigvecs[ix, iy, iz] = reordered_vecs
                tracked_eigvals[ix, iy, iz] = reordered_vals

    return tracked_eigvecs, tracked_eigvals

# Example usage:
# eigvecs_3d has shape (nkx, nky, nkz, nband, norb)
# eigvals_3d has shape (nkx, nky, nkz, nband)
# tracked_vecs, tracked_vals = track_bands_3d(eigvecs_3d, eigvals_3d)


# Example usage:
# Suppose `evecs` has shape (nkx, nky, nkz, nband, norb)
# and `evals` has shape (nkx, nky, nkz, nband)
# tracked_vecs, tracked_vals = track_bands(evecs, evals)

def track_bands_1d(eigvecs, eigvals, tol=1e-4, degenerate_tol=1e-4):
    """
    Tracks band indices along a 1D k-path using overlap matching,
    including explicit handling of degenerate bands.

    Parameters:
        eigvecs: np.ndarray, shape (nk, nband, norb)
            Eigenvectors along the k-path.
        eigvals: np.ndarray, shape (nk, nband)
            Eigenvalues along the k-path.
        tol: float
            Tolerance for phase adjustment.
        degenerate_tol: float
            Energy difference threshold for identifying degenerate bands.
    
    Returns:
        tracked_eigvecs: np.ndarray, shape (nk, nband, norb)
            Eigenvectors with band indices reordered to ensure continuity.
        tracked_eigvals: np.ndarray, shape (nk, nband)
            Reordered eigenvalues.
    """
    nk, nband, norb = eigvecs.shape
    tracked_eigvecs = np.empty_like(eigvecs)
    tracked_eigvals = np.empty_like(eigvals)
    
    # For the first k-point, sort bands by energy.
    order0 = np.argsort(eigvals[0])
    tracked_eigvecs[0] = eigvecs[0][order0]
    tracked_eigvals[0] = eigvals[0][order0]
    
    # Loop over subsequent k-points.
    for k in range(1, nk):
        prev_vecs = tracked_eigvecs[k-1]  # shape: (nband, norb)
        current_vecs = eigvecs[k]          # shape: (nband, norb)
        
        # Compute overlap matrix: S[m, n] = |<prev_vec_m | current_vec_n>|
        S = np.abs(np.dot(prev_vecs, current_vecs.conj().T))  # shape: (nband, nband)
        
        # Use the Hungarian algorithm to determine optimal matching.
        cost = -S  # maximize total overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Reorder current eigenvectors and eigenvalues according to the matching.
        reordered_vecs = current_vecs[col_ind].copy()
        reordered_vals = eigvals[k][col_ind].copy()
        
        # --- Identify degenerate blocks in the previous k-point ---
        # Here we group bands whose eigenvalue differences (from the previous k-point)
        # are less than degenerate_tol.
        prev_vals = tracked_eigvals[k-1]
        blocks = []    # List of lists, each sub-list is a block of degenerate band indices.
        current_block = [0]
        for i in range(1, nband):
            if np.abs(prev_vals[i] - prev_vals[i-1]) < degenerate_tol:
                current_block.append(i)
            else:
                blocks.append(current_block)
                current_block = [i]
        blocks.append(current_block)
        
        # For each degenerate block (with more than one band), adjust the current block.
        for block in blocks:
            if len(block) > 1:
                b = np.array(block)
                prev_block = prev_vecs[b]         # shape: (d, norb)
                current_block = reordered_vecs[b]   # shape: (d, norb)
                # Compute the overlap matrix within the degenerate subspace.
                S_block = np.dot(prev_block, current_block.conj().T)  # shape: (d, d)
                # SVD of the overlap matrix.
                U, sigma, Vh = np.linalg.svd(S_block)
                # Determine the rotation that best aligns the current block with the previous one.
                R = np.dot(Vh.conj().T, U.conj().T)  # shape: (d, d)
                # Rotate the current block.
                reordered_vecs[b] = np.dot(R, current_block)
                # (Eigenvalues remain unchanged.)
        
        # --- Adjust the phase for each band to ensure smooth continuity ---
        for band in range(nband):
            phase = np.vdot(prev_vecs[band], reordered_vecs[band])
            if np.abs(phase) > tol:
                reordered_vecs[band] *= np.conj(phase) / np.abs(phase)
        
        # Store the tracked eigenvectors and eigenvalues.
        tracked_eigvecs[k] = reordered_vecs
        tracked_eigvals[k] = reordered_vals
        
    return tracked_eigvecs, tracked_eigvals

# Example usage:
# Assume `evecs` has shape (nk, nband, norb) and `evals` has shape (nk, nband)
# tracked_vecs, tracked_vals = track_bands_1d(evecs, evals)
