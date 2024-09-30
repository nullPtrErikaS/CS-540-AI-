
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)

    # - mean of each col 4 center
    x_centered = x - np.mean(x, axis = 0)
    return x_centered
    # raise NotImplementedError

def get_covariance(dataset):
    # Your implementation goes here!
    # calc covariance matrix
    covariance_matrix = np.dot(dataset.T, dataset) / (dataset.shape[0] - 1)
    return covariance_matrix
    # raise NotImplementedError

def get_eig(S, k):
    # Your implementation goes here!
     # Eigendecomposition
    eigvals, eigvecs = eigh(S)

    # Sort eigenval & eigenvec in descending order
    sorted_indices = np.argsort(eigvals)[::-1]  # Sort indices by eigenval
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Select top k eigenval & eigenvec
    top_k_eigvals = eigvals_sorted[:k]   # Select top k eigenval
    top_k_eigvecs = eigvecs_sorted[:, :k]  # Select top k eigenvec

    # Diagonal matrix of top k eigenval
    lambda_matrix = np.diag(top_k_eigvals)

    # Return diagonal matrix of top k eigenval & eigenvec
    return lambda_matrix, top_k_eigvecs
    # raise NotImplementedError

def get_eig_prop(S, prop):
    # Your implementation goes here!
    # Eigendecomposition on S
    eigvals, eigvecs = eigh(S)
    # Sort descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]
    # Compute cum var ratio
    total_variance = np.sum(eigvals_sorted)
     # Select eigenval & eigenvec > prop
    selected_indices = [i for i, val in enumerate(eigvals_sorted) if val / total_variance > prop]
    
    # Diagonal matrix for eigenval
    selected_eigvals = eigvals_sorted[selected_indices]
    lambda_matrix = np.diag(selected_eigvals)

    # Select eigenvec
    U = eigvecs_sorted[:, selected_indices]
    
    return lambda_matrix, U
    # raise NotImplementedError

def project_image(image, U):
    # Your implementation goes here!
    # Project image onto subspace spanned by the eigenvecs in U
    return np.dot(U, np.dot(U.T, image))
    # raise NotImplementedError

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # Reshape images to 64x64
    orig_image = orig.reshape(64, 64)
    proj_image = proj.reshape(64, 64)
    # Create plot w 2 subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    # Display original image
    img1 = ax1.imshow(orig_image, aspect='equal')
    ax1.set_title("Original")
    fig.colorbar(img1, ax=ax1)
    # Display projected image
    img2 = ax2.imshow(proj_image, aspect='equal')
    ax2.set_title("Projection")
    fig.colorbar(img2, ax=ax2)
    return fig, ax1, ax2
    # raise NotImplementedError

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    # Project image onto the PCA subspace
    alpha = np.dot(U.T, image)
    # Perturb weights w Gaussian noise
    perturbed_alpha = alpha + np.random.normal(0, sigma, alpha.shape)
    # Reconstruct image from perturbed weights
    return np.dot(U, perturbed_alpha)
    # raise NotImplementedError

def combine_image(image1, image2, U, lam):
    # Your implementation goes here!
    # Compute PCA weights
    alpha1 = np.dot(U.T, image1)
    alpha2 = np.dot(U.T, image2)
    # Create convex combination of weights
    combined_alpha = lam * alpha1 + (1 - lam) * alpha2
    # Reconstruct image from combined weights
    return np.dot(U, combined_alpha)
    # raise NotImplementedError
