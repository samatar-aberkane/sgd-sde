import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- Définir le chemin d'enregistrement des fichiers ---
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
output_folder = os.path.join(desktop_path, 'Figures_Projet_Recherche')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Fonctions pour les nouveaux graphiques ---

# Fonction de perte simple (pour le gradient)
def simple_loss(w1, w2):
    return (w1 - 1)**2 + (w2 - 2)**2

def simple_grad(w1, w2):
    return np.array([2 * (w1 - 1), 2 * (w2 - 2)])

# Fonction pour le paysage non-convexe
def non_convex_loss(w1, w2):
    return 0.1 * (w1**2 + w2**2) + np.sin(5 * w1) + np.sin(5 * w2) + 0.5

def generate_noise_vectors(num_points, scale=1.0):
    return np.random.normal(0, scale, (num_points, 2))

# --- Figure 1: Comparaison des gradients ---
def create_gradient_comparison():
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(-1, 3, 400)
    y_range = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = simple_loss(X, Y)

    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Perte')

    # Gradient exact (pointant vers le minimum global)
    plt.arrow(1, 2, -1, -2, color='blue', width=0.05, head_width=0.2, length_includes_head=True, label='Gradient exact')
    
    # Gradient stochastique
    start_point = np.array([2.5, 2.5])
    true_grad = simple_grad(start_point[0], start_point[1])
    
    noise_vectors = generate_noise_vectors(5, scale=1.5)
    for i, noise in enumerate(noise_vectors):
        stochastic_grad = true_grad + noise
        label = 'Gradient stochastique' if i == 0 else ""
        plt.arrow(start_point[0], start_point[1], -stochastic_grad[0]*0.1, -stochastic_grad[1]*0.1, color='red', width=0.05, head_width=0.2, length_includes_head=True, alpha=0.6, label=label)
    
    plt.title('Comparaison du gradient exact vs stochastique')
    plt.xlabel('Paramètre $\\theta_1$')
    plt.ylabel('Paramètre $\\theta_2$')
    plt.scatter(1, 2, color='red', s=100, label='Minimum global')
    plt.scatter(start_point[0], start_point[1], color='black', s=50, zorder=5)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'gradient_comparison.pdf'))
    plt.close()


# --- Figure 2: Evasion d'un point de selle ---
def create_saddle_point_evasion():
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 - Y**2 + 0.1 * (X**4 + Y**4)
    
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Perte')
    
    # Point de selle
    plt.scatter(0, 0, color='red', marker='x', s=200, label='Point de selle (S)', zorder=5)
    
    # Trajectoire GD (piégée)
    gd_steps = 10
    gd_trajectory = np.zeros((gd_steps, 2))
    gd_trajectory[0] = [-1.5, -0.1]
    for i in range(1, gd_steps):
        grad = np.array([2*gd_trajectory[i-1, 0] + 0.4*gd_trajectory[i-1, 0]**3, -2*gd_trajectory[i-1, 1] + 0.4*gd_trajectory[i-1, 1]**3])
        gd_trajectory[i] = gd_trajectory[i-1] - 0.1 * grad
    plt.plot(gd_trajectory[:, 0], gd_trajectory[:, 1], 'o-', color='blue', alpha=0.8, label='Trajectoire GD')
    
    # Trajectoire SGD (s'échappant)
    sgd_steps = 20
    sgd_trajectory = np.zeros((sgd_steps, 2))
    sgd_trajectory[0] = [-1.5, -0.1]
    for i in range(1, sgd_steps):
        grad = np.array([2*sgd_trajectory[i-1, 0] + 0.4*sgd_trajectory[i-1, 0]**3, -2*sgd_trajectory[i-1, 1] + 0.4*sgd_trajectory[i-1, 1]**3])
        noise = np.random.normal(0, 0.5, 2)
        sgd_trajectory[i] = sgd_trajectory[i-1] - 0.1 * (grad + noise)
    plt.plot(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 'o-', color='red', alpha=0.8, label='Trajectoire SGD')

    plt.title('Évasion d\'un point de selle par la SGD')
    plt.xlabel('Paramètre $\\theta_1$')
    plt.ylabel('Paramètre $\\theta_2$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'non_convex_saddle.pdf'))
    plt.close()


# --- Figure 3: Minima plats vs. minima pointus ---
def create_flat_sharp_minima():
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(-15, 15, 400)
    
    # Définition des minima
    flat_min_x = np.linspace(-15, 0, 200)
    sharp_min_x = np.linspace(0, 15, 200)
    
    flat_min_y = 0.05 * (flat_min_x + 7)**4 + 2
    sharp_min_y = 1.5 * (sharp_min_x - 10)**2 + 10
    
    plt.plot(flat_min_x, flat_min_y, color='black')
    plt.plot(sharp_min_x, sharp_min_y, color='black', label='Paysage de perte')

    # Représentation des minima
    plt.scatter(-7, 2, s=100, c='blue', label='Minimum Plat')
    plt.scatter(10, 10, s=100, c='red', label='Minimum Pointu')

    # Illustration de la dynamique de la SGD
    # Fluctuation autour du minimum plat
    noise_x_flat = np.random.normal(-7, 1.5, 50)
    noise_y_flat = 0.05 * (noise_x_flat + 7)**4 + 2
    plt.plot(noise_x_flat, noise_y_flat, 'o', c='blue', alpha=0.4)

    # Fluctuation autour du minimum pointu
    noise_x_sharp = np.random.normal(10, 0.3, 50)
    noise_y_sharp = 1.5 * (noise_x_sharp - 10)**2 + 10
    plt.plot(noise_x_sharp, noise_y_sharp, 'o', c='red', alpha=0.4)
    
    plt.title('Minima plats vs. minima pointus')
    plt.xlabel('Paramètres du modèle')
    plt.ylabel('Fonction de perte')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'minima_flat_vs_sharp.pdf'))
    plt.close()


# --- Exécution des fonctions de création de figures ---
create_gradient_comparison()
create_saddle_point_evasion()
create_flat_sharp_minima()

print(f"Les nouvelles figures ont été enregistrées dans le dossier : {output_folder}")
