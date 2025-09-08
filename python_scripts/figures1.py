import numpy as np
import matplotlib.pyplot as plt
import os

# --- Définir le chemin d'enregistrement des fichiers ---
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
output_folder = os.path.join(desktop_path, 'Figures_Projet_Recherche')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# --- Fonctions pour les nouveaux graphiques ---

# Fonction de perte simple (convexe)
def convex_loss(w):
    return (w - 2)**2

def gradient_convex_loss(w):
    return 2 * (w - 2)

# Fonction de perte non-convexe 2D (utilisée pour les figures SDE)
def loss_function_2d(w1, w2):
    return (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2

def gradient_loss_2d(w):
    grad_w1 = 4 * w[0] * (w[0]**2 + w[1] - 11) + 2 * (w[0] + w[1]**2 - 7)
    grad_w2 = 2 * (w[0]**2 + w[1] - 11) + 4 * w[1] * (w[0] + w[1]**2 - 7)
    return np.array([grad_w1, grad_w2])

# --- Figure 1: GD vs SGD ---
def simulate_gd(w0, eta, num_steps):
    weights = [w0]
    w = w0
    for _ in range(num_steps):
        w = w - eta * gradient_convex_loss(w)
        weights.append(w)
    return np.array(weights)

def simulate_sgd(w0, eta, num_steps, sigma):
    weights = [w0]
    w = w0
    for _ in range(num_steps):
        w = w - eta * (gradient_convex_loss(w) + np.random.normal(0, sigma))
        weights.append(w)
    return np.array(weights)

w_vals = np.linspace(-1, 5, 400)
V_vals = convex_loss(w_vals)

gd_trajectory = simulate_gd(w0=4.5, eta=0.1, num_steps=20)
sgd_trajectory = simulate_sgd(w0=4.5, eta=0.1, num_steps=100, sigma=0.8)

plt.figure(figsize=(8, 6))
plt.plot(w_vals, V_vals, label="Fonction de perte")
plt.plot(gd_trajectory, convex_loss(gd_trajectory), 'o-', color='blue', label="Trajectoire GD")
plt.plot(sgd_trajectory, convex_loss(sgd_trajectory), 'x-', color='red', alpha=0.6, label="Trajectoire SGD")
plt.title('GD vs SGD sur un paysage de perte convexe')
plt.xlabel('Poids ($w$)')
plt.ylabel('Perte ($V(w)$)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'gd_vs_sgd.pdf'))
plt.close()


# --- Figure 2: SGD vs SDE Trajectories (non-convexe) ---
# Note: Nous utilisons le même code que pour la section 6, mais nous traçons une seule trajectoire
# pour illustrer le concept de manière plus claire.

w0_2d = np.array([0.0, 0.0])
eta_2d = 0.01
num_steps_2d = 50000
sigma_2d = 0.5
weights_trajectory_sgd = simulate_sgd_2d(w0_2d, eta_2d, num_steps_2d, sigma_2d)

# Tracé du paysage de perte 2D
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = loss_function_2d(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
plt.plot(weights_trajectory_sgd[:, 0], weights_trajectory_sgd[:, 1], 'r-', alpha=0.5, label='Trajectoire de la SGD (EDS)')
plt.scatter(weights_trajectory_sgd[-1, 0], weights_trajectory_sgd[-1, 1], c='red', s=50, label='Point final')
plt.title('Analogie entre SGD et EDS de Langevin')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'sgd_vs_sde_trajectories.pdf'))
plt.close()


# --- Figure 3: Impact du bruit (bas et haut) ---
# Simulation avec un faible bruit
weights_low_noise = simulate_sgd_2d(w0_2d, eta_2d, num_steps_2d, sigma=0.1)

# Simulation avec un bruit élevé
weights_high_noise = simulate_sgd_2d(w0_2d, eta_2d, num_steps_2d, sigma=1.5)

plt.figure(figsize=(12, 6))

# Sous-figure 1: Faible bruit
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
plt.plot(weights_low_noise[:, 0], weights_low_noise[:, 1], 'r-', alpha=0.5)
plt.scatter(weights_low_noise[-1, 0], weights_low_noise[-1, 1], c='red', s=50)
plt.title('Faible bruit')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.grid(True)

# Sous-figure 2: Bruit élevé
plt.subplot(1, 2, 2)
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
plt.plot(weights_high_noise[:, 0], weights_high_noise[:, 1], 'r-', alpha=0.5)
plt.scatter(weights_high_noise[-1, 0], weights_high_noise[-1, 1], c='red', s=50)
plt.title('Bruit élevé')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'noise_impact.pdf'))
plt.close()

print(f"Les nouvelles figures ont été enregistrées dans le dossier : {output_folder}")
