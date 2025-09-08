import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation d'une SGD sur un potentiel double puits ---
def double_well_potential(w):
    """Fonction de potentiel pour un puits double."""
    return w**4 - 2 * w**2

def gradient_double_well(w):
    """Gradient du potentiel double puits."""
    return 4 * w**3 - 4 * w

def simulate_sgd_double_well(w0, eta, num_steps, batch_size, sigma):
    """
    Simule la SGD sur le potentiel double puits.
    w0: poids initial
    eta: learning rate
    num_steps: nombre d'étapes de la simulation
    batch_size: taille du mini-batch
    sigma: intensité du bruit gaussien
    """
    weights = [w0]
    w = w0
    for _ in range(num_steps):
        # Bruit stochastique
        noise = np.random.normal(0, sigma)
        # Gradient de la fonction de perte
        grad_v = gradient_double_well(w) + noise
        # Mise à jour des poids (SGD)
        w = w - eta * grad_v
        weights.append(w)
    return np.array(weights)

# Paramètres de la simulation
w0 = -1.5
eta = 0.05
num_steps = 1000
sigma = 0.3
num_trajectories = 5

# Tracé du potentiel
w_vals = np.linspace(-2.5, 2.5, 400)
V_vals = double_well_potential(w_vals)

plt.figure(figsize=(8, 6))
plt.plot(w_vals, V_vals, label="Potentiel $V(w)$")

# Tracé de plusieurs trajectoires
for i in range(num_trajectories):
    trajectory = simulate_sgd_double_well(w0, eta, num_steps, 1, sigma)
    plt.plot(trajectory, double_well_potential(trajectory), '.-', alpha=0.6, label=f'Trajectoire {i+1}' if i==0 else "")

plt.title('Potentiel à double puits et trajectoires de la SGD')
plt.xlabel('Poids ($w$)')
plt.ylabel('Potentiel ($V(w)$)')
plt.grid(True)
plt.legend()
plt.savefig('double_puits_potentiel.pdf')
plt.close()

# --- 2. Visualisation de trajectoires et de distributions sur un potentiel 2D ---
def loss_function_2d(w1, w2):
    """Fonction de perte non-convexe 2D."""
    return (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2

def gradient_loss_2d(w):
    """Gradient de la fonction de perte 2D."""
    grad_w1 = 4 * w[0] * (w[0]**2 + w[1] - 11) + 2 * (w[0] + w[1]**2 - 7)
    grad_w2 = 2 * (w[0]**2 + w[1] - 11) + 4 * w[1] * (w[0] + w[1]**2 - 7)
    return np.array([grad_w1, grad_w2])

def simulate_sgd_2d(w0, eta, num_steps, sigma):
    """Simule la SGD sur un potentiel 2D."""
    weights = np.zeros((num_steps, 2))
    w = w0
    for i in range(num_steps):
        noise = np.random.normal(0, sigma, 2)
        grad_v = gradient_loss_2d(w) + noise
        w = w - eta * grad_v
        weights[i] = w
    return weights

# Paramètres
w0_2d = np.array([0.0, 0.0])
eta_2d = 0.01
num_steps_2d = 50000
sigma_2d = 0.5
weights_trajectory = simulate_sgd_2d(w0_2d, eta_2d, num_steps_2d, sigma_2d)

# Tracé du paysage de perte 2D
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = loss_function_2d(X, Y)

plt.figure(figsize=(12, 6))

# Sous-figure 1: Trajectoire longue
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
plt.plot(weights_trajectory[:, 0], weights_trajectory[:, 1], 'r-', alpha=0.5, label='Trajectoire de la SGD')
plt.scatter(weights_trajectory[-100:, 0], weights_trajectory[-100:, 1], c='red', s=5, label='Points finaux')
plt.title('Visualisation d\'une trajectoire de la SGD')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.legend()
plt.grid(True)
plt.savefig('trajectoire_longue.pdf')
plt.close()

# Sous-figure 2: Distribution stationnaire
plt.subplot(1, 2, 2)
plt.hist2d(weights_trajectory[:, 0], weights_trajectory[:, 1], bins=50, cmap='inferno')
plt.title('Distribution de l\'ensemble des points d\'équilibre')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.colorbar(label='Fréquence')
plt.savefig('distribution_stationnaire.pdf')
plt.close()

# --- 3. Comparaison avec la solution de l'équation de Fokker-Planck ---
# Ici, nous allons simuler une distribution de points qui correspond
# à la solution analytique de la distribution de Boltzmann.
# La densité de probabilité est p(w) ~ exp(-V(w)).
def get_boltzmann_dist(x, y, T):
    """Calcule la densité de probabilité de Boltzmann pour un potentiel donné."""
    return np.exp(-loss_function_2d(x, y) / T)

# Température effective (correspond à sigma^2 / learning_rate)
T_eff = sigma_2d**2 / eta_2d

# Création d'une grille pour le potentiel
x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
boltzmann_density = get_boltzmann_dist(x_grid, y_grid, T_eff)

plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, boltzmann_density, 50, cmap='viridis')
plt.scatter(weights_trajectory[::100, 0], weights_trajectory[::100, 1], c='red', s=5, alpha=0.5, label='Points SGD')
plt.title('Comparaison entre simulation SGD et solution de Fokker-Planck')
plt.xlabel('Poids $w_1$')
plt.ylabel('Poids $w_2$')
plt.colorbar(label='Densité de probabilité')
plt.legend()
plt.savefig('fokker_planck_comparison.pdf')
plt.close()
