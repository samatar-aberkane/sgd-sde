import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Définir le paysage de perte (Loss landscape)
def loss_function(x, y):
    return np.sin(x) * np.cos(y) * np.exp(- (x**2 + y**2) / 4) + (x**2 + y**2) / 20

# 2. Générer les données pour le paysage en 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# 3. Simuler la trajectoire de la Descente de Gradient Stochastique (SGD)
start_point = np.array([-4.0, 4.0])
learning_rate = 0.5
noise_scale = 0.5
num_steps = 200

path = [start_point]
current_position = start_point

for i in range(num_steps):
    grad_x = np.cos(current_position[0]) * np.cos(current_position[1]) * np.exp(- (current_position[0]**2 + current_position[1]**2) / 4) - current_position[0] / 10
    grad_y = -np.sin(current_position[0]) * np.sin(current_position[1]) * np.exp(- (current_position[0]**2 + current_position[1]**2) / 4) - current_position[1] / 10

    noise = np.random.normal(0, noise_scale, 2)

    new_position = current_position - learning_rate * np.array([grad_x, grad_y]) + noise
    path.append(new_position)
    current_position = new_position

path = np.array(path)
path_z = loss_function(path[:, 0], path[:, 1])

# 4. Créer le graphique en 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Dessiner le paysage de perte (surface) avec un thème de couleurs plus sombre
ax.plot_surface(X, Y, Z, cmap='cividis', alpha=0.9, antialiased=True, rstride=1, cstride=1, linewidth=0)

# Ajouter les lignes de contour sur le fond pour plus de profondeur
ax.contourf(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], cmap='cividis', levels=20, alpha=0.6)

# Dessiner la trajectoire de la SGD (ligne) avec une couleur vive
ax.plot(path[:, 0], path[:, 1], path_z, color='red', linewidth=3, label='SGD Trajectory')

# Ajouter le point de départ et d'arrivée
ax.scatter(path[0, 0], path[0, 1], path_z[0], color='red', s=100, label='Point de d\'epart')
ax.scatter(path[-1, 0], path[-1, 1], path_z[-1], color='blue', s=100, label='Point d\'arrivée')

# Ajouter les étiquettes et le titre
ax.set_xlabel('Paramètre $w_1$')
ax.set_ylabel('Paramètre $w_2$')
ax.set_zlabel('Fonction de perte $L(w)$')
ax.set_title("Visualisation de la SGD sur un paysage de perte")
ax.legend()
ax.view_init(elev=30, azim=45) # Permet de définir l'angle de vue initial

# Sauvegarder l'image
plt.savefig("sgd_trajectory_3d_enhanced.png")

# Afficher le graphique
plt.show()