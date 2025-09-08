# Connection between Stochastic Gradient Descent and Stochastic Differential Equations

This project explores the connection between **Stochastic Gradient Descent (SGD)**, a central algorithm in deep learning, and the mathematical framework of **Stochastic Differential Equations (SDEs)**.  
Inspired by the analogy with the **Langevin equation** in statistical physics, the work proposes a theoretical framework to analyze the dynamics of SGD and explain its ability to converge toward generalizable solutions.

---

## Project Structure

- `sgd_et_sde.pdf`: Final report (theoretical framework + numerical results).  
- `figures/`: Folder containing the plots and visualizations..  
- `python_scripts/`: Python scripts for numerical experiments and figure generation.  

---

## Key Points

- **SGD as an SDE**: Discrete updates of SGD are modeled as a continuous stochastic process (Langevin SDE).  
- **Analogy with statistical physics**:  
  - Loss function gradient = drift force.  
  - Mini-batch noise = random diffusion force.  
- **Convergence to flat minima**: The stochasticity of SGD favors wide, flat minima, empirically linked to better generalization.  
- **Distribution analysis**: Comparison with the Fokker–Planck equation validates the theoretical analogy.  

---

## Goal

Provide a rigorous framework to understand the dynamics of SGD and its generalization properties, bridging deep learning and the theory of stochastic processes.
