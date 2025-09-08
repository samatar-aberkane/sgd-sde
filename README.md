Lien entre Descente de Gradient Stochastique et Équations Différentielles Stochastiques
Ce projet de recherche explore la connexion fondamentale entre la dynamique des algorithmes de Descente de Gradient Stochastique (SGD), qui sont au cœur de l'apprentissage profond, et le formalisme mathématique des Équations Différentielles Stochastiques (EDS). En se basant sur l'analogie avec l'équation de Langevin issue de la physique statistique, ce travail propose un cadre théorique robuste pour analyser le comportement de la SGD.

Concepts clés abordés

La SGD comme un processus stochastique : Le rapport modélise les mises à jour discrètes et bruyantes de la SGD comme un processus continu régi par une EDS de Langevin. Cette approche permet de décrire le mouvement des paramètres du modèle à travers le paysage de perte en utilisant des outils de la mécanique statistique.

L'analogie avec le mouvement brownien : Le gradient de la fonction de perte est interprété comme une force de dérive qui dirige l'optimisation, tandis que le bruit inhérent au mini-batch agit comme une force de diffusion. Cette "diffusion" permet à l'algorithme d'explorer efficacement le paysage de perte, évitant ainsi les minima locaux et les points de selle.

Convergence vers les minima plats : Une des conclusions majeures du rapport est que la stochasticité de la SGD agit comme une "température" qui facilite la convergence vers des minima larges et plats. Cette propriété est directement liée à une meilleure généralisation des modèles en apprentissage automatique.

Dynamique des distributions : Le projet va au-delà de l'analyse de trajectoires pour étudier le comportement global du système. Le rapport compare les distributions de probabilité des poids obtenues par simulation de la SGD avec la solution de l'équation de Fokker-Planck, validant l'analogie théorique.

Expériences numériques et visualisation

Le rapport s'appuie sur des simulations numériques pour illustrer les concepts théoriques. Les scripts Python utilisés pour générer les figures sont inclus et peuvent être reproduits. Ils visualisent notamment :

La trajectoire d'une SGD sur un paysage de perte en 3D, mettant en évidence son caractère bruyant et exploratoire.

La capacité de la SGD à échapper aux points de selle.

La distribution stationnaire des paramètres du modèle, comparée à celle prédite par la théorie de Fokker-Planck.

Structure du dépôt

sgd et sde.pdf : Le rapport final complet au format PDF.

main.tex : Le fichier source principal du rapport en LaTeX.

biblio.bib : La bibliographie au format BibTeX.

figures/ : Le dossier contenant les graphiques et visualisations.

python_scripts/ : Les scripts Python utilisés pour générer les figures.

Comment reproduire le rapport

Générer les figures : Exécutez les scripts Python dans le dossier python_scripts/ pour créer les images du rapport. Ces scripts requièrent numpy et matplotlib.

Bash
pip install numpy matplotlib
python generate_figure_1.py
# ... et ainsi de suite
Compiler le rapport : Utilisez un compilateur LaTeX (par exemple, Overleaf) pour compiler main.tex. Assurez-vous d'utiliser BibTeX pour la bibliographie.

Ce projet démontre comment des outils issus de la physique statistique et du calcul stochastique peuvent éclairer des questions fondamentales en apprentissage automatique.
