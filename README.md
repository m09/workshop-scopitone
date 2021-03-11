# Workshop machine learning & expérimentations créatives

## Installation

Avant de commencer l'installation, vous pouvez créer un environnement virtuel.

    pip install --user virtualenv
    python3 -m virtualenv env
    source env/bin/activate

Vous pouvez maintenant installer PyTorch en vous rendant sur le [site
officiel](https://pytorch.org/) et en choisissant la combinaison qui correspond à votre
environnement.

Vous pouvez ensuite installer le reste des dépendances avec [Poetry](https://python-poetry.org/) :

    poetry install

## Utilisation

Lancez l'espace de travail avec

    python -m jupyter notebook

Il faut ensuite exécuter le notebook _Entrainer un réseau en le modifiant pour une tâche
spécifique_ avant d'aller plus loin, pour créer un modèle utilisable.

Vous pouvez ensuite lancer le notebook _Combinaison de modèles_.

Le serveur de style est quasiment entièrement tiré d'[un exemple de
PyTorch](https://github.com/pytorch/examples/tree/master/fast_neural_style).

## Ressources

Voici quelques ressources citées pendant le workshop :

- https://processing.org/
- https://openframeworks.cc/
- http://ml4a.github.io/
- https://ml5js.org/
- https://pytorch.org/tutorials/

## Exemples

Voici quelques exemples montrés pendant le workshop :

- https://github.com/luanfujun/deep-painterly-harmonization
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://github.com/pytorch/examples/tree/master/fast_neural_style
- https://github.com/ryankiros/neural-storyteller
- https://github.com/karpathy/neuraltalk2
- https://github.com/karpathy/char-rnn
- https://experiments.withgoogle.com/ai/bird-sounds/view/
- https://experiments.withgoogle.com/ai/drum-machine/view/
