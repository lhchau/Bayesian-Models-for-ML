# Restricted Boltzmann Machines

- Implementation of RBMs for toy dataset MNIST

## Intuition

- Model: a fully connected layer
- Optimization: Contrastive Divergence (CD) + SGD momentum + Gibbs sampling

The idea behind CD is that the model distribution will eventually converge to the true data distribution through estimating the difference between the two distributions, CD allows the RBM to learn the features of the input data and generate new data samples that are similar to the input data.
