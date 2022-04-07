Models analyzed:
- ResNet-18
- ResNet-32
- VGG

Datasets:
- CIFAR10
- ImageNet

Metrics:
- L1 energy: integral of the absolute value of the density of the spectrum
- L2 energy: integral of the squared value of the density of the spectrum
- (outlier) eigenvalue ratio: max eigenvalue / Kth eigenvalue (often K = num_classes)
- truncated SVD with k vectors: compute best rank-$k$ approximation of hessian, runtime $\mathcal{O}(nk^2)$ for $n \times n$ Hessian
- we can get the top k eigenvalues $\lambda_1, ..., \lambda_k$ and solve for $H x = \lambda x$ using gradient descent on $x$ (i.e. minimize $\| Hx - \lambda \mathbb{I} x \|_2$ subject to $x$), since it is a convex problem and we can efficiently compute $H x$
- gradient fraction in top subspace: $$ \frac{\| P \nabla_\theta L(\theta) \|^2_2}{\| \nabla_\theta L(\theta) \|^2_2}$$ where P is a projection matrix from a rank-$k$ approximation of the hessiaN
- normalized inner product: the normalized inner product between the gradient vector $\nabla_\theta L(\theta_t)$ and the actual optimal direction $\theta_t - \theta^\star$, i.e. $$\frac{\nabla_\theta L(\theta_t)^T (\theta_t - \theta^\star)}{| \nabla_\theta L(\theta_t)^T (\theta_t - \theta^\star) |}$$

Matrices:
- Hessian: $$\nabla_\theta^2 L(\theta)$$
- Covariance of Minibatch Gradients: $$\frac{1}{N} \sum_{i=1}^{N} \nabla_\theta L(\theta)_i \nabla_\theta L(\theta)_i^T$$

Loss Landscape:
- sharper Hessian = more values far away from zero (max, mean); flatness is opposite of sharpness

Things to look for:
- at initialization: large negative eigenvalues, early in training these shift to small positive
- intuition: optimization proceeds in the directions of strongest (negative) curvature first
- for most of training, spectrum is flat and has positive eigenvalues
- negative eigenvalues at end of training are orders of magnitude smaller than those at beginning
- warning: since you get get zero loss on MNIST (and CIFAR10), the results there might not generalize
- small dataset: ResNet-18 has similar L1/L2 energy of positive eigenvalue subset vs negative eigenvalue subset on ImageNet, but on CIFAR10 the L2 energy of the positive eigenvalue subset is a lot bigger; this might be due to the fact that you're at a global minimum with CIFAR, so there are barely any directions to decrease
- learning rate: ResNet-32 spectral density becomes flatter (less extreme) after learning rate drop; critically, this behaviour persists after learning rate drop
- residual connections: ResNet-32 sepctral density becomes much sharper with addition of residual connections, this persists late into training
- batch norm: addition of batch norm makes loss smoother, more so with VGG than ResNets
- batch norm: for both hessian and Covariance, addition of batch norm makes eigenvalue ratio smaller, i.e. outliers are less extreme
- batch norm: using the population statistics for batch norm, as opposed to the minibatch statistics, actually makes training a lot slower with more outliers in the Hessian spectrum, however the top eigenvalue of the hessian remains the same for both methods (population vs minibatch)

Questions:
- how does the scale invariance stuff play a role?
- 

Other ideas:
- KL divergence / Wasserstein metric for eigenvalue densities
- compare with ADAM's approximation

Questions for Roger:

- Hessian-based metrics have been shown to be sensitive to reparameterizations of the neural network that do NOT actually effect the predictions of the model on the data (as we saw in Lecture 8). Do we still have those issues with the Hessian eigenspectrum metrics as tools for evaluating flatness of minima? Is this less of an issue for analyzing training dynamics since the reparameterizations would actually affect how training progresses (by affecting the Hessian)?

- We are looking at the covariance of the Jacobian matrix, which is something that the original work from Ghorbani et al did. I don't think this was directly covered in our course, although it's sort of similar to the 

- generalization work of NTK: (look at NTK paper)

- Gilmer properties of the eigenspectrum https://openreview.net/pdf?id=OcKMT-36vUs

- lanczos gives approximate tridiagonalization, basis for Krylov subspace, could be a reasonable approximation

- Lecture 6, benign overfitting (properties of feature covariance matrix for lin reg problem, exactly match training data, we want heavy tailed eigenspectrum, top eigenvalues be aligned with the signal)

- generalized trace estimation