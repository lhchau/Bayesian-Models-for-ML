## Intuition

- Laplace approximation is a method to **approximate posterior into Normal distribution** through **Taylor expansion** (second-order) and MAP for estimating a good point for Taylor expansion

## Approach

- Setup a simple problem

### Requirement

- Data in the form of labeled paris D = $\{(x_i, y_i)\}^N_{i=1}$ where $x \in \R^d$ and $y \in \{ 0, 1\}$

### Modeling

- $y_i \sim^{iid} \text{Bernoulli}(\sigma(x^T_iw)) $
- $\sigma(x_i^Tw) = \frac{e^{x^T_iw}}{1 + e^{x^T_iw}}$

### Prior

- $w \sim Normal(\theta, cI)$

### Posteior

- $p(w|X,\hat{y}) = \frac{\Pi^N_{i=1} p(y_i|x_i, w)p(w)}{\int\Pi^N_{i=1} p(y_i|x_i, w)p(w)dw}$
- If we plug in $y_i \sim \text{Bernoulli}(\sigma(x^T_iw))$ right here, we can see that denominator cannot be solved, intracable integral. So we need Laplace approach

### Laplace approximation

- **Approximate posterior into Normal distribution** => $p(\theta|X) \sim \text{Normal}(\mu, \Sigma)$
- Trick to express posterior => $p(\theta|X) = \frac{e^{lnp(X,\theta)}}{\int e^{lnp(X,\theta)}d\theta}$
- Let $f(\theta) = lnp(X, \theta)$
- **Taylor expansion** => $f(\theta) \sim f(\theta_0) + (\theta - \theta_0)^T \nabla f(\theta_0) + \frac{1}{2}(\theta - \theta_0)^T \nabla^2 f(\theta_0)(\theta-\theta_0)$
- Choose $\theta_0 = \theta_{MAP}$ => $ (\theta - \theta*{MAP})^T \nabla f(\theta*{MAP}) = 0 \ (\text{bcz} \ \nabla f(\theta\_{MAP}) = 0)$
- Final: $p(\theta|X) = \frac{e^{ln(X, \theta_{MAP}) + \frac{1}{2}(\theta - \theta_{MAP})^T \nabla^2ln(x, \theta_{MAP}) (\theta - \theta_{MAP})}}{\int e^{ln(X, \theta_{MAP}) + \frac{1}{2}(\theta - \theta_{MAP})^T \nabla^2ln(x, \theta_{MAP}) (\theta - \theta_{MAP})}}d\theta = \newline \frac{e^{-\frac{1}{2}(\theta - \theta_{MAP})^T -\nabla^2ln(x, \theta_{MAP}) (\theta - \theta_{MAP})}}{\int e^{-\frac{1}{2}(\theta - \theta_{MAP})^T -\nabla^2ln(x, \theta_{MAP}) (\theta - \theta_{MAP})}}d\theta$
- $p(\theta|X) \sim \text{Normal} (\theta_{MAP}, (-\nabla^2ln(x, \theta_{MAP}))^{-1})$

### Approximate Hessian

- [Generalized Gauss-Newton matrix (GGN)](https://jmlr.org/papers/volume21/17-678/17-678.pdf)

### Prediction

- $p(y^*|x^*, D) = \int p(y^*|x^*, \theta)p(\theta|D)d\theta$

- For each input we make total S predictions: $\frac{1}{S}\Sigma^S_{s=1}p(y^*|x^*, \theta_S)$
