# Tips and Tricks

Here, I will lay out some of the **aha!**, **gotcha!** and **touché!** :smile:

- **Generic**
   1. configs.yaml file is a better alternatives than manual argparse. 
   It lets us save different default values for the parameter combination (i.e. train, debug, test, gymnasium, dm-control etc.)
   2. Use wrapper for environments from gymnasium or dm-control to change the default behaviour rather 
   than writing out those changes explicitly.
   3. Use `namedtuple` to maintain multiple attributes of a variable.


- **Technical**
   1. The reward from the dm-control is bounded between 0.0 and 1.0. 
   Using the mean of unit variance Gaussian to model them might seem weird and counter-intuitive.
   But, remember that we only sample with that variance during training time. During test, we just take the predicted mean.
   2. Do not clip the sampled values (i.e. pixel values, rewards etc.) during the training phase. 
   However, you must clip them within valid range during the planning and test phase.
   3. Usage of `torch.distributions.Independent` is very handy to handle multivariate Gaussian distributions. 
   Check out [this](https://bochang.me/blog/posts/pytorch-distributions/) 
   and [this](https://pytorch.org/docs/stable/distributions.html#independent) for further clarifications.
   4. When computing the loss, take average across the batch and temporal dimension
   ([ref](https://github.com/google-research/planet/issues/28#issuecomment-487433102)). 
   Specially when computing the reconstruction loss, sum the squared error across all the pixels and the channels. 
   Then take average across the batch and time dimension.
   5. The kl-divergence should be computed from the posterior to the prior. 
   That is we want to train the prior towards the posterior while regularizing the posterior towards prior.
   Later on, we could use techniques like [kl-balancing](https://arxiv.org/pdf/2010.02193.pdf) 
   where we can minimize the kl-loss faster with respect to the prior than the representation (posterior).
   6. Free nats should be applied to each pair of prior and posterior individually rather than on the mean kl-loss.
   ([ref](https://github.com/google-research/planet/issues/28#issuecomment-487373263)).
   7. `rsample()` is the reparametarized sampling that allows the gradient to flow through the sampling step. 
   Whereas `sample()` is used to implement REINFORCE with the help of `log_prob()`. 
   Check [here](https://pytorch.org/docs/stable/distributions.html#score-function) for more.
   8. For a CNN-based encoder, `MaxPooling2d` is used to enforce spatial invariance, 
   whereas `stride` is used to aggressively reduce the dimension. Here, we use `stride`.
   9. For TransposedCNN-based decoder, `stride` gives us better flexibility over `Upsample()`, 
   cause the `stride` option has to learn the proper weights to be used for expansion, that is it learns the required function. 
   `Upsample()` simply implements some predefined function instead.
   10. When using some neural network to predict the standard deviation of the stochastic states, 
   use `softplus(pre_std + 0.55)` to make sure that the variables have closer to unit variance.
   11. Since there is an RNN within, `grad_norm_clip()` is useful for preventing exploding-gradient problem.
   12. Add small white (zero-mean Gaussian) noise to the observation before feeding them into the model during training.