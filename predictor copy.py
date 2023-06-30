import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from scipy.ndimage import gaussian_filter

# We will use the simplest form of GP model, exact inference
class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,)), lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.01, 1.0, sigma=0.1)),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def fit(train_x, train_y):
    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(train_y.flatten(), learn_additional_noise=True)
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        # if i % 5 == 0:
        #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #         i + 1, training_iter, loss.item(),
        #         model.covar_module.base_kernel.lengthscale.mean().item(),
        #         model.likelihood.second_noise_covar.noise.mean().item()
        #     ))
        optimizer.step()
    model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        # test_dist = model(test_x)

        # pred_means = test_dist.loc
        # pred_cov = likelihood(test_dist).stddev
        return model

def choose_next_position(model, test_x, measured_positions):
    test_dist = model(test_x)
    while True:
        pred_stddev = test_dist.stddev
        stddev_pdf = pred_stddev.sum(0).detach().numpy()
        # stddev_pdf -= stddev_pdf.min()
        stddev_pdf /= stddev_pdf.sum()
        # print(choice_pdf.shape)
        
        pred_samples = test_dist.sample(torch.Size((64,))).exp()
        probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        tot = np.ones(probabilities.shape[1:])
        tot /= np.sum(tot)
        for probability in probabilities:
            p = probability.numpy()#.reshape((test_grid_size,test_grid_size))
            tot *= p/np.sum(p)
        p_pdf = tot/np.sum(tot)

        choice_pdf = 1 * p_pdf + 3 * stddev_pdf
        choice_pdf /= choice_pdf.sum()

        # least_certain_test_index = np.random.choice(np.arange(len(choice_pdf.flatten())), p=choice_pdf.flatten())
        least_certain_test_index = np.argmax(choice_pdf.flatten())
        # least_certain_test_index = least_certain_test_index if isinstance(least_certain_test_index, int) else least_certain_test_index[0]


        if test_x[least_certain_test_index] not in measured_positions:
            break
    
    return least_certain_test_index, choice_pdf
    