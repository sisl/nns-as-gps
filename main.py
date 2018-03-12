import numpy as np
import tensorflow as tf
import pdb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from models import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():

    with tf.Session() as sess:


        print('Making EnsembleMSE model...')
        ensemble_mse = EnsembleMSE(sess, 1, 1, 'ensemble_mse', 
            max_epochs = 300, hidden_sizes = [128,64,64,64], validation_ratio=0.2,
            weight_reg = 0., dropout_rate=0.00, learning_rate=1e-3)

        print('Making EnsembleNLL model...')
        ensemble_nll = EnsembleNLL(sess, 1, 1, 'ensemble_nll', 
            max_epochs = 2400, hidden_sizes = [128,64,64,64], validation_ratio=0.2,
            weight_reg = 0., dropout_rate=0.00, learning_rate=1e-4)

        print('Making MCDropoutMSE model...')
        mcdropout_mse = MCDropoutMSE(sess, 1, 1, 'mcdropout_mse', 
            max_epochs = 2400, hidden_sizes = [128,64,64,64], validation_ratio=0.2,
            weight_reg = 0., dropout_rate=0.25, learning_rate=5e-4) 

        tf.global_variables_initializer().run()

        ## load training data
        xx = np.linspace(-1,1, 10).reshape((10,1))
        def fun(xx):
            return np.sin(xx*np.pi) + 0.2*np.sin(xx*4*np.pi)

        ensemble_mse.load_data([xx, fun(xx)])
        ensemble_nll.load_data([xx, fun(xx)])
        mcdropout_mse.load_data([xx, fun(xx)])
        # set seed so that training sets are the same
        np.random.seed(1)
        ensemble_mse.split_train_test()
        np.random.seed(1)
        ensemble_nll.split_train_test()
        np.random.seed(1)
        mcdropout_mse.split_train_test()

        ## create test data, evaluate GP
        xx_test = np.linspace(-1.5,1.5, 100).reshape((100,1))

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(ensemble_mse.train_data_X, ensemble_mse.train_data_Y)

        gp_y, gp_sigma = gp.predict(xx_test, return_std=True)
        gp_sigma = gp_sigma.reshape(gp_y.shape)

        ## plot GP results
        fig = plt.figure()
        gp_ax = fig.add_subplot(2,2,1, xlim=(-1.5,1.5), ylim=(-2,2))
        gp_ax.set_title('GP (RBF Kernel)')
        gp_ax.plot(xx_test, fun(xx_test), color='blue')
        gp_ax.plot(ensemble_mse.train_data_X, 
            ensemble_mse.train_data_Y, 'o', color='yellow')
        gp_ax.plot(xx_test, gp_y, color='red')
        gp_ax.plot(xx_test, gp_y + 2*gp_sigma, '--', color='red')
        gp_ax.plot(xx_test, gp_y - 2*gp_sigma, '--', color='red')

        xx_test.reshape((100,))

        ## train models, queries fit on test set on quadratically numbered epochs
        ensemble_mse_means = []
        ensemble_mse_stds = []
        ensemble_nll_means = []
        ensemble_nll_stds = []
        mcdropout_mse_means = []
        mcdropout_mse_stds = []

        test_epochs = [i**2 + 1 for i in range(50)]

        # sum GP stds to match to when plotting
        gp_std_sum = np.sum(gp_sigma)

        prev_test_epoch = 0
        for i in range(len(test_epochs)):
            print('Epoch %d' % prev_test_epoch)
            epochs_to_run = test_epochs[i] - prev_test_epoch
            prev_test_epoch = test_epochs[i]

            ensemble_mse.max_epochs = epochs_to_run
            ensemble_nll.max_epochs = epochs_to_run
            mcdropout_mse.max_epochs = epochs_to_run

            ensemble_mse.train()
            ensemble_nll.train()
            mcdropout_mse.train()

            # test models
            means, sigsqs = ensemble_mse.eval(xx_test)
            stds = np.sqrt(sigsqs)
            scalefac = np.sum(gp_sigma) / np.sum(stds)
            stds *= scalefac
            ensemble_mse_means.append(means)
            ensemble_mse_stds.append(stds)

            means, sigsqs = ensemble_nll.eval(xx_test)
            stds = np.sqrt(sigsqs)
            scalefac = np.sum(gp_sigma) / np.sum(stds)
            stds *= scalefac
            ensemble_nll_means.append(means)
            ensemble_nll_stds.append(stds)

            means, sigsqs = mcdropout_mse.eval(xx_test)
            stds = np.sqrt(sigsqs)
            scalefac = np.sum(gp_sigma) / np.sum(stds)
            stds *= scalefac
            mcdropout_mse_means.append(means)
            mcdropout_mse_stds.append(stds)

        mean_datas = [ensemble_mse_means, ensemble_nll_means, mcdropout_mse_means]
        std_datas = [ensemble_mse_stds, ensemble_nll_stds, mcdropout_mse_stds]



        # plot animation
        ensemble_mse_ax = fig.add_subplot(222, xlim=(-1.5,1.5), ylim=(-2,2))
        ensemble_mse_ax.set_title('Ensemble')
        ensemble_mse_ax.plot(xx_test, fun(xx_test), color='blue')
        ensemble_mse_ax.plot(ensemble_mse.train_data_X, 
            ensemble_mse.train_data_Y, 'o', color='yellow')
        emse_mean, = ensemble_mse_ax.plot([],[], color='red')
        emse_std_u, = ensemble_mse_ax.plot([],[], '--', color='red')
        emse_std_l, = ensemble_mse_ax.plot([],[], '--', color='red')
        emse_epoch = ensemble_mse_ax.text(0.05, 0.9, '', transform=ensemble_mse_ax.transAxes)

        ensemble_nll_ax = fig.add_subplot(223, xlim=(-1.5,1.5), ylim=(-2,2))
        ensemble_nll_ax.set_title('Ensemble with Pred. Var')
        ensemble_nll_ax.plot(xx_test, fun(xx_test), color='blue')
        ensemble_nll_ax.plot(ensemble_mse.train_data_X, 
            ensemble_mse.train_data_Y, 'o', color='yellow')
        enll_mean, = ensemble_nll_ax.plot([],[], color='red')
        enll_std_u, = ensemble_nll_ax.plot([],[], '--', color='red')
        enll_std_l, = ensemble_nll_ax.plot([],[], '--', color='red')
        enll_epoch = ensemble_nll_ax.text(0.05, 0.9, '', transform=ensemble_nll_ax.transAxes)

        mcdropout_mse_ax = fig.add_subplot(224, xlim=(-1.5,1.5), ylim=(-2,2))
        mcdropout_mse_ax.set_title('MC-Dropout')
        mcdropout_mse_ax.plot(xx_test, fun(xx_test), color='blue')
        mcdropout_mse_ax.plot(ensemble_mse.train_data_X, 
            ensemble_mse.train_data_Y, 'o', color='yellow')
        dmse_mean, = mcdropout_mse_ax.plot([],[], color='red')
        dmse_std_u, = mcdropout_mse_ax.plot([],[], '--', color='red')
        dmse_std_l, = mcdropout_mse_ax.plot([],[], '--', color='red')
        dmse_epoch = mcdropout_mse_ax.text(0.05, 0.9, '', transform=mcdropout_mse_ax.transAxes)

        mean_lines = [emse_mean, enll_mean, dmse_mean]
        std_u_lines = [emse_std_u, enll_std_u, dmse_std_u]
        std_l_lines = [emse_std_l, enll_std_l, dmse_std_l]

        epoch_texts = [emse_epoch, enll_epoch, dmse_epoch]

        def init():
            for lnum, line in enumerate(mean_lines):
                line.set_data([], [])
                std_u_lines[lnum].set_data([], [])
                std_l_lines[lnum].set_data([], [])
            for text in epoch_texts:
                text.set_text('')

            return tuple(mean_lines) + tuple(std_u_lines) + tuple(std_l_lines) + tuple(epoch_texts)

        def animate(i):
            for lnum, line in enumerate(mean_lines):
                line.set_data(xx_test, mean_datas[lnum][i])
                std_u_lines[lnum].set_data(xx_test, mean_datas[lnum][i] + 2*std_datas[lnum][i])
                std_l_lines[lnum].set_data(xx_test, mean_datas[lnum][i] - 2*std_datas[lnum][i])
            for tnum, text in enumerate(epoch_texts):
                text.set_text('Epoch %d' % test_epochs[i])

            return tuple(mean_lines) + tuple(std_u_lines) + tuple(std_l_lines) + tuple(epoch_texts)

        plt.tight_layout()
        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(test_epochs)),
                              interval=100, blit=True, init_func=init)

        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=10, bitrate=3600)
        # ani.save('./gp_comp.mp4', writer=writer)

        writer = animation.ImageMagickWriter(fps=10, bitrate=5000)
        ani.save('./gp_comp.gif', writer=writer)

        plt.show()



















if __name__ == '__main__':
    main()

