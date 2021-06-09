from scipy import signal
from scipy import signal
import numpy as np

def butterworth(data,order,w_n,f_sampling =1000,filt_type='lp'):
    sos = signal.butter(order, w_n, filt_type, fs=f_sampling, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered

def lfilter(data,n_iter,denom=1):
    b = [1.0 / n_iter] * n_iter
    filtered = signal.lfilter(b, denom, data)
    return filtered

def savgol_filter(data,order,window_length):
    filtered = savgol_filter(data, window_length, order)
    return filtered
if __name__=="__main__":
    import matplotlib.pyplot as plt

    mu, sigma = 0, 500

    x = np.arange(1, 100, 0.1)  # x axis
    z = np.random.normal(mu, sigma, len(x))  # noise
    y = x ** 2 + z  # data
    #plt.plot(x, y, linewidth=2, linestyle="-", c="b")

    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lin_filter(y, n, a)
    plt.plot(x, yy, linewidth=2, linestyle="-", c="b")  # smooth by filter
    plt.show()