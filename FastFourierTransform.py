import numpy as np
import matplotlib.pyplot as plt

def n_big_coef(ck, n):
	 
	ck1 = np.absolute(ck)
	#ck2 = np.argsort(ck1)[:n]
	#print(ck2)
	ck1_sort = (np.sort(ck1))[-n:]

	ck1 = list(ck1)
	x = np.zeros(len(ck), dtype= 'complex128')
	
	for i in range(n):
		pos = ck1.index(ck1_sort[i])
		x[pos] = ck[pos]
		#x[i] = ck[i]
	print(x)
	return x

#def min_appx(apx, dax)
def ifft_ncoeff (arr, n):
    return np.fft.irfft(take_fft_coeff(arr,n))
 
if __name__ == '__main__':	
	dax = np.genfromtxt('dax_data.txt',delimiter = ' ')
	dax = np.array(dax)

	daxx = np.fft.rfft(dax)
	dax1 = np.fft.irfft(n_big_coef(daxx, 1))
	dax5 = np.fft.irfft(n_big_coef(daxx, 5))
	dax10 = np.fft.irfft(n_big_coef(daxx, 10))
	dax123 = np.fft.irfft(n_big_coef(daxx, 123))
	```````````````````````````
	print(np.average(dax))

	plt.plot(dax, 'g')
	plt.plot(dax5, 'b', dax1, 'r', dax10)
	plt.xticks([])
	plt.show()


