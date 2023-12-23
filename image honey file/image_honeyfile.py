import cv2
import matplotlib.pyplot as plt
import numpy as np

imgGray = cv2.imread("output.png")
new_img = np.zeros(imgGray.shape)

for i in range(3):
    imgFloat32 = np.float32(imgGray[:, :, i])
    dft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    result_1 = 20*np.log(cv2.magnitude(dftShift[:, :,0], dftShift[:, :,1]))

    zNorm = cv2.imread("wartMark512_512.png", flags=0)
    zNorm = zNorm / 255
    zNorm = zNorm.astype(int)
    rows, cols = imgGray.shape[:2]
    maskGauss = np.zeros((rows, cols, 2), np.uint8)
    maskGauss[:, :, 0] = zNorm
    maskGauss[:, :, 1] = zNorm
    dftTrans = dftShift * maskGauss

    result_2 = 20*np.log(cv2.magnitude(dftTrans[:, :,0], dftTrans[:, :, 1]))
    ishift = np.fft.ifftshift(dftTrans)
    idft = cv2.idft(ishift)
    imgRebuild = cv2.normalize(cv2.magnitude(idft[:, :, 0], idft[:, :, 1]), None, 0, 255, cv2.NORM_MINMAX)
    new_img[:, :, i] = imgRebuild

new_img = np.around(new_img)
new_img = new_img.astype(int)

y = -0.1
fs = 6
plt.figure()
plt.subplot(2, 2, 1), plt.title("(A)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(imgGray)
plt.subplot(2, 2, 2), plt.title("(B)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(new_img)
plt.subplot(2, 2, 3), plt.title("(C)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(result_1, 'gray')
plt.subplot(2, 2, 4), plt.title("(D)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(result_2, 'gray')
plt.savefig('fft_highpass.pdf')


imgGray = new_img
imgFloat32 = np.float32(imgGray[:, :, 1])
dft = cv2.dft(imgFloat32, flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
result_1 = 20*np.log(cv2.magnitude(dftShift[:, :,0], dftShift[:, :,1]))

plt.figure()
plt.subplot(1, 2, 1), plt.title("(A)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(imgGray)
plt.subplot(1, 2, 2), plt.title("(B)", y=y, fontsize=fs), plt.axis('off')
plt.imshow(result_1, 'gray')
plt.savefig('ifft_highpass.pdf')


