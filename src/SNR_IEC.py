'''
This script is tested with Python 3.6 on 64 bit Windows 10
It should work fine on other Platforms and Python 2.7 or higher.

I run it, <numpy> and <opencv> have to be installed.

For Windows:
- Download packages:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
- Install via pip in a command shell:
>>> pip install PACKAGENAME.whl

For other OS (Linux):
- Follow...
https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html
https://www.scipy.org/install.html
'''

import numpy as np


def SNR_IEC(i1, i2, ibg=0):
    '''
    Calculate the averaged signal-to-noise ratio SNR50
    as defined by IEC NP 60904-13
    
    i1, i2 ... EL images of same PV device
    ibg    ... background image
    
    All images should be cut to the shape of the PV device. 
    Otherwise SNR will be influenced by background.
    '''

    def prepare(img):
        assert isinstance(img, np.ndarray), 'images need to numpy arrays'
        assert img.ndim in (2, 3), 'image ill shaped {}'.format(img.shape)
        assert i1.shape[:2] == img.shape[:2], 'all input images need to have the same resolution: {}!={}'.format(
            i1.shape, img.shape)
        
        if img.ndim == 3:  # format rgb -> gray
            assert img.shape[2] == 3, 'only grayscale or RGB images are accepted'
            img = np.average(img, axis=2, weights=(0.114, 0.587, 0.299))
        return  img.astype(np.float64)

    i1 = prepare(i1)
    i2 = prepare(i2)
    
    if ibg is not 0:
        ibg = prepare(ibg)

    # SNR calculation:
    signal = ((i1 + i2) / 2 - ibg).sum()
    f = (0.5 ** 0.5) * ((2 / np.pi) ** -0.5)
    noise = np.abs(i1 - i2).sum() * f
    
    return signal / noise


if __name__ == '__main__':
    import os
    import cv2
    
    p = os.path
    root = p.dirname(p.dirname(__file__))
    root = p.join(root, 'media')
    
    i1 = cv2.imread(p.join(root, 'EL1.jpg'))
    i2 = cv2.imread(p.join(root, 'EL2.jpg'))
    ibg = cv2.imread(p.join(root, 'bg.jpg'))

    print('Signal-to-noise ratio = ', SNR_IEC(i1, i2, ibg))
    
