from scipy.io import loadmat
import matplotlib.pyplot as plt


def showSamples(m):
    X = m['X']
    y = m['y']

    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
    for i in range(4):
        for j in range(4):
            nIndex = 4*i+j
            theImage = X[:, :, :, nIndex]
            theLabel = y[nIndex][0]
            axs[i, j].imshow(theImage)
            axs[i, j].set_title(f'Label={theLabel}')
            axs[i, j].axis("off")
    plt.show()

    # plt.imshow(theImage)
    # plt.show()
    # print(f'theLabel={theLabel}')


train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')

showSamples(train)
