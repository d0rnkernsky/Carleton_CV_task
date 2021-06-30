import numpy as np
from matplotlib import pyplot as plt


def count_subarrays(arr, sub):
    """
    Computes the number of occurrences of any subset array in the given array
    :param arr: the array to search the subsets
    :param sub: a subset
    :return: the number of occurrences
    """
    assert arr is not None
    assert sub is not None
    assert isinstance(arr, np.ndarray)
    assert isinstance(arr, np.ndarray)

    # basic edge-case check
    if sub.shape[0] == 0:
        return 0
    if arr.shape[0] == sub.shape[0] and np.allclose(arr, sub, 1e-10):
        return 1

    count = 0
    for i in range(arr.shape[0] - sub.shape[0] - 1):
        a = arr[i:i + sub.shape[0]]
        if np.allclose(a, sub, 1e-10):
            count = count + 1

    return count


def histogram(arr, bins_num=10):
    """
    Plots histogram using matplotlib bars of unique values in arr
    :param arr: array with values
    """
    assert arr is not None
    assert isinstance(arr, np.ndarray)

    # create bins for frequencies
    left, right = arr.min(), arr.max()
    bins = np.linspace(start=left, stop=right, num=bins_num + 1, endpoint=True)

    # count frequencies
    hist = np.zeros(bins_num, arr.dtype)
    for i in range(arr.shape[0]):
        num = arr[i]
        for bi in range(bins.shape[0]):
            if bi == bins.shape[0] - 1 and num == bins[bi]:
                hist[bi - 1] += 1
                break
            else:
                if bins[bi] <= num < bins[bi + 1]:
                    hist[bi] += 1
                    break

    # plot histogram as bars for each bin with height as a frequency
    width = abs(bins[1]) - abs(bins[0])
    for i in range(hist.shape[0]):
        plt.bar(x=bins[i], height=hist[i], width=width, alpha=0.5, align='edge', color='blue')

    plt.xticks(bins)
    plt.show()


def main():
    # tests
    arr = np.array([0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3])
    assert count_subarrays(arr, np.array([1, 1])) == 3
    assert count_subarrays(arr, np.array([1, 1, 1])) == 1
    assert count_subarrays(arr, arr) == 1
    assert count_subarrays(arr, np.array([])) == 0
    assert count_subarrays(arr, np.array([1])) == 5
    assert count_subarrays(arr, np.array([1, 3])) == 1

    histogram(arr)
    d = np.random.laplace(loc=15, scale=3, size=500)
    histogram(d)

    # plt.hist(arr)
    # plt.show()
    #
    # plt.hist(d)
    # plt.show()


if __name__ == '__main__':
    main()
