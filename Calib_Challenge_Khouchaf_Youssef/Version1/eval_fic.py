import numpy as np


TEST_DIR = 'video_4/'
GT_DIR = 'video_4/'

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))


zero_mses = []
mses = []


gt = np.loadtxt(GT_DIR + str(4) + '.txt')
zero_mses.append(get_mse(gt, np.zeros_like(gt)))

test = np.loadtxt(TEST_DIR + str(4) + '_predict.txt')
mses.append(get_mse(gt, test))

percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
