import time
import warnings
from solver import *

warnings.filterwarnings("error")


def main():
    np.random.seed(seed=round(time.time()))

    B = 500

    for idx1 in [1]:
        for idx2 in [1]:
            print('Model %d/Case %d --- start!' % (idx1, idx2))
            mc_samples = [[] for _ in range(solver_cnt)]

            time_start = time.time()
            for iter in range(B):
                if iter % 50 == 0:
                    cur_time = time.time()
                    print('elapsed time: ' + str(cur_time - time_start))
                    time_start = cur_time
                print(iter + 1, '/', B)
                while True:
                    X, Y, D = generate_data(error_model=idx1, mean_structure=idx2)
                    try:
                        res = [
                            full_sample(Y),
                            missing_at_random(X, Y, D),
                            fully_parametric(X, Y, D),
                            gaussian_mixture(X, Y, D),
                            new_method(X, Y, D)
                        ]
                        break
                    except RuntimeWarning:
                        print('No Convergence Error! Re-sampling...')

                for i in range(solver_cnt):
                    mc_samples[i].append(res[i])
                # mc_samples[0].append(res[0])
                # print('new_method:')
                # print('Mean: %f, Variance: %f' % (np.average(mc_samples[4]), np.var(mc_samples[4])))

            # print('new_method:')
            # print('Mean: %f, Variance: %f' % (np.average(mc_samples[0]), np.var(mc_samples[0])))
            print('full sample:')
            print('Mean: %f, Variance: %f' % (np.average(mc_samples[0]), np.var(mc_samples[0])))
            print('missing_at_random:')
            print('Mean: %f, Variance: %f' % (np.average(mc_samples[1]), np.var(mc_samples[1])))
            print('fully_parametric:')
            print('Mean: %f, Variance: %f' % (np.average(mc_samples[2]), np.var(mc_samples[2])))
            print('gaussian_mixture:')
            print('Mean: %f, Variance: %f' % (np.average(mc_samples[3]), np.var(mc_samples[3])))
            print('new_method:')
            print('Mean: %f, Variance: %f' % (np.average(mc_samples[4]), np.var(mc_samples[4])))


if __name__ == '__main__':
    main()

