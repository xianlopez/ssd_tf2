import numpy as np
import mean_ap


def test_compute_ap_1():
    print('    - test_compute_ap_1: Zero predictions')
    preds = np.zeros((0, 6), np.float32)
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP) < 1e-6


def test_compute_ap_2():
    print('    - test_compute_ap_2: Zero GT boxes')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.zeros((0, 5), np.float32)
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP) < 1e-6


def test_compute_ap_3():
    print('    - test_compute_ap_3: One GT and one correct prediction')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP - 1.0) < 1e-6


def test_compute_ap_4():
    print('    - test_compute_ap_4: One GT and one wrong prediction')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.6, 0.6, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP) < 1e-6


def test_compute_ap_5():
    print('    - test_compute_ap_5: Two predictions (both good) for one GT')
    preds = np.array([[0.1, 0.15, 0.2, 0.2, 3, 0.8], [0.15, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    # The result in this case is still 1 because we already reach recall 1 with the first prediction.
    assert np.abs(AP - 1.0) < 1e-6


def test_compute_ap_6():
    print('    - test_compute_ap_6: Two predictions (the least confident wrong) for one GT')
    preds = np.array([[0.1, 0.6, 0.2, 0.2, 3, 0.8], [0.1, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    # The result in this case is still 1 because we already reach recall 1 with the first prediction.
    assert np.abs(AP - 1.0) < 1e-6


def test_compute_ap_7():
    print('    - test_compute_ap_7: Two predictions (the most confident wrong) for one GT')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.8], [0.6, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP - 0.5) < 1e-6


def test_compute_ap_8():
    print('    - test_compute_ap_8: Three predictions (the middle one wrong) for two GTs')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.7], [0.6, 0.6, 0.22, 0.19, 3, 0.6],
                      [0.11, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3], [0.6, 0.6, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    # Here precision rectification plays a role.
    expectedAP = 0.5 + 0.5 * 2.0 / 3.0
    assert np.abs(AP - expectedAP) < 1e-6


def test_compute_ap_9():
    print('    - test_compute_ap_9: Three predictions (the last one wrong) for two GTs')
    preds = np.array([[0.1, 0.1, 0.2, 0.2, 3, 0.6], [0.6, 0.6, 0.22, 0.19, 3, 0.7],
                      [0.11, 0.1, 0.2, 0.2, 3, 0.9]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3], [0.6, 0.6, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    assert np.abs(AP - 1.0) < 1e-6


def test_compute_ap_10():
    print('    - test_compute_ap_10: Several predictions and GTs')
    preds = np.array([[0.12, 0.1, 0.2, 0.2, 3, 0.85], [0.1, 0.12, 0.2, 0.2, 3, 0.9],
                      [0.09, 0.59, 0.21, 0.2, 3, 0.7], [0.11, 0.11, 0.2, 0.2, 3, 0.75],
                      [0.6, 0.61, 0.19, 0.2, 3, 0.8]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3], [0.6, 0.1, 0.2, 0.2, 3],
                   [0.1, 0.6, 0.2, 0.2, 3], [0.6, 0.6, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    expectedAP = 0.25 * (1.0 + 2.0 / 3.0 + 3.0 / 5.0)
    assert np.abs(AP - expectedAP) < 1e-6


def test_compute_ap_11():
    print('    - test_compute_ap_11: Long precision rectification')
    preds = np.array([[0.12, 0.1, 0.2, 0.2, 3, 0.85], [0.1, 0.12, 0.2, 0.2, 3, 0.9],
                      [0.1, 0.6, 0.2, 0.2, 3, 0.75], [0.6, 0.61, 0.19, 0.2, 3, 0.8]])
    gt = np.array([[0.1, 0.1, 0.2, 0.2, 3], [0.6, 0.1, 0.2, 0.2, 3],
                   [0.1, 0.6, 0.2, 0.2, 3], [0.6, 0.6, 0.2, 0.2, 3]])
    AP = mean_ap.compute_ap(preds, gt)
    expectedAP = 0.25 + 0.5 * 0.75
    assert np.abs(AP - expectedAP) < 1e-6


if __name__ == '__main__':
    print('Tests for mean_ap')
    test_compute_ap_1()
    test_compute_ap_2()
    test_compute_ap_3()
    test_compute_ap_4()
    test_compute_ap_5()
    test_compute_ap_6()
    test_compute_ap_7()
    test_compute_ap_8()
    test_compute_ap_9()
    test_compute_ap_10()
    test_compute_ap_11()
    print('All tests OK')
