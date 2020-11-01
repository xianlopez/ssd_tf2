import numpy as np
from non_maximum_suppression import non_maximum_suppression

nclasses = 20


def test_nms_1():
    print('    - test_nms_1: No input boxes')
    predictions_full = np.zeros((0, 6), np.float32)
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == (0, 6)


def test_nms_2():
    print('    - test_nms_2: One input box')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_3():
    print('    - test_nms_3: Two input box, with enough distance')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.1, 0.2, 0.2, 0.3, 4, 0.9]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_4():
    print('    - test_nms_4: Two input box, different classes')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0, 0, 0.2, 0.3, 3, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_5():
    print('    - test_nms_5: Two input box, same class and overlapped')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.05, 0, 0.2, 0.3, 4, 0.9]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])


def test_nms_6():
    print('    - test_nms_6: Two input box, same class and overlapped (reverse order)')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.05, 0, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert np.all(predictions_full[0, :] == predictions_full_orig[0, :])


def test_nms_7():
    print('    - test_nms_7: Three input box, same class and overlapped')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.03, 0, 0.2, 0.3, 4, 0.91],
                                      [0, 0.02, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[2, 4])) == nclasses
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])


def test_nms_8():
    print('    - test_nms_8: Chain suppression')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


def test_nms_9():
    print('    - test_nms_9: Double chain')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6],
                                      [0.1, 0, 0.2, 0.2, 4, 0.65], [0.05, 0, 0.2, 0.2, 4, 0.75]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert int(np.round(predictions_full[4, 4])) == nclasses
    assert int(np.round(predictions_full[5, 4])) == nclasses
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


def test_nms_10():
    print('    - test_nms_10: Chain broken by different class')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 0, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert np.all(predictions_full[0, :] == predictions_full_orig[0, :])
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


if __name__ == '__main__':
    print('Tests for non-maximum suppression')
    test_nms_1()
    test_nms_2()
    test_nms_3()
    test_nms_4()
    test_nms_5()
    test_nms_6()
    test_nms_7()
    test_nms_8()
    test_nms_9()
    test_nms_10()
    print('All tests OK')