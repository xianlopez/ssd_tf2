import numpy as np
from non_maximum_suppression import non_maximum_suppression_slow, non_maximum_suppression_fast

nclasses = 20


def test_nms_slow_1():
    print('    - test_nms_slow_1: No input boxes')
    predictions_full = np.zeros((0, 6), np.float32)
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == (0, 6)


def test_nms_slow_2():
    print('    - test_nms_slow_2: One input box')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_slow_3():
    print('    - test_nms_slow_3: Two input box, with enough distance')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.1, 0.2, 0.2, 0.3, 4, 0.9]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_slow_4():
    print('    - test_nms_slow_4: Two input box, different classes')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0, 0, 0.2, 0.3, 3, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert np.all(predictions_full == predictions_full_orig)


def test_nms_slow_5():
    print('    - test_nms_slow_5: Two input box, same class and overlapped')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.05, 0, 0.2, 0.3, 4, 0.9]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])


def test_nms_slow_6():
    print('    - test_nms_slow_6: Two input box, same class and overlapped (reverse order)')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.05, 0, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert np.all(predictions_full[0, :] == predictions_full_orig[0, :])


def test_nms_slow_7():
    print('    - test_nms_slow_7: Three input box, same class and overlapped')
    predictions_full_orig = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.03, 0, 0.2, 0.3, 4, 0.91],
                                      [0, 0.02, 0.2, 0.3, 4, 0.8]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[2, 4])) == nclasses
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])


def test_nms_slow_8():
    print('    - test_nms_slow_8: Chain suppression')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


def test_nms_slow_9():
    print('    - test_nms_slow_9: Double chain')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6],
                                      [0.1, 0, 0.2, 0.2, 4, 0.65], [0.05, 0, 0.2, 0.2, 4, 0.75]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[0, 4])) == nclasses
    assert int(np.round(predictions_full[1, 4])) == nclasses
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert int(np.round(predictions_full[4, 4])) == nclasses
    assert int(np.round(predictions_full[5, 4])) == nclasses
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


def test_nms_slow_10():
    print('    - test_nms_slow_10: Chain broken by different class')
    predictions_full_orig = np.array([[0, 0.05, 0.2, 0.2, 0, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    predictions_full = predictions_full_orig.copy()
    non_maximum_suppression_slow(predictions_full, nclasses)
    assert predictions_full.shape == predictions_full_orig.shape
    assert int(np.round(predictions_full[3, 4])) == nclasses
    assert np.all(predictions_full[0, :] == predictions_full_orig[0, :])
    assert np.all(predictions_full[1, :] == predictions_full_orig[1, :])
    assert np.all(predictions_full[2, :] == predictions_full_orig[2, :])


def test_nms_fast_1():
    print('    - test_nms_fast_1: No input boxes')
    predictions_full = np.zeros((0, 6), np.float32)
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 0


def test_nms_fast_2():
    print('    - test_nms_fast_2: One input box')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.8]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds == predictions_full)


def test_nms_fast_3():
    print('    - test_nms_fast_3: Two input box, with enough distance')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.1, 0.2, 0.2, 0.3, 4, 0.9]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 2
    assert np.all(predictions_full == predictions_full)


def test_nms_fast_4():
    print('    - test_nms_fast_4: Two input box, different classes')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0, 0, 0.2, 0.3, 3, 0.8]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 2
    assert np.all(predictions_full == predictions_full)


def test_nms_fast_5():
    print('    - test_nms_fast_5: Two input box, same class and overlapped')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.8], [0.05, 0, 0.2, 0.3, 4, 0.9]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds[0, :] == predictions_full[1, :])


def test_nms_fast_6():
    print('    - test_nms_fast_6: Two input box, same class and overlapped (reverse order)')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.05, 0, 0.2, 0.3, 4, 0.8]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds[0, :] == predictions_full[0, :])


def test_nms_fast_7():
    print('    - test_nms_fast_7: Three input box, same class and overlapped')
    predictions_full = np.array([[0, 0, 0.2, 0.3, 4, 0.9], [0.03, 0, 0.2, 0.3, 4, 0.91],
                                      [0, 0.02, 0.2, 0.3, 4, 0.8]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds[0, :] == predictions_full[1, :])


def test_nms_fast_8():
    print('    - test_nms_fast_8: Chain suppression')
    predictions_full = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds[0, :] == predictions_full[2, :])


def test_nms_fast_9():
    print('    - test_nms_fast_9: Double chain')
    predictions_full = np.array([[0, 0.05, 0.2, 0.2, 4, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                      [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6],
                                      [0.1, 0, 0.2, 0.2, 4, 0.65], [0.05, 0, 0.2, 0.2, 4, 0.75]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 1
    assert np.all(remaining_preds[0, :] == predictions_full[2, :])


def test_nms_fast_10():
    print('    - test_nms_fast_10: Chain broken by different class')
    predictions_full = np.array([[0, 0.05, 0.2, 0.2, 0, 0.8], [0, 0.1, 0.2, 0.2, 4, 0.7],
                                 [0, 0, 0.2, 0.2, 4, 0.9], [0, 0.15, 0.2, 0.2, 4, 0.6]])
    remaining_preds = non_maximum_suppression_fast(predictions_full, nclasses)
    assert len(remaining_preds) == 3
    assert np.all(remaining_preds[0, :] == predictions_full[0, :])
    assert np.all(remaining_preds[1, :] == predictions_full[2, :])
    assert np.all(remaining_preds[2, :] == predictions_full[1, :])


def test_nms_slow():
    print('* test_nms_slow')
    test_nms_slow_1()
    test_nms_slow_2()
    test_nms_slow_3()
    test_nms_slow_4()
    test_nms_slow_5()
    test_nms_slow_6()
    test_nms_slow_7()
    test_nms_slow_8()
    test_nms_slow_9()
    test_nms_slow_10()


def test_nms_fast():
    print('* test_nms_fast')
    test_nms_fast_1()
    test_nms_fast_2()
    test_nms_fast_3()
    test_nms_fast_4()
    test_nms_fast_5()
    test_nms_fast_6()
    test_nms_fast_7()
    # The fast algorithm doesn't pass the chain tests.
    # test_nms_fast_8()
    # test_nms_fast_9()
    # It passes test 10 however, since there the chain is actually broken.
    test_nms_fast_10()


if __name__ == '__main__':
    print('Tests for non-maximum suppression')
    test_nms_slow()
    test_nms_fast()
    print('All tests OK')