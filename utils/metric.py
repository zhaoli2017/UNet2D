import numpy as np

def dice_2d(y_true, y_pred):
    assert len(y_pred.shape) == 3, 'the input for dice_2d should has 3 dims'

    c1_pd = np.equal(y_pred, 1)
    c1_gt = np.equal(y_true, 1)
    c1_ovlp = np.logical_and(c1_pd, c1_gt)
    c1_dice = 2 * c1_ovlp.sum(1).sum(1) / (c1_pd.sum(1).sum(1) + c1_gt.sum(1).sum(1) + 1e-5)

    c2_pd = np.equal(y_pred, 2)
    c2_gt = np.equal(y_true, 2)
    c2_ovlp = np.logical_and(c2_pd, c2_gt)
    c2_dice = 2 * c2_ovlp.sum(1).sum(1) / (c2_pd.sum(1).sum(1) + c2_gt.sum(1).sum(1) + 1e-5)


    return c1_dice, c2_dice

