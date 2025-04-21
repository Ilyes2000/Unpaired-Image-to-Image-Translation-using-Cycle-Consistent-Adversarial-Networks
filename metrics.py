import numpy as np

def _fast_hist(true, pred, num_classes):
    """
    Compute confusion matrix for a single image.
    """
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask].astype(int) + pred[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist

def compute_metrics(preds, gts, num_classes):
    """
    Compute semantic segmentation metrics:
        - per-pixel accuracy
        - per-class accuracy
        - mean Intersection over Union (mIoU)

    Parameters
    ----------
    preds : list or numpy.ndarray
        List/array of predicted label images, each shape (H, W).
    gts : list or numpy.ndarray
        List/array of ground truth label images, each shape (H, W).
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    dict
        {
            'pixel_accuracy': float,
            'class_accuracy': numpy.ndarray (shape [num_classes]),
            'mean_iou': float,
            'iou_per_class': numpy.ndarray (shape [num_classes])
        }
    """
    # Initialize confusion matrix
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Accumulate over dataset
    for pred, gt in zip(preds, gts):
        hist += _fast_hist(gt.flatten(), pred.flatten(), num_classes)

    # Pixel accuracy
    pixel_acc = np.diag(hist).sum() / hist.sum()

    # Per-class accuracy
    class_acc = np.diag(hist) / np.maximum(hist.sum(axis=1), 1)

    # Intersection over Union per class
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1)

    # Mean IoU
    mean_iou = np.nanmean(iou)

    return {
        'pixel_accuracy': pixel_acc,
        'class_accuracy': class_acc,
        'mean_iou': mean_iou,
        'iou_per_class': iou
    }

if __name__ == "__main__":

    num_classes = 3
    dummy_pred = [np.random.randint(0, num_classes, (256, 256)) for _ in range(10)]
    dummy_gt   = [np.random.randint(0, num_classes, (256, 256)) for _ in range(10)]
    metrics = compute_metrics(dummy_pred, dummy_gt, num_classes)
    print("Pixel Accuracy:", metrics['pixel_accuracy'])
    print("Per-class Accuracy:", metrics['class_accuracy'])
    print("Mean IoU:", metrics['mean_iou'])
    print("IoU per class:", metrics['iou_per_class'])

