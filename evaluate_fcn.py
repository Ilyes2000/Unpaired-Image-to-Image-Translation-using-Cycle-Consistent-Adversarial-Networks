import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from PIL import Image
import os, glob, numpy as np, csv

def compute_fcn_score(results_dir, output_dir, output_csv='fcn_scores.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fix deprecated usage of 'pretrained'
    model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT).eval().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    ious, accuracies, mean_class_ious = [], [], []

    # Get the paths of the real_A images from the results directory
    real_paths = sorted(glob.glob(os.path.join(results_dir, 'real_A_epoch_100_test_*.png')))
    if not real_paths:
        print("⚠️ No 'real_A_*' images found in the specified directory.")
        return
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the CSV file
    output_path = os.path.join(output_dir, output_csv)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'IOU', 'Pixel_Accuracy', 'Mean_Class_IOU'])

        for real_path in real_paths:
            filename = os.path.basename(real_path)
            fake_path = os.path.join(results_dir, filename.replace('real_A', 'fake_A'))
            if not os.path.exists(fake_path): continue

            real_img = Image.open(real_path).convert('RGB')
            fake_img = Image.open(fake_path).convert('RGB')

            with torch.no_grad():
                real_pred = model(transform(real_img).unsqueeze(0).to(device))['out'].argmax(1).cpu().numpy()[0]
                fake_pred = model(transform(fake_img).unsqueeze(0).to(device))['out'].argmax(1).cpu().numpy()[0]

            intersection = np.logical_and(real_pred == fake_pred, real_pred > 0)
            union = np.logical_or(real_pred > 0, fake_pred > 0)
            iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
            acc = (real_pred == fake_pred).sum() / real_pred.size

            # Class-wise IOU
            class_ious = []
            num_classes = max(real_pred.max(), fake_pred.max()) + 1
            for cls in range(1, num_classes):
                cls_inter = np.logical_and(real_pred == cls, fake_pred == cls).sum()
                cls_union = np.logical_or(real_pred == cls, fake_pred == cls).sum()
                if cls_union > 0:
                    class_ious.append(cls_inter / cls_union)
            mean_cls_iou = np.mean(class_ious) if class_ious else 0

            ious.append(iou)
            accuracies.append(acc)
            mean_class_ious.append(mean_cls_iou)

            writer.writerow([filename, f"{iou:.4f}", f"{acc:.4f}", f"{mean_cls_iou:.4f}"])

        # Write average
        if ious:
            writer.writerow([
                'AVERAGE',
                f"{np.mean(ious):.4f}",
                f"{np.mean(accuracies):.4f}",
                f"{np.mean(mean_class_ious):.4f}"
            ])
            print(f"\n✅ Metrics saved to: {output_path}")
            print(f"📊 Mean IOU:           {np.mean(ious):.4f}")
            print(f"📊 Pixel Accuracy:     {np.mean(accuracies):.4f}")
            print(f"📊 Mean Class-wise IOU:{np.mean(mean_class_ious):.4f}")
        else:
            print("⚠️ No valid image pairs found.")

if __name__ == '__main__':
    # Update paths: Images are in 'results/', scores will be saved in 'score/'
    compute_fcn_score('results/', 'score/')
