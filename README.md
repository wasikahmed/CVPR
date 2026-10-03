# CVPR: scene classification with CNNs

Course project for Computer Vision and Pattern Recognition at AIUB. Two CNNs trained from scratch in PyTorch on the Intel Image Classification dataset (6 classes: buildings, forest, glacier, mountain, sea, street; 14,034 training and 3,000 test images, resized to 150×150).

| Model | Test accuracy |
|---|---|
| Base CNN | 85.0% |
| Regularized CNN (batch norm, dropout, augmentation) | 88.6% |

The base model overfits: its training accuracy reaches 99.9%. Regularization keeps the training and test curves close. Both models were trained for 30 epochs with Adam and a step learning-rate schedule.

## Files

- `scene_classification_cnn.ipynb` — data loading, both models, training, evaluation and sample predictions
- `BaseCNN_best.pth`, `RegularizedCNN_best.pth` — trained weights
