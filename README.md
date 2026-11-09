# Model Inversion Attack on a Digit Classifier

Project for Data Analytics in Cybersecurity (UTS).

## What it does
This project trains a small convolutional neural network (TargetCNN) on the scikit-learn 8x8 handwritten digits dataset, then runs a gradient-based model inversion attack against it (Fredrikson et al., 2015 style). The attack uses only the model's outputs and gradients to reconstruct what each digit class looks like, showing how a trained model can leak information about its training data.

It also tests three defences against the attack:
1. Output perturbation (Gaussian noise on logits)
2. Confidence masking (top-k / rounding outputs)
3. Differential privacy (DP-SGD via Opacus, if installed)

## Results
- Target model overall accuracy: 68.9%
- `outputs/reconstructions.png` shows the images the attack recovered for each digit class
- `outputs/mse_summary.png` compares the reconstructions with the real class averages (mean squared error)
- `outputs/confusion_matrices.png` shows where the model gets digits confused

## How to run
    pip install torch scikit-learn matplotlib opacus
    python model_inversion_attack.py

## What I learned
Models can reveal private patterns from their training data through normal predictions, so defences such as limiting output detail, adding noise and differential privacy matter when models are deployed.
