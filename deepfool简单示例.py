import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool attack algorithm optimized with a correct stop condition.
    """
    image = image.unsqueeze(0)  # Add batch dimension
    image.requires_grad = True
    output = model(image)
    _, initial_pred = torch.max(output.data, 1)

    total_perturbation = torch.zeros_like(image)
    for i in range(max_iter):
        image.grad = None  # Reset gradients
        output = model(image)
        _, current_pred = torch.max(output.data, 1)
        if current_pred != initial_pred:
            # Stop if the prediction has changed
            break

        output[0, initial_pred].backward(retain_graph=True)
        grad_orig = image.grad.data.clone()

        pert = None
        for k in range(num_classes):
            if k == initial_pred:
                continue
            zero_gradients(image)
            output[0, k].backward(retain_graph=True)
            cur_grad = image.grad.data.clone()
            # Compute perturbation
            w_k = cur_grad - grad_orig
            f_k = (output[0, k] - output[0, initial_pred]).data
            pert_k = abs(f_k) / w_k.norm()
            if pert is None or pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / w.norm()
        total_perturbation += (1 + overshoot) * r_i
        image.data = image.data + (1 + overshoot) * r_i

    image = image.squeeze(0)  # Remove batch dimension
    total_perturbation = total_perturbation.squeeze(0)
    return image, i + 1, total_perturbation.norm().item()  # i + 1 to account for the zero index

def zero_gradients(x):
    if x.grad is not None:
        x.grad.data.zero_()

# Set device to MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load a pre-trained model
model = models.mobilenet_v2(pretrained=True).to(device)
model.eval()

# Load an image and preprocess it
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert image to RGB
    transforms.ToTensor(),
])
img_path = "./dog.png"  # Update this path accordingly
img = Image.open(img_path)
img = transform(img).to(device)

# Perform the attack
perturbed_image, iter_num, total_perturbation = deepfool(img, model)

print(f"Number of iterations: {iter_num}")
print(f"Total perturbation norm: {total_perturbation}")
