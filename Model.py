import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Function

from torch.utils.data import DataLoader, Dataset

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradReverse(Function):

    @staticmethod

    def forward(ctx, x, lambd):

        ctx.lambd = lambd

        return x.view_as(x)

    @staticmethod

    def backward(ctx, grad_output):

        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):

    return GradReverse.apply(x, lambd)

class CNN_DANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(3, 32, 5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(48*8*8, 100),

            nn.ReLU()

        )

        self.class_classifier = nn.Sequential(

            nn.Linear(100, 50),

            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.domain_classifier = nn.Sequential(

            nn.Linear(100, 50),

            nn.ReLU(),

            nn.Linear(50, 2)  

        )

   

    def forward(self, x, lambd=1.0):

        feat = self.feature(x)

        class_output = self.class_classifier(feat)

        domain_output = self.domain_classifier(grad_reverse(feat, lambd))

        return class_output, domain_output

def train(model, source_loader, target_loader, optimizer, class_criterion, domain_criterion, epochs=10):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        total_class_loss = 0

        total_domain_loss = 0

        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):

            source_data, source_labels = source_data.to(device), source_labels.to(device)

            target_data = target_data.to(device)

           

            optimizer.zero_grad()

           

            class_out_s, domain_out_s = model(source_data, lambd=1.0)

            class_loss = class_criterion(class_out_s, source_labels)

            domain_label_s = torch.zeros(len(source_data), dtype=torch.long).to(device)

    

            _, domain_out_t = model(target_data, lambd=1.0)

            domain_label_t = torch.ones(len(target_data), dtype=torch.long).to(device)

           

            domain_out = torch.cat([domain_out_s, domain_out_t], dim=0)

            domain_labels = torch.cat([domain_label_s, domain_label_t], dim=0)

            domain_loss = domain_criterion(domain_out, domain_labels)

         

            loss = class_loss + domain_loss

            loss.backward()

            optimizer.step()

           

            total_loss += loss.item()

            total_class_loss += class_loss.item()

            total_domain_loss += domain_loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}, "

              f"Class Loss: {total_class_loss:.4f}, Domain Loss: {total_domain_loss:.4f}")

def evaluate(model, loader, domain_name="Domain"):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for data, labels in loader:

            data, labels = data.to(device), labels.to(device)

            class_out, _ = model(data, lambd=0)

            preds = class_out.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    acc = 100 * correct / total

    print(f"{domain_name} Accuracy: {acc:.2f}%")

    return acc

transform = transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor(),

    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

])

def load_image_frame(filepath):

    img = Image.open(filepath).convert('RGB')

    img_t = transform(img).unsqueeze(0)  

    return img_t.to(device)

def predict_frame(model, img_tensor):

    model.eval()

    with torch.no_grad():

        class_logits, _ = model(img_tensor, lambd=0)

        probs = torch.softmax(class_logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        prob = probs[0, pred].item()

    classes = ['Clear', 'Degraded']

    print(f"Prediction: {classes[pred]} with confidence {prob*100:.2f}%")

    return classes[pred], prob

class SyntheticADASDataset(Dataset):

    def __init__(self, n_samples=1000, domain='source'):

        super().__init__()

        self.n_samples = n_samples

        self.domain = domain

        self.data = torch.randn(n_samples, 3, 32, 32)

        if domain == 'source':

            self.labels = torch.randint(0, 2, (n_samples,))

            self.labels[:int(0.7*n_samples)] = 0

        else:

            self.labels = torch.randint(0, 2, (n_samples,))

            self.labels[:int(0.7*n_samples)] = 1

    def __len__(self):

        return self.n_samples

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]

if __name__ == "__main__":

    source_dataset = SyntheticADASDataset(domain='source')

    target_dataset = SyntheticADASDataset(domain='target')

   

    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)

    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

    model = CNN_DANN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    class_criterion = nn.CrossEntropyLoss()

    domain_criterion = nn.CrossEntropyLoss()

    print("Training DANN model...")

    train(model, source_loader, target_loader, optimizer, class_criterion, domain_criterion, epochs=10)

    print("\nEvaluating on source domain (clear footage):")

    evaluate(model, source_loader, "Source (Clear)")

    print("\nEvaluating on target domain (degraded footage):")

    evaluate(model, target_loader, "Target (Degraded)")

    img_path = input("\nEnter the path to a road frame image for classification: ")

    img_tensor = load_image_frame(img_path)

    predict_frame(model, img_tensor)
