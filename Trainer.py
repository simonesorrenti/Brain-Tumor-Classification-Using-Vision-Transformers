import torch
from torch import nn, optim
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, model, optimizer, loss_fn, model_name:str="model", device:str="cpu", patience:int=5):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.device = device
        self.patience = patience

        self.best_accuracy = 0
        self.early_stopping_counter = 0

    def train(self, train_loader, test_loader, epochs):
        train_losses, test_losses, accuracies = [], [], []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            accuracy, test_loss = self.evaluate(test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_model()
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Early Stopping
            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training Epoch", unit="batch")

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0

        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()

        accuracy = correct / len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss

    def save_model(self):
        torch.save(self.model, f"{self.model_name}.pt")
        print(f"Model saved as {self.model_name}.pt")