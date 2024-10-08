import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import optuna  # Added for hyperparameter tuning
import pandas as pd  # For saving results
import seaborn as sns  # For enhanced visualizations
from src.dataloader_fer import loadBatches
from src.model_fer import loadModel
from utils import checkDir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Objective function for Optuna
def objective(trial):
    # Hyperparameter space definition
    epochs = 10
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    is_weight = trial.suggest_categorical('weight', [True, False])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    grad_clip_value = trial.suggest_float('grad_clip_value', 0.1, 0.5)

    # Print the current trial's hyperparameter values
    print(f"Trial {trial.number}:")
    print(f" - Learning rate: {learning_rate}")
    print(f" - Optimizer: {optimizer_name}")
    print(f" - Batch size: {batch_size}")
    print(f" - Use weight: {is_weight}")
    print(f" - Weight decay: {weight_decay}")
    print(f" - Gradient clipping: {grad_clip_value}\n")

    # Load data and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batches, valid_batches, weight = loadBatches(batch_size=batch_size)

    if is_weight:
        weight = torch.tensor(weight, dtype=torch.float).to(device)
    else:
        weight = None

    model = loadModel(num_classes=7)
    model = model.to(device)

    # Define optimizer and loss function
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Training loop with loss tracking
    def train(model, train_batches, valid_batches, epochs, opt, cri, device):
        train_losses, valid_losses = [], []
        min_val_loss = torch.inf

        for epoch in range(epochs):
            train_loss, valid_loss = 0., 0.
            model.train()

            tr_bar = tqdm(train_batches, desc=f'epoch {epoch}')
            for images, labels in tr_bar:
                images = images.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                outputs = model(images)
                loss = cri(outputs, labels)
                loss.backward()
                
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

                opt.step()

                train_loss += loss.item() * images.size(0)
                tr_bar.postfix = f"loss: {loss.item():.3f}"
                torch.cuda.empty_cache()

            train_loss /= len(train_batches.sampler)
            train_losses.append(train_loss)

            model.eval()
            val_bar = tqdm(valid_batches, desc=f'epoch {epoch}')
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = cri(outputs, labels)
                valid_loss += loss.item() * images.size(0)

            valid_loss /= len(valid_batches.sampler)
            valid_losses.append(valid_loss)

            # Save the model with the best validation loss
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss

        return min_val_loss, train_losses, valid_losses

    # Call the train function and return validation loss
    validation_loss, train_losses, valid_losses = train(model, train_batches, valid_batches, epochs, optimizer, criterion, device)
    
    # Log the trial results
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("valid_losses", valid_losses)
    trial.report(validation_loss, step=epochs)

    return validation_loss

# Function to visualize and save Optuna results
def plot_optuna_results(study):
    # Save trial information as a dataframe
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    
    # Save trial information to CSV
    trials_df.to_csv('optuna_trials.csv', index=False)

    # Plot optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.savefig("optimization_history.png")
    plt.show()

    # Plot parameter importance
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.savefig("param_importances.png")
    plt.show()

    # Plot parameter vs performance (loss) distribution
    for param in study.best_trial.params.keys():
        if param in study.trials_dataframe(attrs=('params',)).columns:  # 파라미터가 실제 데이터프레임에 존재하는지 확인
            optuna.visualization.matplotlib.plot_slice(study, param)
            plt.title(f"Performance based on {param}")
            plt.savefig(f"{param}_performance.png")
            plt.show()
        else:
            print(f"Parameter {param} does not exist in the study trials dataframe.")

    # Correlation plot for all hyperparameters
    trials_df['params_optimizer'] = trials_df['params_optimizer'].map({'Adam': 0, 'SGD': 1})  # Adam -> 0, SGD -> 1
    trials_df.drop(['state', 'number'], axis=1, inplace=True)
    correlation_matrix = trials_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Hyperparameter Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.show()

# Function to plot training vs validation loss curves for the best trial
def plot_loss_curves(study):
    best_trial = study.best_trial
    train_losses = best_trial.user_attrs["train_losses"]
    valid_losses = best_trial.user_attrs["valid_losses"]

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig("train_vs_valid_loss.png")
    plt.show()

# Optuna Study setup
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3000)

    # Log best trial
    print(f'Best trial: {study.best_trial.params}')

    # Plot and save results
    plot_optuna_results(study)
    plot_loss_curves(study)
