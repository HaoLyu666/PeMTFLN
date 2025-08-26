import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import time

# Local imports
from config import args, DEVICE
from data_loader import HighSimDataset
from model import Encoder, Predictor
from evaluate_highsim import Evaluate
from DWA import DWA


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_distribution_loss(params):
    """
    Applies a soft constraint to the physical parameters to encourage them
    to stay within a reasonable range.
    """
    p1, p2, p3 = params[..., 0], params[..., 1], params[..., 2]
    p_min, p_max = 0.02, 5.0

    # Penalize values outside the [p_min, p_max] range
    p1_penalty = torch.mean(torch.relu(p_min - p1) + torch.relu(p1 - p_max))
    p2_penalty = torch.mean(torch.relu(p_min - p2) + torch.relu(p2 - p_max))
    p3_penalty = torch.mean(torch.relu(p_min - p3) + torch.relu(p3 - p_max))

    return p1_penalty + p2_penalty + p3_penalty


def save_model(epoch_num, model_encoder):
    """Saves the model checkpoint."""
    checkpoint_dir = args['path']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, f'epoch{epoch_num}_encoder.tar')
    torch.save(model_encoder.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    # --- Initialization ---
    evaluate = Evaluate()
    dwa = DWA(args)  # Dynamic Weight Averaging for multi-task loss
    encoder = Encoder(args).to(DEVICE)
    predictor = Predictor(args)
    print(f"Total trainable parameters in Encoder: {count_parameters(encoder):,}")

    # --- Data Loading ---
    train_dataset = HighSimDataset('../data/platoons_data_split.npz', 'train_data', args['in_length'],
                                   args['out_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                  num_workers=args['num_worker'], pin_memory=True, drop_last=True)

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(encoder.parameters(), lr=args['learning_rate'])
    scheduler = ExponentialLR(optimizer, gamma=args['gamma'])

    train_loss_buffer = torch.zeros(args['num_task'] + 1, args['epoch'])

    # --- Training Loop ---
    for epoch in range(args['epoch']):
        encoder.train()
        print(f"\n--- Epoch: {epoch + 1}/{args['epoch']}, LR: {optimizer.param_groups[0]['lr']:.6f} ---")

        epoch_loss_gap, epoch_loss_velocity, epoch_kl_div, epoch_total_loss = 0.0, 0.0, 0.0, 0.0

        for data in tqdm(train_dataloader, desc="Training"):
            hist, fut, nextv = [d.to(DEVICE) for d in data]

            # --- Forward Pass ---
            params, mu, log_var = encoder(hist)
            initial_state = hist[:, :, -1, 0:2]
            initial_history = hist[:, :, :, 0:2]
            predictions = predictor.forward(params, nextv, initial_state, initial_history)

            # --- Loss Calculation ---
            loss_fn = torch.nn.MSELoss()
            # 1. Trajectory Prediction Loss (MSE)
            loss_gap = loss_fn(predictions[:, :, :, 0], fut[:, :, :, 0])
            loss_velocity = loss_fn(predictions[:, :, :, 1], fut[:, :, :, 1])

            # 2. KL Divergence Loss (VAE Regularization)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # 3. Dynamic Weight Averaging for multi-task losses
            trajectory_losses = [loss_gap, loss_velocity]
            batch_weights = dwa.backward(trajectory_losses, epoch, train_loss_buffer[:args['num_task'], :])

            # 4. Parameter Distribution Regularization Loss
            param_loss = param_distribution_loss(params)

            # --- Combine Losses ---
            weighted_trajectory_loss = sum(loss * weight for loss, weight in zip(trajectory_losses, batch_weights))
            param_weight = torch.clamp(0.1 * weighted_trajectory_loss.item() / (param_loss.item() + 1e-8), min=0.05,
                                       max=0.2)
            total_loss = weighted_trajectory_loss + 0.0025 * kl_divergence + param_weight * param_loss

            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10)
            optimizer.step()

            # Accumulate epoch losses for logging
            epoch_loss_gap += loss_gap.item()
            epoch_loss_velocity += loss_velocity.item()
            epoch_kl_div += kl_divergence.item()
            epoch_total_loss += total_loss.item()

        # --- End of Epoch Logging ---
        avg_total_loss = epoch_total_loss / len(train_dataloader)
        avg_loss_gap = epoch_loss_gap / len(train_dataloader)
        avg_loss_velocity = epoch_loss_velocity / len(train_dataloader)

        print(
            f"Avg Epoch Loss: {avg_total_loss:.4f} | Gap Loss: {avg_loss_gap:.4f} | Vel Loss: {avg_loss_velocity:.4f}")
        print(f"DWA Weights: Gap={batch_weights[0]:.2f}, Vel={batch_weights[1]:.2f}")

        train_loss_buffer[:, epoch] = torch.tensor([avg_loss_gap, avg_loss_velocity, avg_total_loss])

        # --- Save Model and Evaluate ---
        save_model(epoch + 1, encoder)
        early_stop = evaluate.main(epoch + 1, val=True)
        if early_stop:
            print("Early stopping triggered!")
            break

        scheduler.step()

    # --- Final: Save Loss Log ---
    csv_path = os.path.join(args['l_path'], "train_loss_log.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'gap_loss', 'velocity_loss', 'total_loss'])
        for i in range(epoch + 1):
            writer.writerow([i + 1] + train_loss_buffer[:, i].tolist())
    print(f"Training loss log saved to {csv_path}")


if __name__ == '__main__':
    main()
