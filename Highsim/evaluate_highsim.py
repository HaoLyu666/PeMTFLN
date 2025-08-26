import torch
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from config import args, DEVICE
from data_loader import HighSimDataset
from model import Encoder, Predictor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, delta=0, trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Evaluate:
    def __init__(self):
        self.early_stopper = EarlyStopping(patience=4)

    def calculate_rmse(self, pred, gt, mask):
        """Calculates masked Root Mean Squared Error."""
        error = torch.pow(pred - gt, 2) * mask
        loss_val = torch.pow(torch.sum(error, dim=0), 0.5)
        counts = torch.pow(torch.sum(mask, dim=0), 0.5)
        return loss_val, counts

    def calculate_mape(self, pred, gt, mask):
        """Calculates masked Mean Absolute Percentage Error."""
        epsilon = 1e-8
        error = torch.abs(pred - gt) / (torch.abs(gt) + epsilon) * 100
        error = error * mask
        error[torch.isnan(error)] = 0  # Replace NaNs with 0
        loss_val = torch.sum(error, dim=0)
        counts = torch.sum(mask, dim=0)
        return loss_val, counts

    def main(self, epoch_num, val=False):
        # --- Model Initialization ---
        encoder = Encoder(args).to(DEVICE)
        predictor = Predictor(args)

        model_path = os.path.join(args['path'], f'epoch{epoch_num}_encoder.tar')
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint not found at {model_path}")
            return
        encoder.load_state_dict(torch.load(model_path, map_location=DEVICE))
        encoder.eval()

        # --- Data Loading ---
        data_split = 'val_data' if val else 'test_data'
        dataset = HighSimDataset('../data/platoons_data_split.npz', data_split, args['in_length'], args['out_length'])
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False,
                                num_workers=args['num_worker'], pin_memory=True, drop_last=True)

        # --- Error Accumulators ---
        total_rmse_vals = torch.zeros(args['veh_num'], args['out_length'], args['out_dim']).to(DEVICE)
        total_rmse_counts = torch.zeros(args['veh_num'], args['out_length'], args['out_dim']).to(DEVICE) + 1e-8
        total_mape_vals = torch.zeros(args['veh_num'], args['out_length'], args['out_dim']).to(DEVICE)
        total_mape_counts = torch.zeros(args['veh_num'], args['out_length'], args['out_dim']).to(DEVICE) + 1e-8

        print(f"\n--- Evaluating Epoch {epoch_num} on {data_split} ---")
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Evaluating"):
                hist, fut, nextv = [d.to(DEVICE) for d in data]

                # --- Model Inference ---
                params, _, _ = encoder(hist)
                initial_state = hist[:, :, -1, 0:2]
                initial_history = hist[:, :, :, 0:2]
                predictions = predictor.forward(params, nextv, initial_state, initial_history)

                # --- Error Calculation ---
                mask = torch.ones_like(fut[:, :, :, :2]).to(DEVICE)
                pred_traj = predictions[:, :, :, :2]
                gt_traj = fut[:, :, :, :2]

                rmse_val, rmse_count = self.calculate_rmse(pred_traj, gt_traj, mask)
                mape_val, mape_count = self.calculate_mape(pred_traj, gt_traj, mask)

                total_rmse_vals += rmse_val
                total_rmse_counts += rmse_count
                total_mape_vals += mape_val
                total_mape_counts += mape_count

        # --- Final Metrics Calculation ---
        final_rmse = total_rmse_vals / total_rmse_counts
        final_mape = total_mape_vals / total_mape_counts

        avg_rmse = torch.mean(final_rmse, dim=[0, 1]).cpu().numpy()
        avg_mape = torch.mean(final_mape, dim=[0, 1]).cpu().numpy()

        print(f"--- Results for Epoch {epoch_num} ---")
        print(f"Average RMSE (Gap, Vel): [{avg_rmse[0]:.4f}, {avg_rmse[1]:.4f}]")
        print(f"Average MAPE (Gap, Vel): [{avg_mape[0]:.4f}, {avg_mape[1]:.4f}]")

        # --- Save Results to CSV ---
        results_path = os.path.join(args['l_path'], "evaluation_results.csv")
        results_df = pd.DataFrame({
            'epoch': [epoch_num] * args['out_length'],
            'pred_step': np.arange(1, args['out_length'] + 1),
            'rmse_gap': torch.mean(final_rmse[:, :, 0], dim=0).cpu().numpy(),
            'rmse_vel': torch.mean(final_rmse[:, :, 1], dim=0).cpu().numpy(),
            'mape_gap': torch.mean(final_mape[:, :, 0], dim=0).cpu().numpy(),
            'mape_vel': torch.mean(final_mape[:, :, 1], dim=0).cpu().numpy()
        })

        if not os.path.isfile(results_path):
            results_df.to_csv(results_path, index=False)
        else:
            results_df.to_csv(results_path, mode='a', header=False, index=False)
        print(f"Evaluation results saved to {results_path}")

        # --- Early Stopping Check ---
        if val:
            avg_val_loss = torch.mean(final_rmse).item()
            self.early_stopper(avg_val_loss)
            return self.early_stopper.early_stop
        return False


if __name__ == '__main__':
    # --- To run evaluation independently ---
    epoch_to_evaluate = 20  # Specify the epoch number you want to test
    print(f"Running standalone evaluation for epoch: {epoch_to_evaluate}")
    evaluate = Evaluate()
    evaluate.main(epoch_num=epoch_to_evaluate, val=False)
