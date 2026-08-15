from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import multiprocessing
from torch.optim.lr_scheduler import LinearLR
from early_stopping import EarlyStopping
from RiskmapDataset import RiskMapTrain, RiskMapValidation

from DiffResCrime import STFusionNet, DiffRes

if __name__ == '__main__':
    n_epoch = 1000
    batch_size = 16
    n_T = 15
    # n_T = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_feat = 256
    lrate = 1e-4
    drop_prob = 0.1

    fusion_net = STFusionNet(
        in_channels=1,
        n_feat=n_feat,
        context_out_channels=64,  
        poi_channels=13,        
        # x_dim=256       
        x_dim=2048
    )

    diffres_model = DiffRes(
        backbone=fusion_net,
        n_T=n_T,
        device=device,
        drop_prob=drop_prob,
        gamma=1.0,
        lambda_start=0.001, 
        lambda_end=0.999,
        power=2.0
    ).to(device)

    num_workers = min(10, multiprocessing.cpu_count())

    dataset = RiskMapTrain()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, drop_last=True)
    
    datasetVal = RiskMapValidation()
    dataloaderVal = DataLoader(datasetVal, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, drop_last=False)
    
    optim = torch.optim.Adam(diffres_model.parameters(), lr=lrate)

    scheduler = LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=n_epoch)

    early_stopping = EarlyStopping(
        save_path="./Result/Output", 
        patience=500,
        verbose=True
    )

    for ep in range(n_epoch):
        print(f'Epoch {ep}/{n_epoch}')
        diffres_model.train()
        
        pbar = tqdm(dataloader)
        loss_ema = None
        
        for batch in pbar:
            optim.zero_grad()

            label = batch["riskmap_obs"]
            cond_map = batch["map"]
            cond_satellite = batch["satellite"]
            history = batch['riskmap_his']
            poi = batch['poi']

            cond_combined = torch.cat([cond_satellite, cond_map], dim=-1)  # [B, H, W, 6]

            label = label.to(device)
            cond_combined = cond_combined.to(device)
            history = history.to(device)
            poi = poi.to(device)

            loss = diffres_model(label, cond_combined, history, poi)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(diffres_model.parameters(), 1.0)
            optim.step()

            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Train Loss: {loss_ema:.4f}")

        diffres_model.eval()
        val_loss_ema = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in dataloaderVal:
                label = batch["riskmap_obs"].to(device)
                cond_map = batch["map"].to(device)
                cond_satellite = batch["satellite"].to(device)
                history = batch['riskmap_his'].to(device)
                poi = batch['poi'].to(device)
                cond_combined = torch.cat([cond_satellite, cond_map], dim=-1)
                
                loss = diffres_model(label, cond_combined, history, poi)
                val_loss_ema += loss.item() * label.size(0)
                val_samples += label.size(0)
        
        avg_val_loss = val_loss_ema / val_samples
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {ep}/{n_epoch} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        scheduler.step()

        early_stopping(avg_val_loss, diffres_model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    torch.save(diffres_model.state_dict(), "./Result/Output/NY.pth")
    print("Training completed and model saved")
