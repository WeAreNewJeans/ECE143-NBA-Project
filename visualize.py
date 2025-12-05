import torch
import numpy as np
@torch.no_grad()
def compute_player_slot_importance(model, dataloader, num_samples=2000, device="cpu"):
        model.eval()
        num_players = 12
        feats_per_player = 20
    
        importance_sum = numpy.zeros(num_players, dtype=np.float64)
        total_seen = 0
    
        for batch in dataloader:
            (
                team1_features, team2_features,
                team1_home, team2_home,
                team1_winrate, team2_winrate,
                team1_season_stats, team2_season_stats,
                labels
            ) = batch
    
            batch_size = team1_features.size(0)
            if batch_size == 0:
                continue
    
    
            if total_seen >= num_samples:
                break
            if total_seen + batch_size > num_samples:
                keep = num_samples - total_seen
                team1_features     = team1_features[:keep]
                team2_features     = team2_features[:keep]
                team1_home         = team1_home[:keep]
                team2_home         = team2_home[:keep]
                team1_winrate      = team1_winrate[:keep]
                team2_winrate      = team2_winrate[:keep]
                team1_season_stats = team1_season_stats[:keep]
                team2_season_stats = team2_season_stats[:keep]
                labels             = labels[:keep]
                batch_size = keep
    
            team1_features     = team1_features.to(device)
            team2_features     = team2_features.to(device)
            team1_home         = team1_home.to(device)
            team2_home         = team2_home.to(device)
            team1_winrate      = team1_winrate.to(device)
            team2_winrate      = team2_winrate.to(device)
            team1_season_stats = team1_season_stats.to(device)
            team2_season_stats = team2_season_stats.to(device)
            baseline_logits = model(
                team1_features, team2_features,
                team1_home, team2_home,
                team1_winrate, team2_winrate,
                team1_season_stats, team2_season_stats
            ).detach()
    
            for p in range(num_players):
                start = p * feats_per_player
                end = (p + 1) * feats_per_player
    
                perturbed_team1 = team1_features.clone()
    
                perm = torch.randperm(batch_size, device=device)
                perturbed_team1[:, start:end] = team1_features[perm, start:end]
    
                logits_perturbed = model(
                    perturbed_team1, team2_features,
                    team1_home, team2_home,
                    team1_winrate, team2_winrate,
                    team1_season_stats, team2_season_stats
                ).detach()
    
                diff = (logits_perturbed - baseline_logits).abs().mean().item()
                importance_sum[p] += diff * batch_size
    
            total_seen += batch_size
    
        importance = importance_sum / total_seen
        return importance

device = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def feature_importance_for_player_slot(model, batch, slot):
    (
        t1, t2, 
        t1_home, t2_home,
        t1_winrate, t2_winrate,
        t1_season, t2_season,
        labels
    ) = batch
    
    B = t1.size(0)
    if B < 2:
        return np.zeros(20)

    # baseline
    baseline = model(
        t1.to(device), t2.to(device),
        t1_home.to(device), t2_home.to(device),
        t1_winrate.to(device), t2_winrate.to(device),
        t1_season.to(device), t2_season.to(device)
    ).detach()

    importance = np.zeros(20)
    start = slot * 20
    end   = (slot + 1) * 20

    for f in range(20):
        x = t1.clone()
        perm = torch.randperm(B)
        x[:, start + f] = t1[perm, start + f]

        out = model(
            x.to(device), t2.to(device),
            t1_home.to(device), t2_home.to(device),
            t1_winrate.to(device), t2_winrate.to(device),
            t1_season.to(device), t2_season.to(device),
        ).detach()

        delta = (out - baseline).abs().mean().item()
        importance[f] = delta

    return importance

def gradient_feature_importance(model, batch, slot):
    (
        t1, t2, 
        t1_home, t2_home,
        t1_winrate, t2_winrate,
        t1_season, t2_season,
        labels
    ) = batch
    
    x = t1.clone()
    x.requires_grad = True  
    
    logits = model(
        x.to(device), t2.to(device),
        t1_home.to(device), t2_home.to(device),
        t1_winrate.to(device), t2_winrate.to(device),
        t1_season.to(device), t2_season.to(device),
    )
    
    logit = logits.mean()  
    logit.backward()

    grads = x.grad.detach().cpu().numpy()
    start = slot * 20
    end   = (slot + 1) * 20

    abs_grad = np.abs(grads[:, start:end]).mean(axis=0)
    return abs_grad / abs_grad.sum()

def integrated_gradients(model, batch, slot, steps=30):
    (
        t1, t2, 
        t1_home, t2_home,
        t1_winrate, t2_winrate,
        t1_season, t2_season,
        labels
    ) = batch
    
    x = t1[:1].clone().to(device)  
    baseline = torch.zeros_like(x).to(device)

    scaled_inputs = [
        baseline + (float(i) / steps) * (x - baseline)
        for i in range(steps + 1)
    ]

    total_grad = torch.zeros_like(x)

    for s_in in scaled_inputs:
        s_in.requires_grad = True
        out = model(
            s_in, t2[:1].to(device),
            t1_home[:1].to(device), t2_home[:1].to(device),
            t1_winrate[:1].to(device), t2_winrate[:1].to(device),
            t1_season[:1].to(device), t2_season[:1].to(device),
        )
        out.backward()
        total_grad += s_in.grad

    avg_grad = total_grad / steps
    ig = (x - baseline) * avg_grad
    ig = ig.detach().cpu().numpy()[0]

    start = slot * 20
    end   = (slot + 1) * 20

    vals = ig[start:end]
    vals = vals / (np.sum(np.abs(vals)) + 1e-9)
    return vals
