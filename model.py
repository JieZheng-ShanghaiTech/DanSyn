from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.autograd import Function
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DANNModule(nn.Module):
    def __init__(self, feature_dim: int = 512, num_domains: int = 2, lambda_adv: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_domains),
        )

    def forward(self, features: torch.Tensor, domain_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reversed_features = self.grl(features)
        logits = self.domain_discriminator(reversed_features)
        loss = F.cross_entropy(logits, domain_labels)
        return loss, logits


class DrugEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_length: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_length:
            input_ids = input_ids[:, : self.max_seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_seq_length]
            seq_len = self.max_seq_length

        word_embeds = self.word_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = self.position_embedding(position_ids)

        encoded = self.layer_norm(word_embeds + position_embeds)
        encoded = self.dropout(encoded)
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        return self.transformer(encoded, src_key_padding_mask=padding_mask)


class CellResidualAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.final_relu = nn.ReLU()

    def forward(self, omics_latent: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(omics_latent)
        out = self.res_block(x) + x
        return self.final_relu(out)


class CrossAttentionFusion(nn.Module):
    def __init__(self, drug_dim: int, cell_dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(cell_dim, drug_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=drug_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(drug_dim)
        self.ffn = nn.Sequential(
            nn.Linear(drug_dim, drug_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(drug_dim * 2, drug_dim),
        )
        self.ffn_norm = nn.LayerNorm(drug_dim)
        self.post_proj = nn.Sequential(
            nn.Linear(drug_dim + cell_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, drug_seq: torch.Tensor, drug_mask: torch.Tensor, cell_vec: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(cell_vec).unsqueeze(1)

        key_padding_mask = None
        all_pad = None
        if drug_mask is not None:
            key_padding_mask = drug_mask == 0
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False

        attn_output, _ = self.cross_attn(
            query=query,
            key=drug_seq,
            value=drug_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = attn_output.squeeze(1)
        query_flat = query.squeeze(1)

        if all_pad is not None and all_pad.any():
            attn_output = torch.where(all_pad.unsqueeze(1), query_flat, attn_output)

        attn_output = self.attn_norm(attn_output + query_flat)
        attn_output = self.ffn_norm(attn_output + self.ffn(attn_output))
        return self.post_proj(torch.cat([attn_output, cell_vec], dim=1))


class CombinedModel(nn.Module):
    def __init__(
        self,
        *,
        espf_vocab_size: int,
        espf_max_len: int,
        cell_in_dim: int,
        llm_dim: int,
        drug_hidden_size: int = 256,
        cell_hidden_dim: int = 128,
        lambda_adv: float = 0.1,
    ):
        super().__init__()
        self.llm_dim = int(llm_dim)

        self.drug_transformer = DrugEncoder(
            vocab_size=espf_vocab_size,
            hidden_size=drug_hidden_size,
            max_seq_length=espf_max_len,
            num_layers=2,
            num_heads=8,
            dropout_rate=0.1,
            pad_token_id=0,
        )
        self.cell_adapter = CellResidualAdapter(
            input_dim=cell_in_dim,
            hidden_dim=cell_hidden_dim,
            dropout=0.1,
        )
        self.fusion = CrossAttentionFusion(
            drug_dim=drug_hidden_size,
            cell_dim=cell_hidden_dim,
            out_dim=drug_hidden_size,
            num_heads=8,
            dropout=0.1,
        )
        self.llm_adapter = nn.Sequential(
            nn.LayerNorm(self.llm_dim),
            nn.Linear(self.llm_dim, drug_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.drug_merge = nn.Sequential(
            nn.Linear(drug_hidden_size * 2, drug_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mlp = MLP(input_size=drug_hidden_size * 2, hidden_size=256)
        self.dann = DANNModule(feature_dim=drug_hidden_size * 2, num_domains=2, lambda_adv=lambda_adv)

    def forward(
        self,
        *,
        espf_a: torch.Tensor,
        espf_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        omics_latent: torch.Tensor,
        llm_a: torch.Tensor,
        llm_b: torch.Tensor,
        is_test: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cell_code = self.cell_adapter(omics_latent)

        espf_ab = torch.cat([espf_a, espf_b], dim=0)
        mask_ab = torch.cat([mask_a, mask_b], dim=0)
        encoded_ab = self.drug_transformer(espf_ab, mask_ab)

        cell_twice = torch.cat([cell_code, cell_code], dim=0)
        fused_ab = self.fusion(encoded_ab, mask_ab, cell_twice)
        fused_a, fused_b = fused_ab.chunk(2, dim=0)

        pair_feature_adv = torch.cat([fused_a, fused_b], dim=1)

        llm_ab = torch.cat([llm_a, llm_b], dim=0)
        llm_projected = self.llm_adapter(llm_ab)
        struct_ab = torch.cat([fused_a, fused_b], dim=0)
        merged_ab = self.drug_merge(torch.cat([struct_ab, llm_projected], dim=1))
        drug_a_final, drug_b_final = merged_ab.chunk(2, dim=0)

        pair_feature_full = torch.cat([drug_a_final, drug_b_final], dim=1)
        output = self.mlp(pair_feature_full)

        if is_test:
            return output, None
        return output, pair_feature_adv


def _compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> tuple[float, float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


def _forward_batch(model: CombinedModel, batch_data: dict, device: torch.device, is_test: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model(
        espf_a=batch_data["ESPF_A"].to(device),
        espf_b=batch_data["ESPF_B"].to(device),
        mask_a=batch_data["mask_A"].to(device),
        mask_b=batch_data["mask_B"].to(device),
        omics_latent=batch_data["omics_latent"].to(device),
        llm_a=batch_data["llm_A"].to(device),
        llm_b=batch_data["llm_B"].to(device),
        is_test=is_test,
    )


def train_supervised(
    model: CombinedModel,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.train()
    running_loss = 0.0
    y_true: list[float] = []
    y_pred: list[float] = []

    progress_bar = tqdm(train_loader, desc="Training (supervised)", leave=False)
    for batch in progress_bar:
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs, _ = _forward_batch(model, batch, device, is_test=False)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(outputs.detach().cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    mse, rmse, mae, r2 = _compute_regression_metrics(y_true, y_pred)
    return running_loss / len(train_loader), mse, rmse, mae, r2


def train_with_dann(
    model: CombinedModel,
    source_loader,
    target_loader,
    optimizer,
    criterion,
    device: torch.device,
    adv_weight: float = 1.0,
) -> tuple[float, float, float, float, float, float, float]:
    model.train()
    model.dann.train()

    running_loss = 0.0
    align_loss_sum = 0.0
    domain_acc_sum = 0.0
    y_true: list[float] = []
    y_pred: list[float] = []

    target_iter = iter(target_loader)
    progress_bar = tqdm(source_loader, desc="Training (DANN)", leave=False)

    for batch_src in progress_bar:
        try:
            batch_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            batch_tgt = next(target_iter)

        labels_src = batch_src["label"].to(device)
        domain_src = batch_src["domain_type"].to(device).long()
        domain_tgt = batch_tgt["domain_type"].to(device).long()

        optimizer.zero_grad()

        outputs_src, feat_src = _forward_batch(model, batch_src, device, is_test=False)
        outputs_src = outputs_src.squeeze(-1)
        task_loss = criterion(outputs_src, labels_src)

        _, feat_tgt = _forward_batch(model, batch_tgt, device, is_test=False)

        all_features = torch.cat([feat_src, feat_tgt], dim=0)
        all_domains = torch.cat([domain_src, domain_tgt], dim=0)
        adv_loss, domain_logits = model.dann(all_features, all_domains)

        loss = task_loss + adv_weight * adv_loss
        loss.backward()
        optimizer.step()

        running_loss += task_loss.item()
        align_loss_sum += adv_loss.item()
        y_true.extend(labels_src.detach().cpu().numpy())
        y_pred.extend(outputs_src.detach().cpu().numpy())
        with torch.no_grad():
            batch_domain_acc = (domain_logits.argmax(dim=1) == all_domains).float().mean().item()
        domain_acc_sum += batch_domain_acc
        progress_bar.set_postfix(loss=loss.item(), dom_acc=batch_domain_acc)

    mse, rmse, mae, r2 = _compute_regression_metrics(y_true, y_pred)
    return (
        running_loss / len(source_loader),
        mse,
        rmse,
        mae,
        r2,
        domain_acc_sum / len(source_loader),
        align_loss_sum / len(source_loader),
    )


@torch.no_grad()
def validate(
    model: CombinedModel,
    val_loader,
    criterion,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()
    running_loss = 0.0
    y_true: list[float] = []
    y_pred: list[float] = []

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    for batch in progress_bar:
        labels = batch["label"].to(device)
        outputs, _ = _forward_batch(model, batch, device, is_test=True)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(outputs.detach().cpu().numpy())

    mse, rmse, mae, r2 = _compute_regression_metrics(y_true, y_pred)
    return running_loss / len(val_loader), mse, rmse, mae, r2
