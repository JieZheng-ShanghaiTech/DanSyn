import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import pandas as pd

# =========================
#  Basic Blocks & DANN Utils
# =========================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DANNModule(nn.Module):
    def __init__(self, feature_dim=512, num_domains=2, lambda_adv=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_domains)
        )
    def forward(self, f, domain_label):
        f_rev = self.grl(f)
        domain_logits = self.domain_discriminator(f_rev)
        adv_loss = F.cross_entropy(domain_logits, domain_label)
        return adv_loss, domain_logits


# =========================
#  Encoders
# =========================

class DrugEncoder(nn.Module):
    """
    ESPF Transformer Encoder.
    pooling="nopool": return [B, L, D]
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_length: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

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
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.size()
        # Truncate
        if L > self.max_seq_length:
            input_ids = input_ids[:, :self.max_seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_length]
            L = self.max_seq_length

        word_embeds = self.word_embedding(input_ids)
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos_embeds = self.position_embedding(position_ids)

        x = self.layer_norm(word_embeds + pos_embeds)
        x = self.dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool() # True=mask out
        else:
            src_key_padding_mask = None

        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # [B, L, D]
        return encoded


class CellResidualAdapter(nn.Module):
    """
    MLP + Residual Block
    """
    def __init__(self, input_dim=200, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.final_relu = nn.ReLU()

    def forward(self, omics_latent):
        x = self.proj_in(omics_latent)
        identity = x
        out = self.res_block(x)
        out += identity
        out = self.final_relu(out)
        return out # [B, 128]


# =========================
#  Fusion: Gated Pooling (The Core Modification)
# =========================

class GatedPoolingFusion(nn.Module):
    """
    Gated Pooling
    """
    def __init__(self, drug_dim, cell_dim, out_dim):
        super().__init__()
        # Attention Score
        self.attn_net = nn.Sequential(
            nn.Linear(drug_dim + cell_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Drug_pooled + Cell
        self.post_proj = nn.Sequential(
            nn.Linear(drug_dim + cell_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, drug_seq, drug_mask, cell_vec):
        """
        drug_seq: [B, L, D_drug]
        drug_mask: [B, L] (1=valid, 0=pad)
        cell_vec: [B, D_cell]
        """
        B, L, D = drug_seq.size()
        
        # 1. Expand Cell Vector to match sequence length
        # [B, D_cell] -> [B, L, D_cell]
        cell_expanded = cell_vec.unsqueeze(1).expand(-1, L, -1)
        
        # 2. Concat at token level: [B, L, D_drug + D_cell]
        combined_seq = torch.cat([drug_seq, cell_expanded], dim=-1)
        
        # 3. Calculate Attention Scores: [B, L, 1]
        raw_scores = self.attn_net(combined_seq)
        
        # 4. Masking (Critical for ESPF padding!)
        if drug_mask is not None:
            mask_bool = (drug_mask == 0) # True where pad
            raw_scores = raw_scores.masked_fill(mask_bool.unsqueeze(-1), -1e9)
        
        # 5. Softmax -> Weights [B, L, 1]
        attn_weights = F.softmax(raw_scores, dim=1)
        
        # 6. Weighted Sum (Pooling): [B, L, D] * [B, L, 1] -> sum -> [B, D]
        drug_pooled = torch.sum(drug_seq * attn_weights, dim=1)
        
        # 7. Final Concatenation: [Drug_pooled, Cell]
        final_rep = torch.cat([drug_pooled, cell_vec], dim=1) # [B, D_drug + D_cell]
        
        # 8. Project to unified dimension
        output = self.post_proj(final_rep) # [B, out_dim]
        
        return output


# =========================
#  Combined Model
# =========================

class CombinedModel(nn.Module):
    def __init__(
        self,
        num_domains,
        espf_vocab_size=3000,
        espf_max_len=50,
        drug_hidden_size=256,
        cell_in_dim=200,
        cell_hidden_dim=128,
        lambda_adv=0.1,
        llm_dim=1536
    ):
        super().__init__()
        self.num_cells = num_domains
        self.llm_dim = int(llm_dim)
        self.drug_hidden_size = int(drug_hidden_size)

        # 1. Drug Encoder (Transformer, returns sequence)
        self.drug_transformer_model = DrugEncoder(
            vocab_size=espf_vocab_size,
            hidden_size=drug_hidden_size,
            max_seq_length=espf_max_len,
            num_layers=2,
            num_heads=8,
            dropout_rate=0.1,
            pad_token_id=0
        )

        # 2. Cell Adapter (MLP + Residual)
        self.cell_adapter = CellResidualAdapter(
            input_dim=cell_in_dim,
            hidden_dim=cell_hidden_dim,
            dropout=0.1
        )

        # 3. Fusion Strategy: Gated Pooling
        # Drug(256) + Cell(128) -> 256
        self.fusion = GatedPoolingFusion(
            drug_dim=drug_hidden_size,
            cell_dim=cell_hidden_dim,
            out_dim=drug_hidden_size 
        )

        # 4. LLM Adapter
        self.llm_adapter = nn.Sequential(
            nn.LayerNorm(self.llm_dim),
            nn.Linear(self.llm_dim, drug_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 5. Merge Structure + LLM
        self.drug_merge = nn.Sequential(
            nn.Linear(drug_hidden_size * 2, drug_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 6. Regression Head
        # DrugA(256) + DrugB(256) = 512
        self.mlp = MLP(input_size=drug_hidden_size * 2, hidden_size=256)

        # 7. DANN Module
        self.dann = DANNModule(feature_dim=drug_hidden_size * 2, num_domains=2, lambda_adv=lambda_adv)

    def forward(
        self,
        ESPF_A, ESPF_B, mask_A, mask_B,
        omics_latent_vectors,
        llm_A=None, llm_B=None,
        labels=None,
        is_test=False
    ):
        # ---- 1. Encode Drug Sequence ----
        tA = self.drug_transformer_model(ESPF_A, mask_A)  # [B, L, 256]
        tB = self.drug_transformer_model(ESPF_B, mask_B)  # [B, L, 256]

        # ---- 2. Encode Cell Vector ----
        cell_code = self.cell_adapter(omics_latent_vectors)  # [B, 128]

        # ---- 3. Gated Pooling Fusion ----
        # Cell context guides the pooling of Drug Sequence
        final_Ainput = self.fusion(tA, mask_A, cell_code)  # [B, 256]
        final_Binput = self.fusion(tB, mask_B, cell_code)  # [B, 256]

        # =============== ADV FEATURE (Structure + Omics Only) ===============
        pair_feature_adv = torch.cat([final_Ainput, final_Binput], dim=1)  # [B, 512]

        # =============== FULL FEATURE (With LLM) ===============
        B_size = final_Ainput.size(0)
        device = final_Ainput.device

        if llm_A is None:
            llm_A = torch.zeros(B_size, self.llm_dim, device=device, dtype=final_Ainput.dtype)
        if llm_B is None:
            llm_B = torch.zeros(B_size, self.llm_dim, device=device, dtype=final_Binput.dtype)

        # LLM Projection
        llmA_256 = self.llm_adapter(llm_A) # [B, 256]
        llmB_256 = self.llm_adapter(llm_B) # [B, 256]

        # Merge Structure + LLM
        drugA_final = self.drug_merge(torch.cat([final_Ainput, llmA_256], dim=1)) # [B, 256]
        drugB_final = self.drug_merge(torch.cat([final_Binput, llmB_256], dim=1)) # [B, 256]

        # Concat for MLP
        pair_feature_full = torch.cat([drugA_final, drugB_final], dim=1) # [B, 512]

        output = self.mlp(pair_feature_full) # [B, 1]

        if is_test:
            return output, None, None

        return output, pair_feature_adv, None


# =========================
#  Train / Val / Test Logic
# =========================

def train_supervised(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    progress_bar = tqdm(train_loader, desc="Training (supervised)", leave=False)

    for batch_data in progress_bar:
        labels = batch_data['label'].to(device)
        omics_latent_vectors = batch_data['omics_latent'].to(device)
        ESPF_A = batch_data['ESPF_A'].to(device)
        ESPF_B = batch_data['ESPF_B'].to(device)
        mask_A = batch_data['mask_A'].to(device)
        mask_B = batch_data['mask_B'].to(device)
        llm_A = batch_data['llm_A'].to(device)
        llm_B = batch_data['llm_B'].to(device)

        optimizer.zero_grad()

        outputs, _, _ = model(
            ESPF_A, ESPF_B, mask_A, mask_B,
            omics_latent_vectors,
            llm_A=llm_A, llm_B=llm_B,
            labels=labels,
            is_test=False
        )

        outputs_squeezed = outputs.squeeze(-1)
        loss = criterion(outputs_squeezed, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(outputs_squeezed.detach().cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return avg_loss, mse, rmse, mae, r2

def train_with_dann(
    model, dann_module,
    source_loader, target_loader,
    optimizer, criterion, device,
    adv_weight=1.0
):
    """
    DANN
    """
    model.train()
    dann_module.train()

    running_loss = 0.0
    y_true, y_pred = [], []
    domain_acc_sum = 0.0

    progress_bar = tqdm(source_loader, desc="Training (DANN)", leave=False)
    target_iter = iter(target_loader)

    for batch_src in progress_bar:
        try:
            batch_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            batch_tgt = next(target_iter)

        # ===== source =====
        labels_src = batch_src['label'].to(device)
        omics_latent_src = batch_src['omics_latent'].to(device)
        ESPF_A_src = batch_src['ESPF_A'].to(device)
        ESPF_B_src = batch_src['ESPF_B'].to(device)
        mask_A_src = batch_src['mask_A'].to(device)
        mask_B_src = batch_src['mask_B'].to(device)
        domain_type_src = batch_src['domain_type'].to(device).long()  # 0

        # NEW: LLM embeddings (source)
        llm_A_src = batch_src['llm_A'].to(device)
        llm_B_src = batch_src['llm_B'].to(device)

        # ===== target =====
        omics_latent_tgt = batch_tgt['omics_latent'].to(device)
        ESPF_A_tgt = batch_tgt['ESPF_A'].to(device)
        ESPF_B_tgt = batch_tgt['ESPF_B'].to(device)
        mask_A_tgt = batch_tgt['mask_A'].to(device)
        mask_B_tgt = batch_tgt['mask_B'].to(device)
        domain_type_tgt = batch_tgt['domain_type'].to(device).long()  # 1

        # NEW: LLM embeddings (target)
        llm_A_tgt = batch_tgt['llm_A'].to(device)
        llm_B_tgt = batch_tgt['llm_B'].to(device)

        optimizer.zero_grad()

        # adv_feature（adv_feature no LLM）
        outputs_src, feat_adv_src, _ = model(
            ESPF_A_src, ESPF_B_src, mask_A_src, mask_B_src,
            omics_latent_src,
            llm_A=llm_A_src, llm_B=llm_B_src,
            labels=labels_src,
            is_test=False
        )
        outputs_src_squeezed = outputs_src.squeeze(-1)
        task_loss = criterion(outputs_src_squeezed, labels_src)

        _, feat_adv_tgt, _ = model(
            ESPF_A_tgt, ESPF_B_tgt, mask_A_tgt, mask_B_tgt,
            omics_latent_tgt,
            llm_A=llm_A_tgt, llm_B=llm_B_tgt,
            labels=None,
            is_test=False
        )

        # domain prediction
        f_all = torch.cat([feat_adv_src, feat_adv_tgt], dim=0)  # (Bsrc+Btgt, 512)
        d_all = torch.cat([domain_type_src, domain_type_tgt], dim=0)  # (Bsrc+Btgt,)

        adv_loss, domain_pred = dann_module(f_all, d_all)

        loss = task_loss + adv_weight * adv_loss
        loss.backward()
        optimizer.step()

        running_loss += task_loss.item()
        y_true.extend(labels_src.detach().cpu().numpy())
        y_pred.extend(outputs_src_squeezed.detach().cpu().numpy())

        with torch.no_grad():
            batch_domain_acc = (domain_pred.argmax(dim=1) == d_all.view(-1)).float().mean().item()
        domain_acc_sum += batch_domain_acc

        progress_bar.set_postfix(loss=loss.item(), dom_acc=batch_domain_acc)

    avg_loss = running_loss / len(source_loader)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    domain_acc_avg = domain_acc_sum / len(source_loader)
    return avg_loss, mse, rmse, mae, r2, domain_acc_avg


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            labels = batch_data['label'].to(device)
            omics_latent_vectors = batch_data['omics_latent'].to(device)
            ESPF_A = batch_data['ESPF_A'].to(device)
            ESPF_B = batch_data['ESPF_B'].to(device)
            mask_A = batch_data['mask_A'].to(device)
            mask_B = batch_data['mask_B'].to(device)

            # NEW: LLM embeddings
            llm_A = batch_data['llm_A'].to(device)
            llm_B = batch_data['llm_B'].to(device)

            outputs, _, _ = model(
                ESPF_A, ESPF_B, mask_A, mask_B,
                omics_latent_vectors,
                llm_A=llm_A, llm_B=llm_B,
                labels=labels,
                is_test=False
            )

            outputs_squeezed = outputs.squeeze(-1)
            loss = criterion(outputs_squeezed, labels)

            running_loss += loss.item()
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs_squeezed.detach().cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(val_loader)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return avg_loss, mse, rmse, mae, r2


def test(model, test_loader, criterion, device, save_path="predictions.csv", domain_name=""):
    model.eval()
    test_loss = 0.0
    y_true, y_pred = [], []
    drugA_smiles, drugB_smiles, S_id, domain_tags = [], [], [], []

    progress_bar = tqdm(test_loader, desc=f"Testing ({domain_name})", leave=False)

    with torch.no_grad():
        for batch_data in progress_bar:
            drugA_smiles_batch = batch_data['smilesA']
            drugB_smiles_batch = batch_data['smilesB']
            sample_id_batch = batch_data['sample_id']

            labels = batch_data['label'].to(device)
            omics_latent_vectors = batch_data['omics_latent'].to(device)
            ESPF_A = batch_data['ESPF_A'].to(device)
            ESPF_B = batch_data['ESPF_B'].to(device)
            mask_A = batch_data['mask_A'].to(device)
            mask_B = batch_data['mask_B'].to(device)

            # NEW: LLM embeddings
            llm_A = batch_data['llm_A'].to(device)
            llm_B = batch_data['llm_B'].to(device)

            outputs, _, _ = model(
                ESPF_A, ESPF_B, mask_A, mask_B,
                omics_latent_vectors,
                llm_A=llm_A, llm_B=llm_B,
                labels=None,
                is_test=True
            )

            outputs_squeezed = outputs.squeeze(-1)
            loss = criterion(outputs_squeezed, labels)
            test_loss += loss.item() * len(labels)

            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs_squeezed.detach().cpu().numpy())
            drugA_smiles.extend(drugA_smiles_batch)
            drugB_smiles.extend(drugB_smiles_batch)
            S_id.extend(sample_id_batch)
            domain_tags.extend([domain_name] * len(labels))

            progress_bar.set_postfix(loss=loss.item())

    results_df = pd.DataFrame({
        "Domain": domain_tags,
        "DrugA_SMILES": drugA_smiles,
        "DrugB_SMILES": drugB_smiles,
        "sample_id": S_id,
        "True_Label": y_true,
        "Predicted_Label": y_pred
    })
    results_df.to_csv(save_path, index=False)
    print(f"[{domain_name}] prediction saved to {save_path}")

    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    avg_loss = test_loss / len(test_loader.dataset)

    print(f"[{domain_name}] Loss: {avg_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return avg_loss, mse, rmse, mae, r2
