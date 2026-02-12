# utils/losses.py
import torch

def continuity_loss_cell6(
    pEqnFlux_pred6: torch.Tensor,   # (B,6)
    X_cell: torch.Tensor,           # (B,18) = (B,6*3)
    cell_face_sign: torch.Tensor,   # (B,6)  (+1/-1)
    V: torch.Tensor,                # (B,)   cell volume
    fields_per_face: int = 3,
    phi_in_face: int = 1,           # within [0..2], phiHbyA position inside each face triplet
) -> torch.Tensor:
    """
    For each sample(cell), compute divergence from 6 face fluxes and penalize div^2.
    Assumes X_cell packs 6 faces * 3 fields, and pEqnFlux_pred6 predicts 6 faces.
    """
    B = X_cell.shape[0]
    X6 = X_cell.view(B, 6, fields_per_face)  # (B,6,3)
    phiHbyA6 = X6[:, :, phi_in_face]         # (B,6)

    phi_pred6 = phiHbyA6 - pEqnFlux_pred6    # (B,6)
    div = (phi_pred6 * cell_face_sign).sum(dim=1) / V  # (B,)
    return (div ** 2).mean()


