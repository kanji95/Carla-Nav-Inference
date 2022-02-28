import torch


@torch.no_grad()
def compute_mask_IOU(masks, target, mask_thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > mask_thresh) * target
    intersection = temp.sum()
    union = (((masks > mask_thresh) + target) - temp).sum()
    return intersection, union


@torch.no_grad()
def compute_batch_IOU(masks, target, mask_thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > mask_thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    union = torch.sum(
        (((masks > mask_thresh) + target) - temp).flatten(1), dim=-1, keepdim=True
    )
    return intersection, union


@torch.no_grad()
def intersection_at_t(masks, target, mask_thresh=0.3, area_thresh=0.5):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > mask_thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    mask_area = torch.sum(
        (masks > mask_thresh).flatten(1), dim=-1, keepdim=True)
    accuracy = (intersection > area_thresh*mask_area).float().mean().item()
    return accuracy


@torch.no_grad()
def pointing_game(masks, target):
    # import pdb;pdb.set_trace()
    assert target.shape[-2:] == masks.shape[-2:]
    batch_size = masks.shape[0]
    max_indices = masks.flatten(1).argmax(dim=-1)[:, None]
    target_values = target.flatten(1).gather(1, max_indices).sum(dim=-1)
    accuracy = (target_values > 0).float().mean().item()
    return accuracy


@torch.no_grad()
def recall_at_k(masks, target, topk=1):
    assert target.shape[-2:] == masks.shape[-2:]
    batch_size = masks.shape[0]
    values, indices = torch.topk(masks.flatten(1), k=topk)
    # if topk == 1:
    ##     indices = indices.unsqueeze(1)
    target_values = target.flatten(1).gather(1, indices).sum(dim=-1)
    accuracy = (target_values > 0).float().mean().item()
    return accuracy


@torch.no_grad()
def dice_score(masks, target, mask_thresh=0.3):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = (masks > mask_thresh) * target
    intersection = torch.sum(temp.flatten(1), dim=-1, keepdim=True)
    union = torch.sum(
        (((masks > mask_thresh) + target) - temp).flatten(1), dim=-1, keepdim=True
    )
    score = 2*intersection.sum()/union.sum()
    return score.item()
