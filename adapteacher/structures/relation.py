import torch

def RankEmbedding(rank_dim=128,feat_dim=1024,wave_len=1000):
    rank_range = torch.arange(0, rank_dim).cuda().float()

    feat_range = torch.arange(feat_dim / 2).cuda()
    dim_mat = feat_range / (feat_dim / 2)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, -1)
    rank_mat = rank_range.view(-1, 1)

    mul_mat = rank_mat * dim_mat
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding

def RankEmbedding_3d(dim_g=64, wave_len=1000, height=1024, width=1024):
    # 生成网格坐标
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    y, x = y.float(), x.float()

    # 归一化坐标
    y, x = y / (height - 1), x / (width - 1)

    # 生成位置矩阵
    position_mat = torch.stack((y, x), dim=-1)  # [height, width, 2]

    # 准备维度矩阵
    feat_range = torch.arange(dim_g // 4)  # 这里调整为 dim_g 的一半
    dim_mat = feat_range / (dim_g // 4)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    # 将位置矩阵扩展至所需维度
    position_mat = position_mat.view(1, height, width, 2).repeat(dim_g // 4, 1, 1, 1)  # [dim_g//4, height, width, 2]
    position_mat = position_mat * dim_mat.view(dim_g // 4, 1, 1, 1)  # [dim_g//4, height, width, 2]

    # 应用正弦和余弦编码
    sin_mat = torch.sin(position_mat)
    cos_mat = torch.cos(position_mat)

    # 组合正弦和余弦嵌入
    embedding = torch.cat((sin_mat, cos_mat), dim=0)  # [dim_g//2, height, width, 2]

    # 调整形状以匹配输出要求
    embedding = embedding.permute(1, 2, 0, 3).reshape(height, width, dim_g)

    return embedding

def PositionalEmbedding( f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8, device=f_g.device)
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding

def PositionalEmbeddingPerClass(boxes, num_classes=9, dim_g=64, wave_len=1000):
    """
    Given a tensor of bounding box predictions of size [N, 4*num_classes],
    compute positional embeddings for each class.
    
    Args:
        boxes (Tensor): Tensor of size [N, 4*num_classes].
        num_classes (int): Number of prediction classes.
        dim_g (int): Dimension of the output embeddings.
        wave_len (float): Wavelength parameter.

    Returns:
        Tensor: Positional embeddings of size [N, num_classes, dim_g].
    """
    # Reshape boxes to [N, num_classes, 4]
    boxes = boxes.view(-1, num_classes, 4)
    N = boxes.shape[0]
    # Compute center, width, and height for each class
    cx = (boxes[:, :, 0] + boxes[:, :, 2]) * 0.5
    cy = (boxes[:, :, 1] + boxes[:, :, 3]) * 0.5
    w = (boxes[:, :, 2] - boxes[:, :, 0]) + 1.
    h = (boxes[:, :, 3] - boxes[:, :, 1]) + 1.

    # Calculate deltas
    delta_x = torch.log(torch.clamp(torch.abs(cx - cx.view(-1, 1, num_classes)) / w, min=1e-3))
    delta_y = torch.log(torch.clamp(torch.abs(cy - cy.view(-1, 1, num_classes)) / h, min=1e-3))
    delta_w = torch.log(w / w.view(-1, 1, num_classes))
    delta_h = torch.log(h / h.view(-1, 1, num_classes))

    # Combine deltas
    position_mat = torch.stack((delta_x, delta_y, delta_w, delta_h), dim=3)  # [N, num_classes, num_classes, 4]

    # Create frequency matrix
    dim_mat = torch.pow(wave_len, -torch.arange(dim_g / 8, device=boxes.device) / (dim_g / 8))
    dim_mat = dim_mat.view(1, 1, 1, 1, -1)  # [1, 1, 1, dim_g/8]

    # Apply frequency encoding
    position_mat = position_mat.view(N, N, num_classes, 4, 1)  # [1024, 1024, 9, 4, 1]
    position_mat = position_mat.repeat(1, 1, 1, 1, int(dim_g/8)) # [1024, 1024, 9, 4, dim_g/8]
    mul_mat = position_mat * dim_mat # [1024, 1024, 9, 4, dim_g/8]
    mul_mat = mul_mat.view(N, N, num_classes, 4 * int(dim_g/8)) # [1024, 1024, 9, 4*dim_g/8]

    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1) # [1024, 1024, 9, 8*dim_g/8]
    # embedding = embedding.view(-1, num_classes, dim_g)  # [N, num_classes, dim_g]
    return embedding
    # position_mat = 100. * position_mat.unsqueeze(3)  # [N, num_classes, num_classes, 4, 1]
    # mul_mat = position_mat * dim_mat  # [N, num_classes, num_classes, 4, dim_g/8]
    # mul_mat = mul_mat.view(-1, num_classes, num_classes, 4 * int(dim_g/8))  # [N, num_classes, num_classes, 4*dim_g/8]

    # # Sinusoidal embedding
    # sin_mat = torch.sin(mul_mat)
    # cos_mat = torch.cos(mul_mat)
    # embedding = torch.cat((sin_mat, cos_mat), -1)  # [N, num_classes, num_classes, 8*dim_g/8]

    # # Reshape to get final embeddings for each class
    # embedding = embedding.view(-1, num_classes, dim_g)  # [N, num_classes, dim_g]

    # return embedding


def PositionalEmbeddingPerBox(boxes, num_classes=9, dim_g=64, wave_len=1000):
    """
    Given a tensor of bounding box predictions of size [N, 4*num_classes],
    compute positional embeddings for each box and each class.
    
    Args:
        boxes (Tensor): Tensor of size [N, 4*num_classes].
        num_classes (int): Number of prediction classes.
        dim_g (int): Dimension of the output embeddings.
        wave_len (float): Wavelength parameter.

    Returns:
        Tensor: Positional embeddings of size [N, num_classes, dim_g].
    """
    N = boxes.shape[0]
    
    # Reshape boxes to [N, num_classes, 4]
    boxes = boxes.view(N, num_classes, 4)

    # Compute center, width, and height for each box
    cx = (boxes[:, :, 0] + boxes[:, :, 2]) * 0.5
    cy = (boxes[:, :, 1] + boxes[:, :, 3]) * 0.5
    w = (boxes[:, :, 2] - boxes[:, :, 0]) + 1.
    h = (boxes[:, :, 3] - boxes[:, :, 1]) + 1.

    # Normalize dimensions
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-6)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-6)
    w = torch.log(w)
    h = torch.log(h)

    # Combine dimensions into a single tensor
    position_mat = torch.stack((cx, cy, w, h), dim=2)  # [N, num_classes, 4]

    # Prepare frequency matrix
    dim_mat = torch.pow(wave_len, -torch.arange(dim_g / 8, device=boxes.device) / (dim_g / 8))
    dim_mat = dim_mat.view(1, 1, 1, -1)  # [1, 1,1, dim_g/8]
    # Apply frequency encoding
    position_mat = position_mat.unsqueeze(3)  # [N, num_classes, 4, 1]
    position_mat = position_mat * 100.0  # Scale position matrix
    mul_mat = position_mat * dim_mat  # [N, num_classes, 4, dim_g/8]
    mul_mat = mul_mat.view(N, num_classes, -1)  # [N, num_classes, 4*dim_g/8]

    # Sinusoidal embedding
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)  # [N, num_classes, 8*dim_g/8]

    return embedding

