import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class TanhPolar(nn.Module):
    def __init__(self, width, height, angular_offset_deg=270):
        super(TanhPolar, self).__init__()

        self.width = width
        self.height = height

        warp_gridx, warp_gridy = self._get_tanh_polar_warp_grids(width, height, angular_offset_deg=angular_offset_deg)
        restore_gridx, restore_gridy = self._get_tanh_polar_restore_grids(width, height, angular_offset_deg=angular_offset_deg)

        self.register_buffer('warp_gridx', torch.tensor(warp_gridx).unsqueeze(0))
        self.register_buffer('warp_gridy', torch.tensor(warp_gridy).unsqueeze(0))
        self.register_buffer('restore_gridx', torch.tensor(restore_gridx).unsqueeze(0))
        self.register_buffer('restore_gridy', torch.tensor(restore_gridy).unsqueeze(0))

    def warp(self, inp):
        batch_size = inp.shape[0]
        warp_gridx = self.warp_gridx.repeat(batch_size, 1, 1)
        warp_gridy = self.warp_gridy.repeat(batch_size, 1, 1)
        grid = torch.stack((warp_gridy, warp_gridx), dim=3)
        out = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out

    def restore(self, inp):
        batch_size = inp.shape[0]
        restore_gridx = self.restore_gridx.repeat(batch_size, 1, 1)
        restore_gridy = self.restore_gridy.repeat(batch_size, 1, 1)
        inp = F.pad(inp, (0, 0, 1, 0, 1, 1), mode='reflect')
        grid = torch.stack((restore_gridy, restore_gridx), dim=3)
        out = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out

    @staticmethod
    def _get_tanh_polar_warp_grids(W, H, angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0
        roi_center = np.array([W // 2, H // 2], np.float32)
        roi_radii = np.array([W, H], np.float32) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W), np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1)
        radii = normalised_dest_indices[..., 0]
        orientation_x = np.cos(normalised_dest_indices[..., 1])
        orientation_y = np.sin(normalised_dest_indices[..., 1])
        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)
        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W, H, angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0
        roi_center = np.array([W // 2, H // 2], np.float32)
        roi_radii = np.array([W, H], np.float32) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float)
        normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                                [sin_offset, cos_offset]]))
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                         roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]
        src_radii = np.tanh(radii)
        src_x_indices = src_radii * W + 1.0
        src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                                2.0 / np.pi) * H, H) + 1.0
        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

nn.TanhPolar = TanhPolar