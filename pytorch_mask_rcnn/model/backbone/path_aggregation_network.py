from collections import OrderedDict

import torch.nn.functional as F
from torch import nn

    
class PathAggregationNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        assert all(in_channels_list), 'in_channels=0 is currently not supported'
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        
        for i, in_channels in enumerate(in_channels_list):
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            
            if i > 0:
                spatial_block_module = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                    nn.ReLU(inplace=True)
                )
                self.spatial_blocks.append(spatial_block_module)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #names = list(x.keys())
        x = list(x.values())

        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        # compute FPN's results
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
            
        #print([tuple(t.shape[2:]) for t in results])
        # compute PANet's results
        for idx in range(len(x) - 1):
            outer_bottom_up = self.spatial_blocks[idx](results[idx])
            results[idx + 1] += outer_bottom_up

        #out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return results