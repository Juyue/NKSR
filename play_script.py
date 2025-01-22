
import torch
import numpy as np
from examples.common import load_bunny_example
import nksr

import pdb; pdb.set_trace()

bunny_geom = load_bunny_example()
device = torch.device("cuda:0")

input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

reconstructor = nksr.Reconstructor(device)
field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
mesh = field.extract_dual_mesh(mise_iter=1)