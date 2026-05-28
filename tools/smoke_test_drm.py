import sys
sys.path.append('.')
import torch
from types import SimpleNamespace

try:
    from models import networks
    from models.DRM_model import DRMModel
except Exception as e:
    print('Failed importing DRMModel:', e)
    raise

MTM_CKPT = '/home/asingh/Desktop/uni/3d_vision/project/latest_net_MTM.pth'

opt = SimpleNamespace()
opt.latent_dim = 128
opt.point_dim = 3
opt.sdf_hidden_dim = 256
opt.sdf_num_layers = 5
# only coarse supervision loss is used by DRM
opt.sdf_num_points = 32
opt.norm = 'instance'
opt.init_type = 'normal'
opt.init_gain = 0.02
opt.gpu_ids = []
opt.isTrain = True
opt.lr = 0.001
opt.checkpoints_dir = './checkpoints'
opt.datamode = 'aligned'
opt.name = 'DRM_test'
opt.display_ncols = 2
opt.ngf = 32

mtm = networks.define_MTM(
    input_nc_A=29,
    input_nc_B=3,
    ngf=opt.ngf,
    n_layers=3,
    img_height=512,
    img_width=320,
    grid_size=3,
    add_tps=True,
    add_depth=True,
    add_segmt=True,
    latent_dim=opt.latent_dim,
    norm='instance',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=opt.gpu_ids,
)

state_dict = torch.load(MTM_CKPT, map_location='cpu')
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata
mtm.load_state_dict(state_dict)
mtm.eval()

model = DRMModel(opt)
model.train()

# simulate the MTM inputs needed to produce z
mtm_inputA = torch.randn(2, 29, 512, 320)
mtm_inputB = torch.randn(2, 3, 512, 320)
with torch.no_grad():
    mtm_output = mtm(mtm_inputA, mtm_inputB)

z = mtm_output.get('z')
batch = {
    'cloth': torch.randn(2, 3, 512, 320),
    'z': z,
}
model.set_input(batch)

# run one optimization step
try:
    model.optimize_parameters()
    print('Forward+backward completed, loss_sdf=', model.loss_sdf.detach().item())
except Exception as e:
    print('Error during optimize_parameters:', e)
    raise
