"""Model class.

You can specify '--model DRM' to use this model.
It implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
"""
import torch
from .base_model import BaseModel
from . import networks
import sys
sys.path.append('..')
from util import util


class DRMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--latent_dim', type=int, default=128, help='dimension of encoded z from MPM')
        parser.add_argument('--point_dim', type=int, default=3, help='dimension of each queried point')
        parser.add_argument('--sdf_hidden_dim', type=int, default=256, help='hidden width of the SDF MLP')
        parser.add_argument('--sdf_num_layers', type=int, default=5, help='number of layers in the SDF MLP')
        # deprecated: previous placeholder pairwise loss settings removed
        parser.add_argument('--sdf_num_points', type=int, default=64, help='number of 3D query points to sample per item when simulated')
        
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.latent_dim = opt.latent_dim
        self.point_dim = opt.point_dim
        # no pairwise loss terms used; training uses a single coarse supervision loss

        # We'll expose a single coarse supervision loss named 'sdf' for training
        self.loss_names = ['sdf']

        self.visual_names = []
        self.model_names = ['DRM']

        self.netDRM = networks.define_DRM(
            latent_dim=opt.latent_dim,
            point_dim=opt.point_dim,
            hidden_dim=opt.sdf_hidden_dim,
            num_layers=opt.sdf_num_layers,
            output_dim=1,
            norm=opt.norm,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids,
        )

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netDRM.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.im_name = input.get('im_name')
        self.c_name = input.get('c_name')

        # try common keys for latent and points
        self.z = input.get('z', input.get('encoded_z', input.get('latent_z')))
        self.points = input.get('points', input.get('sdf_points', input.get('point_xyz', input.get('xyz'))))

        # infer batch size from available tensors (e.g., cloth) else default to 1
        batch_size = 1
        example_tensor = None
        for key in ('cloth', 'person', 'agnostic'):
            val = input.get(key, None)
            if val is not None:
                example_tensor = val
                break
        if isinstance(example_tensor, torch.Tensor):
            batch_size = example_tensor.size(0)

        # simulate latent z if missing (shape: [B, 1, latent_dim])
        if self.z is None:
            self.z = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        else:
            self.z = self.z.to(self.device)

        # simulate query points if missing (shape: [B, N, 3])
        if self.points is None:
            N = getattr(self.opt, 'sdf_num_points', 64)
            self.points = torch.randn(batch_size, N, self.point_dim, device=self.device)
        else:
            self.points = self.points.to(self.device)

        # get ground-truth sdf values for the sampled points; allow multiple key names
        self.sdf_gt = input.get('sdf', input.get('sdf_gt', input.get('sdf_values', None)))
        if self.sdf_gt is None:
            # simulate zeros target so training step can run without dataset SDFs
            self.sdf_gt = torch.zeros(batch_size, self.points.size(1), 1, device=self.device)
        else:
            # ensure shape [B, N, 1]
            self.sdf_gt = self.sdf_gt.to(self.device)
            if self.sdf_gt.dim() == 2:
                self.sdf_gt = self.sdf_gt.unsqueeze(-1)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.sdf_pred = self.netDRM(self.z, self.points)

    def backward_G(self):
        """Calculate losses, gradients; called in every training iteration"""
        # Coarse supervision loss: Lcoarse = mean(|sdf_pred - sdf_gt|)
        # self.sdf_pred shape: [B, N, 1], self.sdf_gt shape: [B, N, 1]
        self.loss_sdf = torch.abs(self.sdf_pred - self.sdf_gt).mean()
        self.loss_sdf.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def compute_visuals(self):
        """Calculate additional output images for tensorbard visualization"""
        self.sdf_pred_vis = self.sdf_pred.detach()