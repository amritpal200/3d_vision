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
import torch.nn.functional as F
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
        parser.add_argument('--sdf_num_points', type=int, default=64, help='number of 3D query points to sample per item when simulated')
        parser.add_argument('--lambda_coarse', type=float, default=1.0, help='weight for coarse supervision loss')
        parser.add_argument('--lambda_surface', type=float, default=0.1, help='weight for surface loss')
        parser.add_argument('--lambda_sign', type=float, default=0.1, help='weight for sign loss')
        parser.add_argument('--lambda_eikonal', type=float, default=0.1, help='weight for eikonal loss')
        parser.add_argument('--lambda_normal', type=float, default=0.1, help='weight for normal consistency loss')
        
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

        # total loss plus individual terms for logging
        self.loss_names = ['sdf', 'coarse', 'surface', 'sign', 'eikonal', 'normal']

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

        self.lambda_coarse = opt.lambda_coarse
        self.lambda_surface = opt.lambda_surface
        self.lambda_sign = opt.lambda_sign
        self.lambda_eikonal = opt.lambda_eikonal
        self.lambda_normal = opt.lambda_normal

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
        self.surface_points = input.get('surface_points', None)
        self.surface_normals = input.get('surface_normals', None)
        self.sign_labels = input.get('sign_labels', None)

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
            print(f'Warning: no query points provided for DRMModel; simulating random points. To fix this, ensure your dataset returns a key like "points" or "sdf_points" with shape [B, N, 3].')
            N = getattr(self.opt, 'sdf_num_points', 64)
            self.points = torch.randn(batch_size, N, self.point_dim, device=self.device)
        else:
            self.points = self.points.to(self.device)

        if self.surface_points is not None:
            self.surface_points = self.surface_points.to(self.device)
        if self.surface_normals is not None:
            self.surface_normals = self.surface_normals.to(self.device)
        if self.sign_labels is not None:
            self.sign_labels = self.sign_labels.to(self.device)

        # get ground-truth sdf values for the sampled points; allow multiple key names
        self.sdf_gt = input.get('sdf', input.get('sdf_gt', input.get('sdf_values', None)))
        if self.sdf_gt is None:
            print(f'Warning: no ground-truth SDF values provided for DRMModel; simulating zeros. To fix this, ensure your dataset returns a key like "sdf" or "sdf_gt" with shape [B, N, 1] corresponding to the query points.')
            # simulate zeros target so training step can run without dataset SDFs
            self.sdf_gt = torch.zeros(batch_size, self.points.size(1), 1, device=self.device)
        else:
            # ensure shape [B, N, 1]
            self.sdf_gt = self.sdf_gt.to(self.device)
            if self.sdf_gt.dim() == 2:
                self.sdf_gt = self.sdf_gt.unsqueeze(-1)

        # derive sign labels from sdf if not explicitly given
        if self.sign_labels is None and self.sdf_gt is not None:
            self.sign_labels = torch.where(self.sdf_gt >= 0, torch.ones_like(self.sdf_gt), -torch.ones_like(self.sdf_gt))

        # optional per-sample sdf scale (if precomputed SDF was normalized)
        self.sdf_scale = input.get('sdf_scale', None)
        if self.sdf_scale is None:
            self.sdf_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        else:
            self.sdf_scale = self.sdf_scale.to(self.device)
            # expand scalar to batch-size if necessary
            if self.sdf_scale.dim() == 0:
                self.sdf_scale = self.sdf_scale.unsqueeze(0).expand(batch_size)
            elif self.sdf_scale.dim() == 1 and self.sdf_scale.size(0) == 1:
                self.sdf_scale = self.sdf_scale.expand(batch_size)

    def _predict_with_grad(self, points):
        points = points.clone().detach().requires_grad_(True)
        sdf_pred = self.netDRM(self.z, points)
        sdf_sum = sdf_pred.sum()
        grads = torch.autograd.grad(
            outputs=sdf_sum,
            inputs=points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return sdf_pred, grads

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.sdf_pred = self.netDRM(self.z, self.points)

    def backward_G(self):
        """Calculate losses, gradients; called in every training iteration"""
        # Coarse supervision loss: L_coarse = mean(|f_coarse(p_i) - s_i|)
        if self.sdf_gt.numel() > 0:
            self.loss_coarse = torch.abs(self.sdf_pred - self.sdf_gt).mean()
        else:
            self.loss_coarse = torch.zeros(1, device=self.device, dtype=self.sdf_pred.dtype)

        # Surface loss: L_surface = mean(|f(p_i)|) on surface points if provided
        if self.surface_points is not None and self.surface_points.numel() > 0:
            surface_pred = self.netDRM(self.z, self.surface_points)
            self.loss_surface = surface_pred.abs().mean()
        else:
            self.loss_surface = torch.zeros(1, device=self.device, dtype=self.sdf_pred.dtype)

        # Sign loss: L_sign = mean(max(0, -y_i * f(p_i)))
        if self.sign_labels is not None and self.sign_labels.numel() > 0:
            self.loss_sign = torch.relu(-self.sign_labels * self.sdf_pred).mean()
        else:
            self.loss_sign = torch.zeros(1, device=self.device, dtype=self.sdf_pred.dtype)

        # Eikonal loss: L_eikonal = mean((||grad f|| - 1)^2)
        if self.points is not None and self.points.numel() > 0:
            _, grads = self._predict_with_grad(self.points)
            grad_norm = torch.linalg.norm(grads, dim=-1)
            # If SDFs were normalized when precomputing, the eikonal target should be 1/scale
            try:
                scale = self.sdf_scale
                if isinstance(scale, torch.Tensor):
                    if scale.dim() == 1:
                        target = (1.0 / scale).view(-1, 1)
                    else:
                        target = (1.0 / scale)
                else:
                    target = 1.0
            except Exception:
                target = 1.0

            self.loss_eikonal = ((grad_norm - target) ** 2).mean()
        else:
            self.loss_eikonal = torch.zeros(1, device=self.device, dtype=self.sdf_pred.dtype)

        # Normal consistency loss: L_normal = mean(1 - dot(n_pred, n_gt))
        if self.surface_points is not None and self.surface_normals is not None and self.surface_points.numel() > 0 and self.surface_normals.numel() > 0:
            _, surface_grads = self._predict_with_grad(self.surface_points)
            n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
            n_gt = F.normalize(self.surface_normals, p=2, dim=-1, eps=1e-8)
            self.loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
        else:
            self.loss_normal = torch.zeros(1, device=self.device, dtype=self.sdf_pred.dtype)

        self.loss_sdf = (
            self.lambda_coarse * self.loss_coarse
            + self.lambda_surface * self.loss_surface
            + self.lambda_sign * self.loss_sign
            + self.lambda_eikonal * self.loss_eikonal
            + self.lambda_normal * self.loss_normal
        )
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