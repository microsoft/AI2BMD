import re
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch_scatter import scatter
from torch_geometric.data import Data
from OGB_ViSNet.Pretrained_3D_ViSNet import output_modules
import warnings

from OGB_ViSNet.Pretrained_3D_ViSNet.visnet_block import ViSNetBlock


def create_model(args, mean=None, std=None):
    model_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        attn_activation=args["attn_activation"],
        num_heads=args["num_heads"],
        distance_influence=args["distance_influence"],
        lmax=args['lmax'],
        vecnorm_type=args['vecnorm_type'],
        vecnorm_trainable=args['vecnorm_trainable'],
        atom_feature=args['atom_feature'],
        bond_feature=args['bond_feature'],
        dropout=args.get("dropout", 0.0),
    )
  
    if args["model"] == "ViSNetBlock":
        from OGB_ViSNet.Pretrained_3D_ViSNet.visnet_block import ViSNetBlock
        representation_model = ViSNetBlock(**model_args)
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    output_model = getattr(output_modules, "Equivariant" + args["output_model"])(args["embedding_dimension"], args["activation"])

    model = ViSNet(representation_model, output_model, reduce_op=args["reduce_op"], mean=mean, std=std)
    
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

def create_clip_model(student_model, teacher_model):
    
    return ViSNetCLIP(student_model=student_model, teacher_model=teacher_model)

def load_clip_model(student_filepath, teacher_filepath, args=None, device="cpu", **kwargs):
    teacher_ckpt = torch.load(teacher_filepath, map_location="cpu")
    teacher_args = teacher_ckpt["hyper_parameters"]
    
    student_ckpt = torch.load(student_filepath, map_location="cpu")
    student_args = student_ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    teacher_model = create_model(teacher_args)
    student_model = create_model(student_args)
    model = create_clip_model(teacher_model=teacher_model, student_model=student_model)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in student_ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    print("Freezing teacher model...")
    for param in teacher_model.parameters():
        param.requires_grad = False
    return model.to(device)

class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model: ViSNetBlock,
        output_model: output_modules.EquivariantScalarKD,
        reduce_op="add",
        mean=None,
        std=None,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        
        self.reduce_op = reduce_op

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(self,
                data: Data,
                use_pos_kind,
                **kwargs) -> Tuple[Tensor, Optional[Tensor]]:


        batch = torch.zeros_like(data.atomic_num) if data.batch is None else data.batch

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(data, use_pos_kind)
        
        # apply the output network
        per_atom_scalar = self.output_model.pre_reduce(x, v, z, pos, batch)

        # aggregate atoms
        out_scalar = scatter(per_atom_scalar, batch, dim=0, reduce=self.reduce_op)

        return out_scalar, None
    
class ViSNetCLIP(nn.Module):
    
    def __init__(
        self,
        teacher_model: ViSNet,
        student_model: ViSNet,    
    ) -> None:
        super().__init__()
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        self.student_channels = self.student_model.representation_model.hidden_channels
        self.teacher_channels = self.teacher_model.representation_model.hidden_channels // 2
        self.mid_channels = (self.student_channels + self.teacher_channels) // 2
        self.share_head = nn.Sequential(
            nn.Linear(self.student_channels, self.student_channels),
            nn.SiLU(),
        )
        self.contrastive_output_head = nn.Sequential(
            nn.Linear(self.student_channels, self.mid_channels),
            nn.SiLU(),
            nn.Linear(self.mid_channels, self.teacher_channels),
        )
        self.energy_output_head = nn.Sequential(
            nn.Linear(self.student_channels, self.student_channels // 2),
            nn.SiLU(),
            nn.Linear(self.student_channels // 2, 1),
        )
        
        self.reset_parameters()
        self.freeze_verbose_params()
        
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.share_head[0].weight)
        self.share_head[0].bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.contrastive_output_head[0].weight)
        self.contrastive_output_head[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.contrastive_output_head[2].weight)
        self.contrastive_output_head[2].bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.energy_output_head[0].weight)
        self.energy_output_head[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.energy_output_head[2].weight)
        self.energy_output_head[2].bias.data.fill_(0)
        
    def freeze_verbose_params(self):
        
        print("Freeze the unused output head params...")
        for params in self.student_model.output_model.out_scalar_netowrk.parameters():
            params.requires_grad = False
        
    def forward(self, data: Data, stage="train", **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        
        out_rdkit, _ = self.student_model(data, use_pos_kind="rdkit")
        out_rdkit = self.share_head(out_rdkit)
        verify_eq = None
        if self.teacher_model is not None and stage == "train":
            with torch.no_grad():
                out_eq, _ = self.teacher_model(data, use_pos_kind="eq")
                verify_eq = self.teacher_model.output_model.post_reduce(out_eq)
                verify_eq = verify_eq * self.teacher_model.std + self.teacher_model.mean
                out_eq = self.teacher_model.output_model.out_scalar_netowrk[0](out_eq)
        else:
            out_eq = None
                
        pred_rdkit = self.energy_output_head(out_rdkit)
        out_rdkit = self.contrastive_output_head(out_rdkit)
        
        return out_eq, out_rdkit, pred_rdkit, verify_eq
        