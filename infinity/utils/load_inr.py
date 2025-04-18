import torch
from infinity.fourier_features import ModulatedFourierFeatures
from infinity.siren import ModulatedSiren


def create_inr_instance(cfg, input_dim=3, output_dim=1, device="cuda"):
    # data_path = "/data/serrano/functa2functa/airfrans/inr/"

    device = torch.device(device)

    if cfg.inr.model_type == "siren":
        inr = ModulatedSiren(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
        ).to(device)

    elif cfg.inr.model_type == "fourier_features":
        inr = ModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            frequency_embedding=cfg.inr.frequency_embedding,
            include_input=cfg.inr.include_input,
            scale=cfg.inr.scale,
            max_frequencies=cfg.inr.max_frequencies,
            base_frequency=cfg.inr.base_frequency,
        ).to(device)

    else:
        raise NotImplementedError(f"The model {cfg.inr.model_type} is not implemented")

    return inr


def load_inr_model(run_dir, run_name, input_dim=2, output_dim=1, device="cuda"):
    # data_path with the following template: "/data/serrano/functa2functa/airfrans/inr/"
    # run_dir with the following template: f"{data_path}/{data_to_encode}/
    inr_train = torch.load(run_dir / f"{run_name}.pt")

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    alpha = inr_train["alpha"]

    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    inr.load_state_dict(inr_state_dict)
    inr.eval()

    return inr, alpha
