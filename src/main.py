"""
Main script for DARKSKIES Bayesian neural network
"""
import os
from typing import Any

import torch
import numpy as np
import pandas as pd
import sciplots as plots
import netloader.networks as nets
from netloader import transforms
from netloader.network import Network
from netloader.utils.utils import get_device, save_name
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from src.utils import analysis
from src.utils.utils import open_config
from src.utils.data import DarkDataset, loader_init
from src.utils.clustering import CompactClusterEncoder


def net_init(dataset: DarkDataset, config: str | dict = '../config.yaml') -> nets.BaseNetwork:
    """
    Initialises the network

    Parameters
    ----------
    dataset : DarkDataset
        Dataset with inputs and outputs
    config : string | dictionary, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    BaseNetwork
        Constructed network
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    save_num = config['training']['network-save']
    load_num = config['training']['network-load']
    learning_rate = config['training']['learning-rate']
    name = config['training']['network-name']
    description = config['training']['description']
    networks_dir = config['data']['network-configs-directory']
    states_dir = config['output']['network-states-directory']
    device = get_device()[1]

    # Initialise network
    if load_num:
        net = nets.load_net(load_num, states_dir, name, weights_only=False)
        net.save_path = save_name(save_num, states_dir, name) if save_num else ''
        transform = net.transforms['inputs']
        param_transform = net.transforms['targets']
    else:
        transform = transforms.MultiTransform(
            transforms.NumpyTensor(),
            transforms.Index(0, dataset.images.shape[1:], slice(3)),
        )
        param_transform = transforms.MultiTransform(
            transforms.NumpyTensor(),
            transforms.Log(),
        )
        param_transform.append(transforms.Normalise(data=param_transform(dataset.labels[np.isin(
            dataset.labels,
            np.unique(dataset.labels)[len(dataset.unknown):],
        )]), mean=False))
        net = Network(
            name,
            networks_dir,
            list(dataset[0][2].shape),
            [len(np.unique(dataset.labels))],
        )
        net = CompactClusterEncoder(
            save_num,
            states_dir,
            torch.unique(param_transform(dataset.labels)),
            net,
            unknown=len(dataset.unknown),
            learning_rate=learning_rate,
            description=description,
            verbose='progress',
            transform=param_transform,
            in_transform=transform,
        )

    # Initialise datasets
    dataset.images = transform(dataset.images)
    dataset.labels = param_transform(dataset.labels)
    return net.to(device)


def init(
        known: list[str],
        config: str | dict[str, Any] = '../config.yaml',
        unknown: list[str] | None = None,
) -> tuple[
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork,
        DarkDataset]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    known : list[str]
        Simulations to train with known labels
    config : str | dict[str, Any], default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary
    unknown : list[str] | None, default = None
        Simulations to train with unknown labels

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], BaseNetwork, DarkDataset]
        Train & validation dataloaders, neural network and dataset
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    batch_size = config['training']['batch-size']
    val_frac = config['training']['validation-fraction']
    data_dir = config['data']['data-dir']

    if unknown is None:
        unknown = []

    # Fetch dataset & network
    dataset = DarkDataset(data_dir, known, unknown)
    net = net_init(dataset, config)

    # Initialise data loaders
    loaders = loader_init(dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    net.idxs = dataset.idxs
    return loaders, net, dataset


def main(config_path: str = '../config.yaml'):
    """
    Main function for training and analysis of the network

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        Path to the configuration file
    """
    _, config = open_config('main', config_path)

    net_epochs = config['training']['epochs']
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    bahamas_colours = ['#0049E0', '#0090E0', '#00D7E0', '#2CDEE6', '#00E09E', '#00E051'][:-2]
    bahamas_agn_colours = ['#F54EDF', '#5D4EF5']
    bahamas_dmo = ['#00FA8F', '#01FB3D', '#89FA00']
    flamingo_colours = ['#FABD00', '#FA2100', '#FA7700']
    flamingo_test = ['#FA07A0']
    colours = ['k'] + flamingo_test + bahamas_dmo + flamingo_colours[1:2] + bahamas_colours
    param_names = [
        r'$\sigma_{\rm DM}$',
        '$M$',
        'Stellar Frac',
        r'$\Delta T$',
        r'$m_{\rm DM}$',
        '$m_b$',
    ]
    known = [
        'flamingo',
        # 'flamingo_low',
        # 'flamingo_hi',
        'bahamas_cdm',
        # 'bahamas_cdm_low',
        # 'bahamas_cdm_hi',
        'bahamas_0.1',
        'bahamas_0.3',
        'bahamas_1',
    ]
    unknown = [
        'noise',
        'flamingo_low_test',
        'bahamas_dmo_cdm',
        'bahamas_dmo_0.1',
        'bahamas_dmo_1',
    ]

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Initialise network
    torch.serialization.add_safe_globals([CompactClusterEncoder])
    loaders, net, dataset = init(known, config, unknown=unknown)

    # Train network
    net.training(net_epochs, loaders)

    # Generate predictions
    data = net.predict(loaders[1])
    data['targets'] = dataset.correct_unknowns(data['targets'].squeeze())
    data['targets'] = data['targets'].squeeze()
    labels = dataset.names[data['ids'][np.unique(
        data['targets'],
        return_index=True,
    )[1]].astype(int)]

    # Plot performance
    plots.PlotPerformance(
        np.array(net.losses),
        x_label='Epoch',
        y_label='Loss',
        labels=['Train', 'Validation'],
        colours=['k', '#0049E0'],
    ).savefig(plots_dir, 'losses')

    # Plot cluster profiles
    names, radii, total, x_ray, stellar = analysis.profiles(
        net.in_transform(dataset.images, back=True),
        dataset.norms,
        dataset.names,
    )
    plots.PlotPlots(
        radii,
        total,
        log_x=True,
        log_y=True,
        x_label='Radius',
        y_label='Total',
        labels=names.tolist(),
    ).savefig(plots_dir, name='total_mass')
    plots.PlotPlots(
        radii,
        x_ray,
        log_x=True,
        log_y=True,
        x_label='Radius',
        y_label='X-Ray Frac',
        labels=names.tolist(),
    ).savefig(plots_dir, name='x-ray_frac')
    plots.PlotPlots(
        radii,
        stellar,
        log_x=True,
        log_y=True,
        x_label='Radius',
        y_label='Stellar Frac',
        labels=names.tolist(),
    ).savefig(plots_dir, name='stellar_frac')

    # Plot distributions
    distributions = analysis.pred_distributions(
        data['targets'],
        net.transforms['targets'](data['latent'], back=True)[:, 0],
    )
    plots.PlotDistribution(
        distributions,
        log=True,
        norm=True,
        y_axes=False,
        density=True,
        axis_pad=False,
        bins=200,
        x_label=r'Predicted $\sigma_{\rm DM}\ \left(\rm cm^2\ g^{-1}\right)$',
        labels=labels,
        colours=colours,
        axis=True,
        rows=len(labels),
        loc='best',
    ).savefig(plots_dir)
    plot = plots.PlotDistributions(
        distributions,
        log=True,
        norm=True,
        y_axes=False,
        density=True,
        titles=labels,
        colours=[bahamas_colours[0], flamingo_colours[-1]],
    )
    plot.plot_twin_data(np.unique(data['targets']) - 0.5)
    plot.savefig(plots_dir)

    # Plot latent dims and physical params comparisons
    latents, params = analysis.phys_params(data, dataset.names, dataset.stellar_frac, dataset.mass)
    plots.PlotPearson(
        np.concat(latents),
        np.concat(params),
        x_labels=[f'Dim {i}' for i in range(latents[0].shape[-1])],
        y_labels=param_names,
    ).savefig(plots_dir)
    plots.PlotParamPairComparison(
        latents,
        params,
        density=True,
        labels=labels,
        x_labels=[f'Dim {i}' for i in range(latents[0].shape[-1])],
        y_labels=param_names,
        colours=colours,
    ).savefig(plots_dir)
    plots.PlotParamPairs(
        params,
        density=True,
        labels=np.unique(dataset.names).tolist(),
        axes_labels=param_names,
        colours=colours,
    ).savefig(plots_dir, name='params')
    plots.PlotPearson(
        np.concat(params),
        np.concat(params),
        x_labels=param_names,
        y_labels=param_names,
    ).savefig(plots_dir, name='params_pearson')

    # Plot predictions
    data['latent'][:, 0] *= 1e6
    pca = PCA(n_components=4).fit(data['latent'][np.isin(
        data['targets'],
        np.unique(data['targets'])[net._unknown:],
    )])
    pca_transform = pca.transform(data['latent'])
    pca_transform[:, 0] /= 1e6
    data['latent'][:, 0] /= 1e6
    plots.PlotClusters(
        pca_transform,
        data['targets'],
        density=True,
        labels=labels,
        alpha=0.1,
        alpha_2d=0.2,
        colours=colours,
        rows=len(labels),
        loc='upper right',
    ).savefig(plots_dir, name='PCA')
    plots.PlotClusters(
        data['latent'],
        data['targets'],
        density=True,
        alpha=0.1,
        alpha_2d=0.2,
        labels=labels,
        colours=colours,
        rows=len(labels),
        loc='upper right',
    ).savefig(plots_dir, name='clusters')
    plots.PlotConfusion(labels, data['preds'], data['targets']).savefig(plots_dir)

    # Plot saliencies
    saliency = net.saliency(loaders[1], net)
    plots.PlotSaliency(saliency['inputs'][0, 0], saliency['saliencies'][0, :, 0]).savefig(plots_dir)

    # Print predicted cross-sections
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 500)
    meds, means, stes = analysis.summary(data)[:3]
    meds = net.transforms['targets'](meds, back=True)
    means, stes = net.transforms['targets'](means, back=True, uncertainty=stes)
    print(pd.DataFrame(
        [meds, means, stes],
        index=['Medians', 'Means', 'STEs'],
        columns=[label.replace(r'\sigma=', '').replace('$', '') for label in labels],
    ).round(3))


if __name__ == '__main__':
    main()
