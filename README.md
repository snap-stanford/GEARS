# GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations

This repository hosts the official implementation of GEARS, a method that can predict transcriptional response to both single and multi-gene perturbations using single-cell RNA-sequencing data from perturbational screens. 


<p align="center"><img src="https://github.com/snap-stanford/GEARS/blob/master/img/gears.png" alt="gears" width="900px" /></p>


### Installation 

Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and then do `pip install cell-gears`.

### [New] Updates in v0.1.1

- Fixed training breakpoint bug from v0.1.0
- Preprocessed dataloader now available for Replogle 2022 RPE1 and K562 essential datasets
- Added custom split, fixed no-test split

### Core API Interface

Using the API, you can (1) reproduce the results in our paper and (2) train GEARS on your perturbation dataset using a few lines of code.

```python
from gears import PertData, GEARS

# get data
pert_data = PertData('./data')
# load dataset in paper: norman, adamson, dixit.
pert_data.load(data_name = 'norman')
# specify data split
pert_data.prepare_split(split = 'simulation', seed = 1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# set up and train a model
gears_model = GEARS(pert_data, device = 'cuda:8')
gears_model.model_initialize(hidden_size = 64)
gears_model.train(epochs = 20)

# save/load model
gears_model.save_model('gears')
gears_model.load_pretrained('gears')

# predict
gears_model.predict([['CBL', 'CNN1'], ['FEV']])
gears_model.GI_predict(['CBL', 'CNN1'], GI_genes_file=None)
```

To use your own dataset, create a scanpy adata object with a `gene_name` column in `adata.var`, and two columns `condition`, `cell_type` in `adata.obs`. Then run:

```python
pert_data.new_data_process(dataset_name = 'XXX', adata = adata)
# to load the processed data
pert_data.load(data_path = './data/XXX')
```

### Demos

| Name | Description |
|-----------------|-------------|
| [Dataset Tutorial](demo/data_tutorial.ipynb) | Tutorial on how to use the dataset loader and read customized data|
| [Model Tutorial](demo/model_tutorial.ipynb) | Tutorial on how to train GEARS |
| [Plot top 20 DE genes](demo/tutorial_plot_top20_DE.ipynb) | Tutorial on how to plot the top 20 DE genes|
| [Uncertainty](demo/tutorial_uncertainty.ipynb) | Tutorial on how to train an uncertainty-aware GEARS model |


### Colab

| Name | Description |
|-----------------|-------------|
| [Using Trained Model](https://colab.research.google.com/drive/11LlzGEUGoBk_Uj6DzlzizAeWse5_E9MK?usp=sharing) | Use a model trained on Norman et al. 2019 to make predictions (Needs Colab Pro)|



### Cite Us

```
@article{roohani2023predicting,
  title={Predicting transcriptional outcomes of novel multigene perturbations with gears},
  author={Roohani, Yusuf and Huang, Kexin and Leskovec, Jure},
  journal={Nature Biotechnology},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
```
Paper: [Link](https://www.nature.com/articles/s41587-023-01905-6)

Code for reproducing figures: [Link](https://github.com/yhr91/gears_misc)
