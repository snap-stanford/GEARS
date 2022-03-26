# PertNet

This repository hosts the official implementation of PertNet, a geometric deep learning tool for perturbation prediction. For detailed information, see our paper [XXX]().


### Installation 

Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and then do `pip install pertnet`.

### API Interface

```python
from pertnet import PertData, PertNet

# get data
pert_data = PertData('./data')
pert_data.load(dataset = 'norman')
pert_data.prepare_split(split = 'simulation', seed = 1)
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# set up and train a model
pertnet_model = PertNet(pert_data, device = 'cuda:8')
pertnet_model.model_initialize(hidden_size = 64)
pertnet_model.train(epochs = 20)

# save/load model
pertnet_model.save_model('pertnet')
pertnet_model.load_pretrained('pertnet')

# predict
pertnet_model.predict([['FOX1A', 'AHR'], ['FEV']])
pertnet_model.GI_predict([['FOX1A', 'AHR'], ['FEV', 'AHR']])
```

To use your own dataset, create a scanpy adata variable with a `gene_name` column in `adata.var`, and two columns `condition`, `cell_type` in `adata.obs`. Then run:

```python
pert_data.new_data_process(dataset_name = 'XXX', adata = adata)
# to load the processed data
pert_data.load(data_path = './data/XXX')
```

See [demo](demo) folder for examples.


### Cite Us

```

```