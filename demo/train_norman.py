import sys
sys.path.append('../')

from gears_new import PertData, GEARS

seed=2
pert_data = PertData('./data')
pert_data.load(data_name = 'norman')
pert_data.prepare_split(split = 'custom', split_dict_path=f'./data/norman/splits/custom_{seed}.pkl')
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
baseline = None

gears_model = GEARS(pert_data, device = 'cuda:7',
                        weight_bias_track = True,
                        proj_name = 'pertnet',
                        exp_name = 'no_pert_custom_'+ str(seed))

gears_model.model_initialize(hidden_size = 64,
                             no_perturb=True,
                             baseline=baseline)

gears_model.train(epochs = 0, lr = 1e-3)
