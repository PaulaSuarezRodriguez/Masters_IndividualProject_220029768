from Third_implementation_attempt.Config_v3 import set_seed
set_seed()
from Third_implementation_attempt.ResNet20_v3 import ResNet20

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset

avalanche_train_dataset = AvalancheDataset(
    train_dataset
)

scenario = nc_benchmark(
    nature_dataset,
    n_experiences=3
)
resnet20_global_model = ResNet20()