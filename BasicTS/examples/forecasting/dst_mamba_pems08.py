from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DSTMamba import DSTMamba, DSTMambaConfig

if __name__ == '__main__':
    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=DSTMamba,
        model_config=DSTMambaConfig(num_channels=170, mamba_backend='official'),
        dataset_name='PEMS08', input_len=96, output_len=12,
        use_timestamps=False, gpus='0', batch_size=32,
        num_epochs=10, loss='MSE', lr=0.001,
    ))
