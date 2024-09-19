import os

def test_and_limit_tensorflow():
    import tensorflow as tf
    print('GPU Available ', tf.test.is_gpu_available())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_limit)]
            )
        except RuntimeError as e:
            print(e)


def download_model(model_link, file_path = './'):
    os.system(f'gdown {model_link} -O {file_path}')
