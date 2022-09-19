import os
import yaml

def get_config(config_name):
    with open(os.path.join(os.path.dirname(__file__), "..", "tasks", "config", "{}.yaml".format(config_name)),
              "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def get_latest_checkpoint_path(model_dir):
    '''
    get latest trained model
    '''
    fs = os.listdir(f'{model_dir}/sacred/')
    l = []
    for i in fs:
        try:
            l.append(int(i))
        except:
            pass
    return sorted(l)[-1]