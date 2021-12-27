import base64
import pickle
import torch


if __name__ == "__main__":
    model_path = "/home/zhangyp-s21/projects/carbon_baseline_cuda/models/best"  # your model file path

    model_state_dict = torch.load(model_path)
    # actor_model = model_state_dict['actor']

    # for name, param in actor_model.items():
    #     actor_model[name] = param.numpy()

    model_byte = base64.b64encode(pickle.dumps(model_state_dict))
    with open("actor.txt", 'wb') as f:
        f.write(model_byte)
    pass
