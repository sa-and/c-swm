import modules
import torch
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    num_objects = 4

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-folder', type=str,
                        default='checkpoints/cw_large2/',
                        help='folder string')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disable CUDA training.')
    parser.add_argument('--data-file', type=str, default='data/example_stacking2_02.pkl', help='Path of obs file.')

    args_eval = parser.parse_args()

    meta_file = os.path.join(args_eval.model_folder, 'metadata.pkl')
    model_file = os.path.join(args_eval.model_folder, 'model.pt')

    args = pickle.load(open(meta_file, 'rb'))['args']

    args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
    args.batch_size = 100
    args.seed = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    #load image
    file = pickle.load(open(args_eval.data_file, 'rb'))[None, :]
    image = torch.tensor(file).swapaxes(1, 3).to(device)

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=(3, 64, 64),
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    extracted_obj = model.obj_extractor(image.to(torch.float32))
    plt.subplot(161)
    plt.imshow(image[0][0])
    plt.axis("off")
    for i in range(num_objects):
        plt.subplot(162+i)
        plt.imshow(extracted_obj[0][i].detach())
        plt.axis("off")
    plt.show()

# to create sample image
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import matplotlib.pyplot as plt
import pickle


def example():
    task = generate_task(task_generator_id='stacking2')
    env = CausalWorld(task=task,
                      skip_frame=10,
                      enable_visualization=True,
                      seed=0,
                      action_mode="joint_positions",
                      observation_mode="pixel",
                      camera_indicies=[0, 1, 2])
    env.reset()
    for _ in range(50):
        obs, reward, done, info = env.step(env.action_space.sample())
    #show last images
    for i in range(6):
        with open('D:/phd/c-swm/data/example_stacking2_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(obs[i], f)
        plt.imshow(obs[i])
        plt.axis('off')
        plt.savefig('D:/phd/c-swm/data/example_stacking2_'+str(i)+'.png', bbox_inches='tight', pad_inches=0)
        plt.show()
    env.close()
#example()