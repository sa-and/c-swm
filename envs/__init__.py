from gym.envs.registration import register
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators import generate_task
from gym.utils import seeding

register(
    'ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'CubesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

register(
    'CausalWorld-v0',
    entry_point=CausalWorld,
    kwargs={'task': generate_task('stacking2'),
            'enable_visualization': False,
            'observation_mode': 'pixel',
            'camera_indicies': [0],
            'max_episode_length': 49})

register(
    'CausalWorld-v1',
    entry_point=CausalWorld,
    kwargs={'task': generate_task('stacking2'),
            'enable_visualization': False,
            'max_episode_length': 49})
