from gym.envs.registration import register 

register(
    id='gym-test-v0',
    entry_point='gym_gazebo_px4.envs:TestEnv',
    max_episode_steps = 400,
) 

register(
    id='gym-px4-v0',
    entry_point='gym_gazebo_px4.envs:PX4Env',
    max_episode_steps = 400,
)