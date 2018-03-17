# Deep RL HW2

# Command Line Parameters

--env [MountainCar-v0, CartPole-v0]
--qnmodel [linear, dqn, ddqn]
--mode [test, train]
--wfile path_to_weight_file
--rmem [true, false]

# Example Commands
To train on a linear network on MountainCar-v0 with no replay memory
python 2.7 DQN_Implementation.py --env MountainCar-v0 --qnmodel linear --mode train --rmem false

To get test information such as average rewards, standard deviation and video of performance of Q network at certain point
python 2.7 DQN_Implementation.py --env MountainCar-v0 --qnmodel linear --mode test --wfile /path/to/weight_file/
