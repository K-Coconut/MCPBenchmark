import numpy as np
import sys
import util
import time
import os
import graphEnv
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import nn
import evaluate_spread
import argparse

#30 20 1 2 0.0005 1 6

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    mu_selected = tf.placeholder(dtype=tf.double, shape=[None, 2], name='mu_selected')
    mu_left = tf.placeholder(dtype=tf.double, shape=[None, 2], name='mu_left')
    mu_v=tf.placeholder(dtype=tf.double, shape=[None, 2], name='mu_v')

    receiver_tensors = {'mu_selected': mu_selected, 'mu_left': mu_left,'mu_v':mu_v}
    features = {'mu_selected': mu_selected, 'mu_left': mu_left,'mu_v':mu_v}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# parameters for the code
args = argparse.ArgumentParser()
args.add_argument('-k', "--steps", type=int, help='number of steps')
args.add_argument("-n", "--numEps", type=int, help='number of episodes')
args.add_argument("-e", "--dimEmbedding", type=int, help='number of dimension of embeddings')
args.add_argument("-w", "--windowSize", type=int, help='size of window')
args.add_argument("-r", "--learningRate", type=float, help='learning rate')
args.add_argument("-p", "--numOfEpochs", type=int, help="number of training epochs in each iteration")
args.add_argument("-b", "--batchSize", type=int, help="batch size")
args.add_argument("-d", "--dataset", default='youtube',help="batch size")
args.add_argument("-m", "--weight_model", default='WT',help="weight model")
parser = args.parse_args()

print(parser)
k = parser.steps
numEps = parser.numEps
dimEmbedding = parser.dimEmbedding
windowSize = parser.windowSize
learningRate = parser.learningRate
numOfEpochs = parser.numOfEpochs
batchSize = parser.batchSize
dataset = parser.dataset
weight_model = parser.weight_model

model_dir_name = "./trained_model_MC_%s_%s" % (dataset, weight_model)
model_log_path = model_dir_name + "/model_log/"
bestValModel = ''
bestValReward = 0
# numSteps should be less than k
# window size must be much smaller than k to generate enough samples
historyOfTuples = []
start_time = time.time()
util.init(dataset, learningRate, numOfEpochs, batchSize, dimEmbedding, weight_model)
for episode in range(0,numEps):
    print("episode ", episode)
    # generate a new graph for each episode
    graph = util.Graph(dimEmbedding, episode, k, dataset, weight_model)
    graphEnv.graphEnvironment.append(graph)
    # print(graph.graphX.degree())
    # if episode==0:
    # 	util.initialze_weights(graph)

    # (numsteps == k) => Terminal Condition
    previous_spread = 0

    for step in range(0, k):
        print("step: ", step)
        # print("isSelected \n", graph.isSelected)

        # select node to be added
        probOfRandomSelection = max(pow(0.1, step), 0.8)
        # print("probOfRandomSelection is : ", probOfRandomSelection)


        if(step==0):
             action_t= util.getRandomNode(episode,step)#graphEnv.graphEnvironment[episode].top_tenpct_nodes[0] action_t=
# graphEnv.graphEnvironment[episode].top_tenpct_nodes[0]#util.getRandomNode(episode,step)#graphEnv.graphEnvironment[episode].top_tenpct_nodes[0]
        # index of the selected node
        else:
            action_t = util.getNode(probOfRandomSelection, episode, step)
        print("Node selected: ", action_t)

        # add action_t to the partial solution
        graphEnv.graphEnvironment[episode].isSelected[action_t] = step
        graphEnv.graphEnvironment[episode].isCounted[action_t] = True
        neighbors_of_chosen_node = graphEnv.graphEnvironment[episode].dict_node_sampled_neighbors[action_t]#neighbors(action_t))
        print("num nbrs of chosen node ", len(neighbors_of_chosen_node))
        new_neighbors_length = len(neighbors_of_chosen_node - graphEnv.graphEnvironment[episode].neighbors_chosen_till_now)
        graphEnv.graphEnvironment[episode].neighbors_chosen_till_now= graphEnv.graphEnvironment[episode].neighbors_chosen_till_now.union(neighbors_of_chosen_node )

        print(" new diff neighbors , ",new_neighbors_length)




        for node in graph.top_tenpct_nodes:
                neighbors_of_node = graphEnv.graphEnvironment[episode].dict_node_sampled_neighbors[node]#neighbors(node))
                new_neighbors_not_in_solutions_neighbors = neighbors_of_node - graphEnv.graphEnvironment[episode].neighbors_chosen_till_now

                graphEnv.graphEnvironment[episode].embedding_time[step+1][node][0] = len(new_neighbors_not_in_solutions_neighbors)



        scaler = StandardScaler()
        temp_column_for_cover=np.ones((len(graphEnv.graphEnvironment[episode].embedding_time[step+1]), 1), dtype='float64')

        i=0
        dict_map_i_key = {}
        for key, value in graphEnv.graphEnvironment[episode].embedding_time[step+1].items():
            temp_column_for_cover[i] =value[0]
            dict_map_i_key[i] = key
            i+=1


        scaler.fit(temp_column_for_cover)
        temp_column_for_cover_norm = scaler.transform(temp_column_for_cover)

        for index, value in enumerate(temp_column_for_cover_norm):
            true_node_id = dict_map_i_key[index]
            graphEnv.graphEnvironment[episode].embedding_time[step+1][true_node_id][0] = value
        print("seleected", graph.state)
        # returns the short term reward and updates the isCounted
        shortReward, previous_spread = util.getShortReward(action_t, episode,previous_spread, weight_model)#= new_neighbors_length
    #	previous_spread= shortReward
        print("Short reward for addition of ", action_t, "is ", shortReward)
        print(" new previous spread, ", previous_spread)
        graphEnv.graphEnvironment[episode].state.append(action_t)

        if (step==0):
            graphEnv.graphEnvironment[episode].cumulativeReward.append(shortReward)
        else :
            # print('*******************************************************************',shortReward)
            graphEnv.graphEnvironment[episode].cumulativeReward.append(graph.cumulativeReward[step-1] + shortReward)
        print(step, windowSize)
        if (step == windowSize):
            netShortReward = graphEnv.graphEnvironment[episode].cumulativeReward[step - 1]
        if (step > (windowSize)):
            netShortReward = graphEnv.graphEnvironment[episode].cumulativeReward[step-1] - graphEnv.graphEnvironment[episode].cumulativeReward[step - (windowSize)-1]

            # The short term reward does not include the reward by adding the step vertex
            # Action Tuples are of the form: (startIdx, nodeAdded, net cumulative reward, last index)
            mu_v_at_that_step_minus_window = graphEnv.graphEnvironment[episode].embedding_time[step-windowSize][action_t].reshape(dimEmbedding+1)

            mu_s,mu_l = util.createMuUtil(step-(windowSize), episode)
            actionTuple = (mu_v_at_that_step_minus_window,graph.state[step-(windowSize)],netShortReward,step, mu_s, mu_l, episode)

            historyOfTuples.append(actionTuple)
            if(len(historyOfTuples)>5) :
                util.updateParameters(historyOfTuples)
                print("saving")
                export_path=nn.model.export_savedmodel(model_dir_name, serving_input_receiver_fn)
                graph_path = "./GraphSAGE-master/real_data/%s/%s/train/"%(dataset, weight_model)
                command = "python get_output.py -p %s -k %d -f %f -d %s -m None --weight_model %s > /dev/null" %(graph_path, 50, 0.003, dataset, weight_model)
                print(command)
                os.system(command)

                rl_result_file = graph_path+'-result_RL_50_nbs_0.003'
                solution_file=open(rl_result_file, "r")

                optimal_nodes=solution_file.readlines()
                int_selected_nodes=[]

                for node_i in optimal_nodes:  # range(0, budget):
                    int_selected_nodes.append(int(node_i))

                print(" loading graph")
                num_mc_simulation =8


                spread=evaluate_spread.evaluate_helper_without_mp(graph_path, None, int_selected_nodes, num_mc_simulation)
                # print(" iter {}  spread {}".format(i, temp_spread))
                # spread = spread + temp_spread
                # spread = spread * 1.0 / num_mc_simulation
                print('Average Spread = ', spread)

                print('Spread = ', spread)
                reward_file_name=graph_path + "reward_RL_budget_" + str(50)
                reward_file=open(reward_file_name, 'w')
                reward_file.write('100mc' + str(spread))
                reward_file.close()
                print("read reward val ", spread)
                print(" best reward val ", bestValReward)
                if spread > bestValReward:
                    bestValReward = spread
                    bestValModel = export_path.decode()
                    print(" got a better one", bestValReward, bestValModel)
                    bestRLModel_logfile = model_dir_name + '/bestRlModel.txt'
                    fileBestRLModel = open(bestRLModel_logfile, 'w')
                    fileBestRLModel.write(bestValModel)
                    fileBestRLModel.close()

                global_step = nn.model.get_variable_value('global_step')
                if global_step % 100 == 0:
                    os.makedirs(model_log_path, exist_ok=True)
                    model_path = export_path.decode()
                    print("record the model that trained for %d epochs: " % global_step, model_path)
                    fileRLModel = open("./%s/model_epoch_%d.txt" % (model_log_path, global_step), 'w')
                    fileRLModel.write(model_path)
                    fileRLModel.close()

                    fileBestRLModel = open("./%s/best_model_epoch_%d.txt" % (model_log_path, global_step), 'w')
                    fileBestRLModel.write(bestValModel)
                    fileBestRLModel.close()

                    end_time = time.time()
                    print("Epoch %d elapsed time: %.2f s" % (global_step, end_time - start_time))
                    time_record = open("./%s/time_epoch_%d.txt" % (model_log_path, global_step), 'w')
                    time_record.write(str(end_time - start_time))
                    time_record.close()

                print("exp path ", export_path)
                print("best exp path ", bestValModel)

            if(len(historyOfTuples) > 3*k):
                historyOfTuples.pop(0)
