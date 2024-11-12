using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork_Heuristic
{
    private NeuralNetwork policyNetwork;
    private float policyLearningRate;

    public NeuralNetwork_Heuristic(int inputSize, int outputSize, int hiddenBreadth, int hiddenHeight)
    {
        policyNetwork = new NeuralNetwork(inputSize, outputSize, hiddenBreadth, hiddenHeight);

        policyLearningRate = 0.001f;
    }

    public void Train(List<Experience> experiences)
    {
        UpdatePolicyNeuralNetwork(experiences);
    }

    private void UpdatePolicyNeuralNetwork(List<Experience> experiences)
    {
        // Compares Neural network behavior to custom, perfect, code to cbeck that backpropogation is working
        float[] totalGradients = new float[policyNetwork.ReadNeuralNetwork().Length]; //an array that can hold the gradients of all weights and biases from backpropagation

        // loop through the experiences to find the error between predicted and actual reward
        for (int i = 0; i < experiences.Count; i++)
        {
            //find the network's output
            float predictedPolicy = policyNetwork.ForwardPass(experiences[i].State)[0];

            // Compute the discounted actual reward
            float actualPolicy = BespokePolicy(experiences); //THIS IS WHERE CUSTOM CODE GOES

            //calculate the instanteneous loss function
            float gain = predictedPolicy - actualPolicy;
            //float lossFunction = Mathf.Pow(gain, 2f);
            //totalLoss += lossFunction;

            //compute the instantenuos gradient change to each weight and bias using backpropogation and add to total gradients
            float[] instantGradients = policyNetwork.BackPropogate(new float[] { gain }, experiences[i].State);
            for (int j = 0; j < totalGradients.Length; j++)
            {
                totalGradients[j] += instantGradients[j];
            }
        }

        //average the gradients
        for (int i = 0; i < totalGradients.Length; i++)
        {
            totalGradients[i] /= experiences.Count;
        }

        //update the value network parameters
        float[] valueNetworkParameters = policyNetwork.ReadNeuralNetwork();
        for (int i = 0; i < totalGradients.Length; i++)
        {
            valueNetworkParameters[i] -= policyLearningRate * totalGradients[i];
        }

        policyNetwork.WriteNeuralNetwork(valueNetworkParameters);
    }

    private float BespokePolicy(List<Experience> experiences)
    {
        //should return the proper thrust
        return 10f;
    }

    public void InnitializeNeuralNetworksHe()
    {
        policyNetwork.InitializeWeightsHe();
    }

    public void SetLearningRates(float policy) // reccomended 0.001f
    {
        policyLearningRate = policy;
    }

    public void SetPolicyNetwork(NeuralNetwork neuralNetwork)
    {
        policyNetwork = neuralNetwork;
    }

    public NeuralNetwork GetPolicyNeuralNetwork()
    {
        return policyNetwork;
    }
}
