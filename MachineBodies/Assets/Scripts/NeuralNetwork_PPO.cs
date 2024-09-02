using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class NeuralNetwork_PPO
{
    private NeuralNetwork policyNetwork;
    private NeuralNetwork valueNetwork;
    private float policyLearningRate;
    private float valueLearningRate;
    private float clipRange;
    private List<Experience> experiences;

    // Initialization and PPO-specific methods
    public NeuralNetwork_PPO(int inputSize, int outputSize, int hiddenLayers, int hiddenUnits)
    {
        policyNetwork = new NeuralNetwork(inputSize, outputSize, hiddenLayers, hiddenUnits);
        valueNetwork = new NeuralNetwork(inputSize, 1, hiddenLayers, hiddenUnits); //could have different dimensions
                                                                                   // Set default values for hyperparameters
        policyLearningRate = 0.001f;
        valueLearningRate = 0.001f;
        clipRange = 0.2f;

        experiences = new List<Experience>();
    }

    public void Train(List<Experience> experiences)
    {
        List<float> advantages = CalculateAdvantages(experiences);
        UpdatePolicy(experiences, advantages);
        UpdateValue(experiences);
    }

    private void UpdatePolicy(List<Experience> experiences, List<float> advantages)
    {
        // Compute policy loss and update policy network
    }

    private void UpdateValue(List<Experience> experiences)
    {
        // Compute value loss and update value network
        //float discountFactor = 0.99f;
        float totalLoss = 0f;
        float[] totalGradients = new float[valueNetwork.BackPropogate(new float[] { 0f }).Length]; //an array that can hold the gradients of all weights and biases from backpropagation

        // loop through the experiences to find the error between predicted and actual reward
        for (int i = 1; i < experiences.Count; i++)
        {
            //find the predicted reward
            float predictedValue = policyNetwork.ForwardPass(experiences[i].State)[0];

            //gather the actual reward
            float actualValue = experiences[i].Reward;
            //actualValued += discountFactor * policyNetwork.ForwardPass(experiences[i + 1].State)[0]; //I dont understand if im doing this line right or if its a necessity

            //calculate the instanteneous loss function
            float gain = predictedValue - actualValue;
            float lossFunction = Mathf.Pow(gain, 2f);
            totalLoss += lossFunction;

            //compute the instantenuos gradient change to each weight and bias using backpropogation and add to total gradients
            float[] instantGradients = valueNetwork.BackPropogate(new float[] { lossFunction });
            for(int j = 0; j < totalGradients.Length; j++)
            {
                totalGradients[j] += instantGradients[j];
            }
        }

        //average the gradients
        for(int i = 0; i < totalGradients.Length; i++)
        {
            totalGradients[i] /= experiences.Count - 1;
        }

        //update the value network parameters
        float[] valueNetworkParameters = valueNetwork.ReadNeuralNetwork();
        for(int  i = 0; i < totalGradients.Length; i++)
        {
            valueNetworkParameters[i] -= valueLearningRate * totalGradients[i];
        }
    }

    private List<float> CalculateAdvantages(List<Experience> experiences)
    {
        // Compute advantage estimates
        return new List<float>();
    }

    public float EvaluatePolicy(float[] state)
    {
        return policyNetwork.ForwardPass(state)[0];
    }

    public float EvaluateValue(float[] state)
    {
        return valueNetwork.ForwardPass(state)[0];
    }

    public void SetLearningRate(float policyLR, float valueLR)
    {
        policyLearningRate = policyLR;
        valueLearningRate = valueLR;
    }

    public void SetClipRange(float clip)
    {
        clipRange = clip;
    }

    [Serializable]
    public class Experience
    {
        public float[] State { get; private set; }
        public float[] Action { get; private set; }
        public float Reward { get; private set; }
        public float[] NextState { get; private set; }
        public bool Done { get; private set; }

        public Experience(float[] state, float[] action, float reward, float[] nextState, bool done)
        {
            State = state;
            Action = action;
            Reward = reward;
            NextState = nextState;
            Done = done;
        }
    }
}

