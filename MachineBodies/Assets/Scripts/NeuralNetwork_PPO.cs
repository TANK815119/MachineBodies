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

    // Initialization and PPO-specific methods
    public NeuralNetwork_PPO(int inputSize, int outputSize, int hiddenLayers, int hiddenUnits)
    {
        policyNetwork = new NeuralNetwork(inputSize, outputSize, hiddenLayers, hiddenUnits);
        valueNetwork = new NeuralNetwork(inputSize, 1, hiddenLayers, hiddenUnits); //could have different dimensions
                                                                                   // Set default values for hyperparameters
        policyLearningRate = 0.001f;
        valueLearningRate = 0.001f;
        clipRange = 0.2f;
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

