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
        valueLearningRate = 0.0001f;
        clipRange = 0.2f;
    }

    public void Train(List<Experience> experiences)
    {
        List<float> advantages = CalculateAdvantages(experiences);
        UpdatePolicyNeuralNetwork(experiences, advantages);
        UpdateValueNeuralNetwork(experiences);
    }

    private void UpdatePolicyNeuralNetwork(List<Experience> experiences, List<float> advantages)
    {
        string str_2 = "advantages: ";
        for (int i = 0; i < advantages.Count; i++)
        {
            str_2 += advantages[i] + ", ";
        }
        Debug.Log(str_2);

        // Initialize variables for accumulating gradients
        float[] totalGradients = new float[policyNetwork.ReadNeuralNetwork().Length];

        //string str_1 = "totalGradients blank: ";
        //for (int i = 0; i < totalGradients.Length; i++)
        //{
        //    str_1 += totalGradients[i] + ", ";
        //}
        //Debug.Log(str_1);

        // Loop through experiences and calculate the policy loss
        for (int i = 0; i < experiences.Count; i++)
        {
            // Get the current state from the experience
            float[] state = experiences[i].State;

            // Calculate the probability of the taken actions under the current policy
            float[] actionProbs = policyNetwork.ForwardPass(state);

            //string str0c = "actionProbs: ";
            //for (int k = 0; k < actionProbs.Length; k++)
            //{
            //    str0c += actionProbs[k] + ", ";
            //}
            //Debug.Log(str0c);

            // Calculate the policy loss for each action
            float[] policyLoss = new float[actionProbs.Length];
            for (int j = 0; j < policyLoss.Length; j++)
            {
                policyLoss[j] = -Mathf.Log(actionProbs[j]) * advantages[i];
            }

            //string str0b = "policylosss: ";
            //for (int k = 0; k < policyLoss.Length; k++)
            //{
            //    str0b += policyLoss[k] + ", ";
            //}
            //Debug.Log(str0b);

            // Compute the gradients for the policy network based on the policy loss
            float[] instantGradients = policyNetwork.BackPropogate(policyLoss, experiences[i].State);

            // Accumulate the gradients
            for (int k = 0; k < totalGradients.Length; k++)
            {
                totalGradients[k] += instantGradients[k];
            }

            //string str0a = "backpropogated instant gradients: ";
            //for (int k = 0; k < instantGradients.Length; k++)
            //{
            //    str0a += instantGradients[k] + ", ";
            //}
            //Debug.Log(str0a);
        }

        string str0 = "totalGradients Network: ";
        for (int i = 0; i < totalGradients.Length; i++)
        {
            str0 += totalGradients[i] + ", ";
        }
        Debug.Log(str0);

        // Average the gradients
        for (int i = 0; i < totalGradients.Length; i++)
        {
            totalGradients[i] /= experiences.Count;
        }

        string str1 = "totalavergaeGradients Network: ";
        for (int i = 0; i < totalGradients.Length; i++)
        {
            str1 += totalGradients[i] + ", ";
        }
        Debug.Log(str1);

        // Update the policy network parameters using the averaged gradients
        float[] policyNetworkParameters = policyNetwork.ReadNeuralNetwork();

        //string str2 = "policyNetworkParameters Network: ";
        //for (int i = 0; i < policyNetworkParameters.Length; i++)
        //{
        //    str2 += policyNetworkParameters[i] + ", ";
        //}
        //Debug.Log(str2);

        for (int i = 0; i < totalGradients.Length; i++)
        {
            policyNetworkParameters[i] -= policyLearningRate * totalGradients[i];
        }

        string str3 = "Writing Policy Network: ";
        for (int i = 0; i < policyNetworkParameters.Length; i++)
        {
            str3 += policyNetworkParameters[i] + ", ";
        }
        Debug.Log(str3);

        policyNetwork.WriteNeuralNetwork(policyNetworkParameters);
    }

    private void UpdateValueNeuralNetwork(List<Experience> experiences)
    {
        // Compute value loss and update value network
        float discountFactor = 0.99f;
        //float totalLoss = 0f; -- USE THE TOTALLOSS HERE IN THE FUTURE TO RECORD PROGRESS
        float[] totalGradients = new float[valueNetwork.ReadNeuralNetwork().Length]; //an array that can hold the gradients of all weights and biases from backpropagation

        // loop through the experiences to find the error between predicted and actual reward
        for (int i = 0; i < experiences.Count; i++)
        {
            //find the predicted reward
            float predictedValue = valueNetwork.ForwardPass(experiences[i].State)[0];

            // Compute the discounted actual reward
            float actualValue = 0f;
            float discount = 1f;
            for (int j = i; j < experiences.Count; j++)
            {
                actualValue += experiences[j].Reward * discount;
                discount *= discountFactor;
            }

            //calculate the instanteneous loss function
            float gain = predictedValue - actualValue;
            float lossFunction = Mathf.Pow(gain, 2f);
            //totalLoss += lossFunction;

            //compute the instantenuos gradient change to each weight and bias using backpropogation and add to total gradients
            float[] instantGradients = valueNetwork.BackPropogate(new float[] { lossFunction }, experiences[i].State);
            for(int j = 0; j < totalGradients.Length; j++)
            {
                totalGradients[j] += instantGradients[j];
            }
        }

        //average the gradients
        for(int i = 0; i < totalGradients.Length; i++)
        {
            totalGradients[i] /= experiences.Count;
        }

        //update the value network parameters
        float[] valueNetworkParameters = valueNetwork.ReadNeuralNetwork();
        for(int  i = 0; i < totalGradients.Length; i++)
        {
            valueNetworkParameters[i] -= valueLearningRate * totalGradients[i];
        }

        valueNetwork.WriteNeuralNetwork(valueNetworkParameters);

        string str3 = "Writing Value Network: ";
        for (int i = 0; i < valueNetworkParameters.Length; i++)
        {
            str3 += valueNetworkParameters[i] + ", ";
        }
        Debug.Log(str3);
    }

    private List<float> CalculateAdvantages(List<Experience> experiences)
    {
        // Compute advantage estimates as (real reward - expected reward from value network)

        List<float> advantages = new List<float>();

        for (int i = 0; i < experiences.Count; i++)
        {
            // Get expected value from the value network for the current state
            float expectedValue = valueNetwork.ForwardPass(experiences[i].State)[0];

            // Compute the advantage as real reward - expected value
            float advantage = experiences[i].Reward - expectedValue;

            // Add the computed advantage to the list
            advantages.Add(advantage);
        }

        return advantages;
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

    public void SetPolicyNetwork(NeuralNetwork neuralNetwork)
    {
        policyNetwork = neuralNetwork;
    }

    public NeuralNetwork GetPolicyNeuralNetwork()
    {
        return policyNetwork;
    }

    public void InnitializeNeuralNetworksHe()
    {
        policyNetwork.InitializeWeightsHe();
        valueNetwork.InitializeWeightsHe();
    }

    public void SetLearningRates(float policy, float value) // reccomended 0.001f, 0.0001f
    {
        policyLearningRate = policy;
        valueLearningRate = value;
    }

    //[Serializable]
    //public class Experience
    //{
    //    public float[] State { get; private set; }
    //    public float[] Action { get; private set; }
    //    public float Reward { get; private set; }
    //    public float[] NextState { get; private set; }
    //    public bool Done { get; private set; }

    //    public Experience(float[] state, float[] action, float reward, float[] nextState, bool done)
    //    {
    //        State = state;
    //        Action = action;
    //        Reward = reward;
    //        NextState = nextState;
    //        Done = done;
    //    }
    //}
}

