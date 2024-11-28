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
    private const float DiscountFactor = 0.95f;

    // Initialization and PPO-specific methods
    public NeuralNetwork_PPO(int inputSize, int outputSize, int hiddenBreadth, int hiddenHeight)
    {
        policyNetwork = new NeuralNetwork(inputSize, outputSize, hiddenBreadth, hiddenHeight, true);
        valueNetwork = new NeuralNetwork(inputSize, 1, hiddenBreadth, hiddenHeight, false); //could have different dimensions
                                                                                   // Set default values for hyperparameters
        policyLearningRate = 0.001f;
        valueLearningRate = 0.0001f;
        clipRange = 0.2f;
    }

    public void Train(List<Experience> experiences)
    {
        List<float> advantages = CalculateAdvantages(experiences);
        UpdatePolicyNeuralNetwork2(experiences, advantages);
        UpdateValueNeuralNetwork(experiences);
    }

    private void UpdatePolicyNeuralNetwork(List<Experience> experiences, List<float> advantages)
    {
        //string str_2 = "advantages: ";
        //for (int i = 0; i < advantages.Count; i++)
        //{
        //    str_2 += advantages[i] + ", ";
        //}
        //Debug.Log(str_2);

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

            string str0a = "backpropogated instant gradients: ";
            for (int k = 0; k < instantGradients.Length; k++)
            {
                str0a += instantGradients[k] + ", ";
            }
            Debug.Log(str0a);
        }



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

        //string str3 = "Writing Policy Network: ";
        //for (int i = 0; i < policyNetworkParameters.Length; i++)
        //{
        //    str3 += policyNetworkParameters[i] + ", ";
        //}
        //Debug.Log(str3);

        policyNetwork.WriteNeuralNetwork(policyNetworkParameters);
    }

    private void UpdatePolicyNeuralNetwork2(List<Experience> experiences, List<float> advantages)
    {
        // Compares Neural network behavior to its associated predicted value and uses backpropogation to update for the most effective network
        float[] totalGradients = new float[policyNetwork.ReadNeuralNetwork().Length]; //an array that can hold the gradients of all weights and biases from backpropagation

        // loop through the experiences to find the error between predicted and actual reward
        float failedGradients = 0f;
        float failedGradientArrays = 0f;
        for (int i = 0; i < experiences.Count; i++)
        {
            // Get the action taken from the experience(assumed to be a continuous action)
            float[] action = experiences[i].Action;
            float[] oldAction = experiences[i].OldAction;

            // Calculate the probability of the taken actions under the current policy
            float[] policyOutput = experiences[i].Output; //policyNetwork.ForwardPass(experiences[i].State);
            float[] oldPolicyOutput = experiences[i].OldOutput;

            // Calculate the policy loss gradients
            float[] policyLossGradients = new float[policyOutput.Length];
            for (int j = 0; j < policyOutput.Length / 2; j++)
            {
                // Extract mean and standard deviation for the current action
                float mean = policyOutput[j];
                float stdDev = 0.5f; //Mathf.Max(policyOutput[j + policyOutput.Length / 2], 1e-6f);
                float oldMean = oldPolicyOutput[j];
                float oldStdDev = 0.5f; //Mathf.Max(oldPolicyOutput[j + policyOutput.Length / 2], 1e-6f);

                // Get the current log-probabilities from the policy network for the taken actions
                float currentLogProb = CalculateLogProb(action[j], mean, stdDev);
                float oldLogProb = CalculateLogProb(oldAction[j], oldMean, oldStdDev);
                //Debug.Log("currentLogProb" + currentLogProb);

                // Calculate the probability ratio r_t(θ)
                float probRatio = Mathf.Exp(currentLogProb - oldLogProb);
                //Debug.Log("probRatio" + probRatio);

                //Compute the clipped objective
                float unclippedLoss = probRatio * advantages[i];
                float clippedLoss = Mathf.Clamp(probRatio, 1 - clipRange, 1 + clipRange) * advantages[i];
                //Debug.Log("clippedLoss" + clippedLoss);

                // Final policy loss (minimum of unclipped and clipped)
                float policyLoss = Mathf.Min(unclippedLoss, clippedLoss);
                //Debug.Log("policyLoss" + policyLoss);

                // Gradient for the mean (µ) & for the standard deviation (σ)
                float delta = action[j] - mean; // Difference between action and mean
                float gradientMean = -policyLoss * (delta / (stdDev * stdDev));
                float gradientSigma = -policyLoss * ((delta * delta) / (stdDev * stdDev * stdDev) - 1 / stdDev);

                //Debug.Log("gradientMean" + gradientMean);
                if (gradientMean.Equals(float.NaN) || gradientSigma.Equals(float.NaN))
                {
                    Debug.Log($"NaN Gradient for experience {i} with gradientMena({gradientMean}): mean({mean}) stdDev({stdDev}) action{action[j]}) oldMean({oldMean}) oldStdDev({oldStdDev}) oldAction{oldAction[j]})");
                    failedGradients++;
                    gradientMean = 0f;
                    gradientSigma = 0f;
                }
                if(float.IsInfinity(gradientMean) || float.IsInfinity(gradientSigma))
                {
                    failedGradients++;
                    Debug.Log($"Infinite Gradient for experience {i} with gradientMena({gradientMean}): mean({mean}) stdDev({stdDev}) action{action[j]}) oldMean({oldMean}) oldStdDev({oldStdDev}) oldAction{oldAction[j]}) advantages{advantages[i]}");
                    gradientMean = 0f;
                    gradientSigma = 0f;
                }

                // Store gradients for backpropagation
                policyLossGradients[j] = gradientMean; // Gradient for mean
                policyLossGradients[j + policyOutput.Length / 2] = gradientSigma; // Gradient for standard deviation

                // Debug for verification
                //Debug.Log($"Policy Loss (Mean) Gradient for experience {i}, action {j}: {gradientMean}");
                //Debug.Log($"Policy Loss (Sigma) Gradient for experience {i}, action {j}: {gradientSigma}");
            }

            // Backpropagate the gradients through the policy network
            float[] instantGradients = policyNetwork.BackPropogate(policyLossGradients, experiences[i].State);

            if(ArrayIsNaN(instantGradients))
            {
                failedGradientArrays++;
                Debug.Log($"NaN in Instant Gradients Experience {i} with policyLossGradients " + StringArray(policyLossGradients));
            }
            else
            {
                for (int j = 0; j < totalGradients.Length; j++)
                {
                    totalGradients[j] += instantGradients[j];
                }
            }
        }


        //print the % of failed gradient caculations
        float totalGradientCount = (float)experiences.Count * (float)experiences[0].Action.Length;
        float failedGradientPercent = (failedGradients / totalGradientCount) * 100f;
        Debug.Log("Total gradients failed: " + failedGradientPercent + "%");

        //print the % of mysteriously failed gradient arrays
        float totalGradientArraysCount = (float)experiences.Count;
        float failedGradientArraysPercent = (failedGradientArrays / totalGradientArraysCount) * 100f;
        Debug.Log("Total gradients arrays failed: " + failedGradientArraysPercent + "%");

        Debug.Log("Culmitive Gradients: " + StringArray(totalGradients));

        //average the gradients
        for (int i = 0; i < totalGradients.Length; i++)
        {
            totalGradients[i] /= experiences.Count;
        }

        //update the value network parameters
        float[] policyNetworkParameters = policyNetwork.ReadNeuralNetwork();
        for (int i = 0; i < totalGradients.Length; i++)
        {
            //clip the gradient
            totalGradients[i] = Mathf.Clamp(totalGradients[i], -clipRange, clipRange); // Symmetrical clipping
            policyNetworkParameters[i] -= policyLearningRate * totalGradients[i];
        }

        policyNetwork.WriteNeuralNetwork(policyNetworkParameters);
    }

    private float CalculateLogProb(float action, float mean, float stdDev)
    {
        // Calculate the log probability using the Gaussian distribution formula
        float logProb = -0.5f * Mathf.Pow((action - mean) / stdDev, 2) - Mathf.Log(stdDev * Mathf.Sqrt(2f * Mathf.PI));
        return logProb;
    }

    private void UpdateValueNeuralNetwork(List<Experience> experiences)
    {
        // Compute value loss and update value network
        float totalLoss = 0f; //used to record progress in the minimization of "loss"
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
                discount *= DiscountFactor;
            }

            //calculate the instanteneous loss function
            float gain = predictedValue - actualValue;

            //stow the loss function for reporting of network accuracy
            float lossFunction = Mathf.Pow(gain, 2f); //not used beyond reporting
            totalLoss += lossFunction;

            //compute the instantenuos gradient change to each weight and bias using backpropogation and add to total gradients
            float[] instantGradients = valueNetwork.BackPropogate(new float[] { gain }, experiences[i].State);
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
            //clip gradients
            totalGradients[i] = Mathf.Clamp(totalGradients[i], -clipRange, clipRange); // Symmetrical clipping
            valueNetworkParameters[i] -= valueLearningRate * totalGradients[i];
        }

        valueNetwork.WriteNeuralNetwork(valueNetworkParameters);

        //report network prediction effectiveness
        Debug.Log("Value Network Loss:" + totalLoss);
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

    private static float RandomGauss(float mean, float standardDeviation)
    {
        // Generate two uniform random numbers between 0 and 1
        float u1 = UnityEngine.Random.value;
        float u2 = UnityEngine.Random.value;

        // Apply the Box-Muller transform
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);

        // Scale to the desired mean and standard deviation
        return Mathf.Abs(mean + standardDeviation * randStdNormal);
    }

    private string StringArray(float[] arr)
    {
        string str = "";
        for(int i = 0; i < arr.Length; i++)
        {
            str += arr[i] + ", ";
        }
        return str;
    }

    private bool ArrayIsNaN(float[] arr)
    {
        for(int i = 0; i < arr.Length; i++)
        {
            if(arr[i].Equals(float.NaN))
            {
                return true;
            }
        }
        return false;
    }
}

