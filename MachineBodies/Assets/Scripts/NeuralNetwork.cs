using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class NeuralNetwork
{
    private int[] networkDimensions = {5, 32, 2}; //nodes per layer // first number is representative of simulation inputs, basically(make sure they match)
    private Layer[] layerArr;

    public NeuralNetwork() : this(5, 2, 1, 32) { }// default: fills networkDimensions with {5, 32, 2}
    public NeuralNetwork(int inputs, int outputs, int hiddenBreadth, int hiddenHeight)//migh want to add some parameters for neural newtwork size
    {
        networkDimensions = new int[hiddenBreadth + 2];
        for (int i = 0; i < hiddenBreadth + 2; i++) //breadth plus 2 for input and output
        {
            if(i == 0)
            {
                networkDimensions[i] = inputs;
            }
            else if(i == networkDimensions.Length - 1)
            {
                networkDimensions[i] = outputs;
            }
            else
            {
                networkDimensions[i] = hiddenHeight;
            }
        }
        //Debug.Log("Input: " + inputs + ", Output: " + outputs);
        Awake();
    }
    public NeuralNetwork(int[] networkDimensions, Layer[] layerArr)
    {
        this.networkDimensions = networkDimensions;
        this.layerArr = layerArr;
    }

    public void Awake()
    {
        layerArr = new Layer[networkDimensions.Length - 1];

        for (int i = 0; i < layerArr.Length; i++)
        {
            layerArr[i] = new Layer(networkDimensions[i], networkDimensions[i + 1]);
        }
    }

    public float[] ForwardPass(float[] inputs)
    {
        //hidden layers
        for (int i = 0; i < layerArr.Length; i ++)
        {
            if(i == 0) //first layer
            {
                layerArr[i].Forward(inputs);
                layerArr[i].Activation();
            }
            else if(i == layerArr.Length - 1) //final layer
            {
                layerArr[i].Forward(layerArr[i - 1].nodeArray);
                //no activation
            }
            else //hidden layers(default)
            {
                layerArr[i].Forward(layerArr[i - 1].nodeArray);
                layerArr[i].Activation();
            }
        }

        return layerArr[layerArr.Length - 1].nodeArray; //return final layer
    }

    public Layer[] CopyLayers() //deep copy of object into its base layers
    {
        Layer[] copiedLayerArr = new Layer[networkDimensions.Length - 1];
        for(int i = 0; i < layerArr.Length; i++)
        {
            copiedLayerArr[i] = new Layer(networkDimensions[i], networkDimensions[i + 1]);
            System.Array.Copy(layerArr[i].weightsArray, copiedLayerArr[i].weightsArray, layerArr[i].weightsArray.Length);
            System.Array.Copy(layerArr[i].biasesArray, copiedLayerArr[i].biasesArray, layerArr[i].biasesArray.Length);
        }
        return copiedLayerArr;
    }

    public int[] CopyNetworkDimensions() //deep copy of network dimensions
    {
        int[] copiedNetworkDimensions = new int[networkDimensions.Length];
        System.Array.Copy(networkDimensions, copiedNetworkDimensions, networkDimensions.GetLength(0));
        return copiedNetworkDimensions; 
    }

    public NeuralNetwork DeepCopyNeuralNetwork()
    {
        NeuralNetwork newNeuralNetwork = new NeuralNetwork(CopyNetworkDimensions(), CopyLayers());
        return newNeuralNetwork;
    }

    public void Mutate(float neuronChance, float standardDeviation) //3f and 0.1f are good default values
    {
        for (int layerIndex = 0; layerIndex < layerArr.Length; layerIndex++) //loop through layers
        {
            Layer thisLayer = layerArr[layerIndex];
            for (int neuroIndex = 0; neuroIndex < thisLayer.weightsArray.GetLength(0); neuroIndex++) //loop through nodes
            {
                for (int inputIndex = 0; inputIndex < thisLayer.weightsArray.GetLength(1); inputIndex++) //loop through input weights
                {
                    if (UnityEngine.Random.Range(0, 100) < neuronChance)
                    {
                        thisLayer.weightsArray[neuroIndex, inputIndex] += RandomGauss(0f, standardDeviation);
                    }
                }

                if (UnityEngine.Random.Range(0, 100) < neuronChance)
                {
                    thisLayer.biasesArray[neuroIndex] += RandomGauss(0f, standardDeviation);
                }
            }
        }
    }

    private static float RandomGauss(float mean, float standardDeviation)
    {
        // Generate two uniform random numbers between 0 and 1
        float u1 = UnityEngine.Random.value;
        float u2 = UnityEngine.Random.value;

        // Apply the Box-Muller transform
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);

        // Scale to the desired mean and standard deviation
        return mean + standardDeviation * randStdNormal;
    }

    public float[] BackPropogate(float[] policyLoss, float[] stowedInputs) //returns an array of gradients calculated through backpropogation to match the error(assumes ReLu)
    {
        float[] gradients = new float[this.ReadNeuralNetwork().Length]; // Holds all the gradients (for weights and biases)

        // Initialize delta as the gradient of the loss with respect to the output (the last layer)
        float[] delta = policyLoss; // For the output layer, delta is just the loss gradient

        // Loop backwards through the layers
        for (int i = layerArr.Length - 1; i >= 0; i--)
        {
            Layer currentLayer = layerArr[i];

            //THESE TWO OPERATIONS ARE VERY ESPENSIVE, WILL LIKELY STORE THE OUTPUTS OF LAYERS IN THE FUTURE INSTEAD
            float[] layerOutput = RecalculateLayerOutput(stowedInputs, i); // Recalculate layer outputs
            float[] layerInput = RecalculateLayerOutput(stowedInputs, i - 1); //recacluate layer inputs

            // Compute ReLU derivative for this layer's outputs
            float[] dReLU = new float[layerOutput.Length];
            for (int j = 0; j < layerOutput.Length; j++)
            {
                // ReLU derivative: 1 if output > 0, 0 otherwise
                dReLU[j] = layerOutput[j] > 0 ? 1f : 0f;
            }

            // Multiply delta by the ReLU derivative (element-wise)
            for (int j = 0; j < delta.Length; j++)
            {
                delta[j] *= dReLU[j];
            }

            // Backpropagate gradients to weights and biases for this layer
            for (int j = 0; j < currentLayer.weightsArray.GetLength(0); j++) // Loop over each neuron
            {
                for (int k = 0; k < currentLayer.weightsArray.GetLength(1); k++) // Loop over each input
                {
                    // Gradient for weight[j, k] is delta[j] * input to that neuron
                    // Gradients are stored by layerIndex * (numNodes*numInputs + numBiases) + numNode*numInputs + numInput to store weights in 1d
                    int weightIndex = GetWeightIndex(i, j, k);
                    gradients[weightIndex] = delta[j] * layerInput[k];
                }

                // Gradient for bias[j] is just delta[j]
                // Gradients are stored by layerIndex * (numNodes*numInputs + numBiases) + numNodes*numInputs + numBias to store biases in 1d
                int biasIndex = GetBiasIndex(i, j);
                gradients[biasIndex] = delta[j];
            }

            // If it's not the input layer, calculate the delta for the previous layer
            if (i > 0)
            {
                float[] newDelta = new float[layerArr[i - 1].nodeArray.Length]; // Delta for the previous layer
                for (int j = 0; j < newDelta.Length; j++) // Loop over previous layer's neurons
                {
                    newDelta[j] = 0; // Initialize new delta for each neuron in the previous layer
                    for (int k = 0; k < delta.Length; k++) // Loop over current layer's neurons
                    {
                        newDelta[j] += delta[k] * currentLayer.weightsArray[k, j]; // Backpropagate delta to previous layer
                    }
                }
                delta = newDelta; // Update delta for the next layer back in the chain
            }
        }

        return gradients; // Return all the computed gradients
    }

    private float[] RecalculateLayerOutput(float[] inputs, int layerIndex)
    {
        // Loop through the layers up to the given index
        for (int i = 0; i <= layerIndex; i++) // No need for +1 here as previously thought
        {
            if (i == 0) // First layer
            {
                // Forward pass for the first layer using the original input
                layerArr[i].Forward(inputs);
                layerArr[i].Activation(); // Apply ReLU or other activation functions
            }
            else if (i == layerArr.Length - 1) // Final layer
            {
                // Forward pass for the last layer using the output from the previous layer
                layerArr[i].Forward(layerArr[i - 1].nodeArray);
                // No activation if it's the output layer
            }
            else // Hidden layers
            {
                // Forward pass for hidden layers using the output from the previous layer
                layerArr[i].Forward(layerArr[i - 1].nodeArray);
                layerArr[i].Activation(); // Apply activation functions
            }
        }

        // Return the node array (output) of the current layer at layerIndex
        return layerArr[layerIndex].nodeArray;
    }

    private int GetWeightIndex(int layerIndex, int neuronIndex, int inputIndex)
    {
        // Gradients were stored by layerIndex * (numNodes*numInputs + numBiases) + numNode*numInputs + numInput to store weights in 1d
        // This code should account for the varying size of the input and output layers
        int offset = 0;

        // Accumulate offsets for previous layers' weights and biases
        for (int i = 0; i < layerIndex; i++)
        {
            offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Weights
            offset += layerArr[i].biasesArray.Length; // Biases
        }

        // Add current layer weight index
        return offset + (neuronIndex * layerArr[layerIndex].weightsArray.GetLength(1)) + inputIndex;
    }

    private int GetBiasIndex(int layerIndex, int neuronIndex)
    {
        // Gradients were stored by layerIndex * (numNodes*numInputs + numBiases) + numNodes*numInputs + numBias to store biases in 1d
        // This code should account for the varying size of the input and output layers
        int offset = 0;

        // Accumulate offsets for previous layers' weights and biases
        for (int i = 0; i < layerIndex; i++)
        {
            offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Weights
            offset += layerArr[i].biasesArray.Length; // Biases
        }

        // Add current layer bias index
        return offset + (layerArr[layerIndex].weightsArray.GetLength(0) * layerArr[layerIndex].weightsArray.GetLength(1)) + neuronIndex;
    }

    private int CountParameters()
    {
        // Count up the number of total weights and biases, hope this code isnt wrong
        int count = 0;
        for (int i = 0; i < layerArr.Length; i++)
        {
            count += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1) + layerArr[i].biasesArray.Length;
        }

        return count;
    }

    public float[] ReadNeuralNetwork() //turn the entire neural network into an 1d array of floats to represent parameters
    {
        // Put all parameters into a 1d list to be turned into array later
        float[] parameters = new float[CountParameters()];

        //loop through layers
        for(int i = 0; i < layerArr.Length; i++)
        {
            //loop through neurons
            for(int j = 0; j < layerArr[i].weightsArray.GetLength(0); j++)
            {
                //loop throguh associated weights
                for(int k = 0; k < layerArr[i].weightsArray.GetLength(1); k++)
                {
                    //add weight
                    parameters[GetWeightIndex(i, j, k)] = layerArr[i].weightsArray[k, j];
                }

                //add bias
                parameters[GetBiasIndex(i, j)] = layerArr[i].biasesArray[j];
            }
        }

        return parameters;
    }
    
    public void WriteNeuralNetwork(float[] parameters) //turn an entire 1d array of floats into the parameters of the neural network
    {
        //loop through layers
        for (int i = 0; i < layerArr.Length; i++)
        {
            //loop through neurons
            for (int j = 0; j < layerArr[i].weightsArray.GetLength(0); j++)
            {
                //loop throguh associated weights
                for (int k = 0; k < layerArr[i].weightsArray.GetLength(1); k++)
                {
                    //assign weight
                    layerArr[i].weightsArray[k, j] = parameters[GetWeightIndex(i, j, k)];
                }

                //assign bias
                layerArr[i].biasesArray[j] = parameters[GetBiasIndex(i, j)];
            }
        }
    }

    //nether save or load are implemented
    //will have to test very soon
    //I should probably test with a simple goal
    public void LoadNetworkState() //this should probably take a parameter like generation
    {
        string dataDirPath = Application.persistentDataPath;
        string dataFileName = "network_data.json";
        string fullPath = Path.Combine(dataDirPath, dataFileName);
        NeuralNetwork loadedNetwork = null;
        if(File.Exists(fullPath))
        {
            try
            {
                //load the serialized data from file
                string dataToLoad = "";
                using (FileStream stream = new FileStream(fullPath, FileMode.Open))
                {
                    using (StreamReader reader = new StreamReader(stream))
                    {
                        dataToLoad = reader.ReadToEnd();
                    }
                }

                //desericalize from Json to C# object
                loadedNetwork = JsonUtility.FromJson<NeuralNetwork>(dataToLoad);
            }
            catch (Exception e)
            {
                Debug.LogError("Error occured when trying to load data from file: + " + fullPath + "\n" + e);
            }
        }
    }
    public void SaveNetworkState()
    {
        string dataDirPath = Application.persistentDataPath;
        string dataFileName = "network_data.json";
        string fullPath = Path.Combine(dataDirPath, dataFileName);
        try 
        {
            // create directory path
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath));

            // serialize the C# game data object into Json
            String dataToStore = JsonUtility.ToJson(this, true);

            //write to actual file
            using (FileStream stream = new FileStream(fullPath, FileMode.Create))
            {
                using (StreamWriter writer = new StreamWriter(stream))
                {
                    writer.Write(dataToStore);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error occured when trying to save data to file: + " + fullPath + "\n" + e);
        }

    }
}