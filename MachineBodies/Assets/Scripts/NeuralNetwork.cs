using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class NeuralNetwork
{
    private int[] networkDimensions; //nodes per layer // first number is representative of simulation inputs, basically(make sure they match)
    private Layer[] layerArr;
    private bool isPolicy;

    //public NeuralNetwork() : this(5, 2, 1, 32) { }// default: fills networkDimensions with {5, 32, 2}

    public NeuralNetwork(int inputs, int outputs, int hiddenBreadth, int hiddenHeight, bool policy)
    {
        networkDimensions = new int[hiddenBreadth + 2];
        for (int i = 0; i < hiddenBreadth + 2; i++) //breadth plus 2 for input and output
        {
            if (i == 0)
            {
                networkDimensions[i] = inputs;
            }
            else if (i == networkDimensions.Length - 1)
            {
                networkDimensions[i] = outputs;
            }
            else
            {
                networkDimensions[i] = hiddenHeight;
            }
        }

        isPolicy = policy;

        //Debug.Log("Input: " + inputs + ", Output: " + outputs);
        Awake();
    }

    public NeuralNetwork(int inputs, int outputs, int hiddenBreadth, int hiddenHeight) : this(inputs, outputs, hiddenBreadth, hiddenHeight, false)
    {
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
                if(isPolicy)
                {
                    layerArr[i].ActivationPPO();
                }
                //layerArr[i].Activation(); //activation, actually. I cant handle negatives -- REMOVED FINAL ACTIVATION
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

    public void InitializeWeightsHe() //Great for innitializing weights of ReLu actication functions
    {
        for (int layerIndex = 0; layerIndex < layerArr.Length; layerIndex++) // loop through layers
        {
            Layer thisLayer = layerArr[layerIndex];
            int fanIn = thisLayer.weightsArray.GetLength(1); // Number of input connections

            // Calculate standard deviation for He initialization
            float standardDeviation = Mathf.Sqrt(2f / fanIn);

            for (int neuroIndex = 0; neuroIndex < thisLayer.weightsArray.GetLength(0); neuroIndex++) // loop through neurons
            {
                for (int inputIndex = 0; inputIndex < thisLayer.weightsArray.GetLength(1); inputIndex++) // loop through input weights
                {
                    thisLayer.weightsArray[neuroIndex, inputIndex] = RandomGauss(0f, standardDeviation);
                }

                // Initialize biases to zero or small values
                thisLayer.biasesArray[neuroIndex] = 0f;  // Often initialized to 0 or small value
            }
        }
        //double[] constCoefficients = { -1.328032, 0.3450068, 0.2178049, -0.07616708, -0.39105, -0.8946154, -0.5541302, -0.6842388, 0.5948431, -0.6567535, 0.1700459, 1.504941, 0.1147478, -1.0639, -0.9487465, 0.3276261, 0.2857761, 0.04824951, -1.773947, 0.2420805, -0.6662324, -0.4928881, 0.8133286, 0.4936297, 0.3945757, 0.6863497, -0.02181687, 0.1210485, -0.2117072, -0.9244639, -0.9834744, 0.02831673, -1.515262, 0.5972223, 0.1859123, -0.5204154, 0.2587827, 1.487662, -0.005826861, -0.8607997, -0.4539259, 0.4966782, -0.2782862, 0.8511897, 0.4708014, 0.9107162, 1.27138, 0.5028019, -0.114648, -0.120683, 0.9232644, 0.275372, 0.0490301, 1.056718, 0.1474769, -0.9564848, -0.3359877, -0.5811074, -0.4101139, -1.278794, -0.5759953, 0.7673454, 1.07981, 0.64053, -0.1156563, -0.8055712, 0.8532441, 0.1208906, -0.4816885, 0.1150411, -0.008410046, -0.02828152, 1.046244, -0.48818, -0.1345343, -0.1528954, 0.4185243, -1.528688, -0.5734174, 0.1243416, 0.9546558, -0.5805066, 0.9325264, 0.9521216, -1.476153, 0.3922449 };
        //float[] floatCoefficients = new float[constCoefficients.Length];
        //for (int i = 0; i < constCoefficients.Length; i++)
        //{
        //    floatCoefficients[i] = (float)constCoefficients[i];
        //}
        //this.WriteNeuralNetwork(floatCoefficients);
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

        for(int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = 999f;
        }

        // Initialize delta as the gradient of the loss with respect to the output (the last layer)
        float[] delta = policyLoss; // For the output layer, delta is just the loss gradient

        // Loop backwards through the layers
        for (int i = layerArr.Length - 1; i >= 0; i--) //removed = from >= to not go out of bounds
        {
            Layer currentLayer = layerArr[i];

            //THESE TWO OPERATIONS ARE VERY ESPENSIVE, WILL LIKELY STORE THE OUTPUTS OF LAYERS IN THE FUTURE INSTEAD
            float[] layerOutput = RecalculateLayerOutput(stowedInputs, i); // Recalculate layer outputs
            float[] layerInput;
            if (i > 0)
            {
                layerInput = RecalculateLayerOutput(stowedInputs, i - 1); //recacluate layer inputs
            }
            else
            {
                layerInput = stowedInputs;
            }

            // Compute derivative for this layer's outputs
            float[] derrivatives = new float[layerOutput.Length];
            for (int j = 0; j < layerOutput.Length; j++)
            {
                if (!isPolicy || i != layerArr.Length - 1)
                {
                    // leaky ReLU derivative: 1 if output > 0, 0.001f otherwise
                    derrivatives[j] = layerOutput[j] > 0 ? 1f : 0.001f;
                }
                else
                {
                    if(j < layerArr.Length / 2)
                    {
                        //Tanh function derrivative
                        float tanHValue = (float)Math.Tanh(layerOutput[j]); //activation(Tanh)
                        derrivatives[j] = 1f - tanHValue * tanHValue;  //derrivative(Tanh)
                    }
                    else
                    {
                        //Exponentuial function derrivative
                        derrivatives[j] = (float)Math.Exp(layerOutput[j]); // derrivative for exponential is same as activation
                    }
                }
            }
            
            

            // Multiply delta by the ReLU derivative (element-wise)
            for (int j = 0; j < delta.Length; j++)
            {
                delta[j] *= derrivatives[j];
            }

            // Backpropagate gradients to weights and biases for this layer
            for (int j = 0; j < currentLayer.weightsArray.GetLength(0); j++) // Loop over each neuron
            {
                for (int k = 0; k < currentLayer.weightsArray.GetLength(1); k++) // Loop over each input
                {
                    // Gradient for weight[j, k] is delta[j] * input to that neuron
                    // Gradients are stored by layerIndex * (numNodes*numInputs + numBiases) + numNode*numInputs + numInput to store weights in 1d
                    int weightIndex = GetWeightIndex(i, j, k);
                    //Debug.Log("Layer Index: " + i + " Node Index: " + j + " Weight Index: " + k + " Output: " + weightIndex);
                    //Debug.Log(delta.Length);
                    //float currDelta = delta[j];
                    //float currLayerOutput = layerInput[k];
                    //float currGradient = currDelta * currLayerOutput;
                    //gradients[weightIndex] = currGradient;
                    gradients[weightIndex] = delta[j] * layerInput[k];
                    //Debug.Log(delta[j] + " delta assigned to " + weightIndex);
                }

                // Gradient for bias[j] is just delta[j]
                // Gradients are stored by layerIndex * (numNodes*numInputs + numBiases) + numNodes*numInputs + numBias to store biases in 1d
                int biasIndex = GetBiasIndex(i, j);
                gradients[biasIndex] = delta[j];
                //Debug.Log("Layer Index: " + i + " bias Index: " + j + " Output: " + biasIndex);
                //Debug.Log(delta[j] + " delta assigned to " + biasIndex);
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

        //Debug.Log("Finished one backprop");
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

        //string str = "Recalculated Layer Output: ";
        //for (int i = 0; i < layerArr[layerIndex].nodeArray.Length; i++)
        //{
        //    str += layerArr[layerIndex].nodeArray[i] + ", ";
        //}
        //Debug.Log(str);

        // Return the node array (output) of the current layer at layerIndex
        return layerArr[layerIndex].nodeArray;
    }

    private int GetWeightIndex(int layerIndex, int nodeIndex, int weightIndex)
    {
        //// Gradients were stored by layerIndex * (numNodes*numInputs + numBiases) + numNode*numInputs + numInput to store weights in 1d
        ////this structure is essentially biases then weights but doesnt do it very well
        //// This code should account for the varying size of the input and output layers
        //int offset = 0;

        //// Accumulate offsets for previous layers' weights and biases
        //for (int i = 0; i < layerIndex; i++)
        //{
        //    //offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Weights
        //    //offset += layerArr[i].biasesArray.Length; // Biases
        //    offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Only weights
        //}

        //// Add current layer weight index
        //return offset + (neuronIndex * layerArr[layerIndex].weightsArray.GetLength(1)) + inputIndex;

        //new Gradient location math
        //will store all weights and then all biases for every layer
        //remember, there is a consistent number of biases, but not a consistent number of weights
        //there are no biases on the input, but there are first weights that connect them to the first hidden layer
        //the output layer is shorter and does have biases
        //so the math should start with weights and end with biases
        //there will be an inconsistent number of first weights
        //there will be an inconsistent number of last weights and last biases
        //all others should be consistent
        //each layer is composed of its inputs(weights) and then neurons(biases)
        //the weights of each layer are categorized by (neuronIndex) * (weightsPerNeuron) + inputIndex

        //int firstWeightsPerNode = layerArr[0].weightsArray.GetLength(1);
        //int standardWeightsPerNode = layerArr[1].weightsArray.Length;
        //int standardBiasCount = layerArr[0].biasesArray.Length; //interchangeable with layerArr[0].weightsArray.GetLength(0)

        //if (layerIndex == 0) //first input and hidden layer 1
        //{
        //    return nodeIndex * firstWeightsPerNode + weightIndex;
        //}
        //else //hidden and final layers
        //{
        //    //the first input weights + standard neuron first layer + 
        //    //then (layerIndex - 1) * weightsStandard * biasesStandard
        //    //+ neuronIndex * layerArr[layerIndex].weightsArray.GetLength(1) + inputIndex
        //    int firstLayer = firstWeightsPerNode * standardBiasCount + standardBiasCount;
        //    int prevLayers = (layerIndex - 1) * standardWeightsPerNode * standardBiasCount + (layerIndex - 1) * standardBiasCount;
        //    return firstLayer + prevLayers + nodeIndex * standardWeightsPerNode + weightIndex;
        //}

        int totalIndex = 0;

        //count up all previous layers
        for (int i = 0; i < layerIndex; i++)
        {
            //Debug.Log("nodes" + layerArr[i].weightsArray.GetLength(0) + " * weights" + layerArr[i].weightsArray.GetLength(1) + " + bias" + layerArr[i].biasesArray.Length);
            totalIndex += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1) + layerArr[i].biasesArray.Length;
        }

        //return the totalIndex + (nodeIndex * weightsPerNode) + (weightIndex + 1) so they dont overlap
        int weightsPerNode = layerArr[layerIndex].weightsArray.GetLength(1);
        //int index = totalIndex + nodeIndex * weightsPerNode + weightIndex;
        //Debug.Log("Layer Index: " + layerIndex + " Node Index: " + nodeIndex + " Weight Index: " + weightIndex + " Output: " + index);
        return totalIndex + nodeIndex * weightsPerNode + weightIndex;
    }

    private int GetBiasIndex(int layerIndex, int biasIndex)
    {
        //// Gradients were stored by layerIndex * (numNodes*numInputs + numBiases) + numNodes*numInputs + numBias to store biases in 1d
        //// This code should account for the varying size of the input and output layers
        //int offset = 0;

        //// Accumulate offsets for previous layers' weights and biases
        //for (int i = 0; i < layerIndex; i++)
        //{
        //    //offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Weights
        //    //offset += layerArr[i].biasesArray.Length; // Biases
        //    offset += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1); // Only weights
        //    offset += layerArr[i].biasesArray.Length; // Only biases
        //}

        //// Add current layer bias index
        //return offset + neuronIndex;
        ////return offset + (layerArr[layerIndex].weightsArray.GetLength(0) * layerArr[layerIndex].weightsArray.GetLength(1)) + neuronIndex;
        ///

        //int firstWeightsPerNode = layerArr[0].weightsArray.GetLength(1);
        //int standardWeightsPerNode = layerArr[1].weightsArray.Length;
        //int standardBiasCount = layerArr[0].biasesArray.Length; //interchangeable with layerArr[0].weightsArray.GetLength(0)
        //int finalBiasCount = layerArr[layerArr.Length - 1].biasesArray.Length;

        //if (layerIndex == 0) //first input and hidden layer 1
        //{
        //    return standardBiasCount * firstWeightsPerNode + (biasIndex + 1);
        //}
        //else if(layerIndex == layerArr.Length - 1) //final layer
        //{
        //    int firstLayer = firstWeightsPerNode * standardBiasCount + standardBiasCount;
        //    int prevLayers = (layerIndex - 1) * standardWeightsPerNode * standardBiasCount + (layerIndex - 1) * standardBiasCount;
        //    return firstLayer + prevLayers + finalBiasCount * standardWeightsPerNode + (biasIndex + 1);
        //}
        //else //hidden layers
        //{
        //    //the first input weights + standard neuron first layer + 
        //    //then (layerIndex - 1) * weightsStandard * biasesStandard
        //    //+ neuronIndex * layerArr[layerIndex].weightsArray.GetLength(1) + inputIndex
        //    int firstLayer = firstWeightsPerNode * standardBiasCount + standardBiasCount;
        //    int prevLayers = (layerIndex - 1) * standardWeightsPerNode * standardBiasCount + (layerIndex - 1) * standardBiasCount;
        //    return firstLayer + prevLayers + standardBiasCount * standardWeightsPerNode + (biasIndex + 1);
        //}

        int totalIndex = 0;

        //count up all previous layers
        for (int i = 0; i < layerIndex; i++)
        {
            totalIndex += layerArr[i].weightsArray.GetLength(0) * layerArr[i].weightsArray.GetLength(1) + layerArr[i].biasesArray.Length;
        }

        //count the weights of this layer
        totalIndex += layerArr[layerIndex].weightsArray.GetLength(0) * layerArr[layerIndex].weightsArray.GetLength(1);

        //int index = totalIndex + biasIndex;
        //Debug.Log("Layer Index: " + layerIndex + " bias Index: " + biasIndex + " Output: " + index);

        //return the totalIndex + the (biasIndex) + 1 so they dont overlap
        return totalIndex + biasIndex;
    }

    public int CountParameters()
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
                    float parameter = layerArr[i].weightsArray[j, k]; //reversed k and j and it worked??
                    parameters[GetWeightIndex(i, j, k)] = parameter;
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
                    layerArr[i].weightsArray[j, k] = parameters[GetWeightIndex(i, j, k)]; //reversed k and j and it worked??
                }

                //assign bias
                layerArr[i].biasesArray[j] = parameters[GetBiasIndex(i, j)];
            }
        }
    }

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