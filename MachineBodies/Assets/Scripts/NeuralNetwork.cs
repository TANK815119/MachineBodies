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