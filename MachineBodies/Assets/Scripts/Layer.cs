using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public float[,] weightsArray;
    public float[] biasesArray;
    public float[] nodeArray;

    private int numInputs;
    private int numNodes;

    public Layer(int numInputs, int numNodes)
    {
        this.numInputs = numInputs;
        this.numNodes = numNodes;

        weightsArray = new float[numNodes, numInputs];
        biasesArray = new float[numNodes];
        nodeArray = new float[numNodes];
    }

    public void Forward(float[] inputsArray) //sum of the weights * inputs + bias // might fuck m eup on first one
    {
        nodeArray = new float[numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            //sum of the weights * inputs
            for (int j = 0; j < numInputs; j++)
            {
                nodeArray[i] += weightsArray[i, j] * inputsArray[j]; //j dimension is same length for either array
            }

            //add the bias
            nodeArray[i] += biasesArray[i];
        }
    }

    //this the mathematical activation function that makes it a neural netwrok and nota linear regression model
    public void Activation() //relu method
    {
        for (int i = 0; i < numNodes; i++)
        {
            if (nodeArray[i] < 0)
            {
                nodeArray[i] = 0;
            }
        }
    }
}