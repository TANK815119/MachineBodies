using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class Simception : MonoBehaviour
{
    public abstract void Startup();

    public abstract void SetNeuralNetwork(NeuralNetwork neuralNetwork);

    public abstract void SetOldNeuralNetwork(NeuralNetwork neuralNetwork);

    public abstract NeuralNetwork GetNeuralNetwork();

    public abstract void GenerateRandomNeuralNetwork();

    public abstract void MutateNeuralNetwork(float neuronChance, float standardDeviation);

    public abstract void SetGoalCreator(GoalCreator goalCreator);

    public abstract float[] GetLastInputs();

    public abstract Experience GetLastExperience();

    public abstract float CalculateLastReward();

    public abstract int[] GetNetworkExtents();
}
