using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//drone simception updated to work with the GoalCreator
public class DroneSimception2 : Simception
{
    private const int inputVolume = 3; // WAS 3
    private const int outputVolume = 2;
    private const int breadth = 3; //was 3
    private const int height = 5; // was 5

    private NeuralNetwork neuralNetwork;
    private NeuralNetwork oldNeuralNetwork;

    private float[] inputs;
    private Rigidbody rb;

    private DroneGoalCreator goalCreator;
    private Vector3 goalPosition = Vector3.zero;

    private float[] lastInputs;
    private float[] lastOutputs;
    private float[] lastActions;
    private float[] lastOldOutputs;
    private float[] lastOldActions;
    private Vector3 lastPosition;

    public void Start()
    {
        inputs = new float[inputVolume];
        rb = GetComponent<Rigidbody>();
    }

    public override void Startup()
    {
        goalPosition = goalCreator.GetGoal();
    }

    private void FixedUpdate()
    {
        inputs[0] = transform.localPosition.y;
        inputs[1] = rb.velocity.magnitude;
        inputs[2] = goalPosition.y;

        //calculate old outputs and actiosn
        float[] oldOutputs = oldNeuralNetwork.ForwardPass(inputs);
        float oldThrustMean = oldOutputs[0] * 50f;
        float oldThrustSD = oldOutputs[1];
        float oldThrust = RandomGauss(oldThrustMean, oldThrustSD);

        //calculate novel outputs and actiosn
        float[] outputs = neuralNetwork.ForwardPass(inputs);
        float thrustMean = outputs[0] * 50f;
        float thrustSD = outputs[1];
        float thrust = RandomGauss(thrustMean, thrustSD);

        if(Random.Range(0, 99) == 0)
        {
            Debug.Log(thrust);
        }

        rb.AddForce(Vector3.up * thrust, ForceMode.Force);

        lastInputs = inputs;
        lastOutputs = outputs;
        lastActions = new float[] { thrust };
        lastOldOutputs = oldOutputs;
        lastOldActions = new float[] { oldThrust };
        lastPosition = this.transform.position;
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

    public override NeuralNetwork GetNeuralNetwork()
    {
        return neuralNetwork;
    }

    public override void SetNeuralNetwork(NeuralNetwork neuralNetwork)
    {
        this.neuralNetwork = neuralNetwork;
        PrintNeuralNetwork();
    }

    public override void SetOldNeuralNetwork(NeuralNetwork neuralNetwork)
    {
        this.oldNeuralNetwork = neuralNetwork;
    }

    public override void GenerateRandomNeuralNetwork()
    {
        neuralNetwork = new NeuralNetwork(inputVolume, outputVolume, breadth, height, true);
        neuralNetwork.InitializeWeightsHe();
        //MutateNeuralNetwork(100f, 0.5f);
    }

    public override void MutateNeuralNetwork(float neuronChance, float standardDeviation)
    {
        neuralNetwork.Mutate(neuronChance, standardDeviation);
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        this.goalCreator = (DroneGoalCreator)goalCreator;
    }

    public override float[] GetLastInputs()
    {
        return lastInputs;
    }

    public override Experience GetLastExperience()
    {
        return new Experience(GetLastInputs(), lastOutputs, lastActions, lastOldOutputs, lastOldActions, CalculateLastReward());
    }

    public override float CalculateLastReward()
    {
        float prevDist = Mathf.Abs(goalPosition.y - lastPosition.y);
        float currDist = Mathf.Abs(goalPosition.y - this.transform.position.y);
        return prevDist - currDist; //will be positive if currDist is smaller than prevDist
    }

    public override int[] GetNetworkExtents()
    {
        return new int[] { inputVolume, outputVolume, breadth, height };
    }

    private void PrintNeuralNetwork()
    {
        string output = "";
        float[] coefficients = neuralNetwork.ReadNeuralNetwork();
        for (int i = 0; i < coefficients.Length; i++)
        {
            output += coefficients[i] + ", ";
        }
        Debug.Log("Network Coefficients: " + output);
    }
}
