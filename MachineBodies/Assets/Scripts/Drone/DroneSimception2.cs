using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//drone simception updated to work with the GoalCreator
public class DroneSimception2 : Simception
{
    private const int inputVolume = 3; // WAS 3
    private const int outputVolume = 1;
    private const int breadth = 3; //was 3
    private const int height = 5; // was 5

    private NeuralNetwork neuralNetwork;

    private float[] inputs;
    private float[] outputs;
    private Rigidbody rb;

    private DroneGoalCreator goalCreator;
    private Vector3 goalPosition = Vector3.zero;

    private float[] lastInputs;
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

        outputs = neuralNetwork.ForwardPass(inputs);
        float thrust = outputs[0] * 50f;

        if(Random.Range(0, 99) == 0)
        {
            Debug.Log(thrust);
        }

        rb.AddForce(Vector3.up * thrust, ForceMode.Force);

        lastInputs = inputs;
        lastPosition = this.transform.position;
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
