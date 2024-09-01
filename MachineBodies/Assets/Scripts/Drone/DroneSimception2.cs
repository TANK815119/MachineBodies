using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//drone simception updated to work with the GoalCreator
public class DroneSimception2 : Simception
{
    private int inputVolume = 3;
    private int outputVolume = 1;
    private int breadth = 3;
    private int height = 5;

    private NeuralNetwork neuralNetwork;

    private float[] inputs;
    private float[] outputs;
    private Rigidbody rb;

    private DroneGoalCreator goalCreator;
    private Vector3 goalPosition = Vector3.zero;

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
        float thrust = outputs[0];

        rb.AddForce(Vector3.up * thrust, ForceMode.Force);
    }

    public override NeuralNetwork GetNeuralNetwork()
    {
        return neuralNetwork;
    }

    public override void SetNeuralNetwork(NeuralNetwork neuralNetwork)
    {
        this.neuralNetwork = neuralNetwork;
    }

    public override void GenerateRandomNeuralNetwork()
    {
        neuralNetwork = new NeuralNetwork(inputVolume, outputVolume, breadth, height);
        MutateNeuralNetwork(100f, 0.5f);
    }

    public override void MutateNeuralNetwork(float neuronChance, float standardDeviation)
    {
        neuralNetwork.Mutate(neuronChance, standardDeviation);
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        this.goalCreator = (DroneGoalCreator)goalCreator;
    }
}
