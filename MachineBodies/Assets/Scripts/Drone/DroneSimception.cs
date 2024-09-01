using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneSimception : Simception
{
    private int inputVolume = 2;
    private int outputVolume = 1;
    private int breadth = 2;
    private int height = 4;

    private NeuralNetwork neuralNetwork;

    private float[] inputs;
    private float[] outputs;
    private Rigidbody rb;

    public void Start()
    {
        inputs = new float[inputVolume];
        rb = GetComponent<Rigidbody>();
    }

    public override void Startup()
    {

    }

    private void FixedUpdate()
    {
        inputs[0] = transform.localPosition.y;
        inputs[1] = rb.velocity.magnitude;

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
    }

    public override void MutateNeuralNetwork(float neuronChance, float standardDeviation)
    {
        neuralNetwork.Mutate(neuronChance, standardDeviation);
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        //throw new System.NotImplementedException();
    }
}
