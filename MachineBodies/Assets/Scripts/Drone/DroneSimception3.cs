using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//drone simception updated to work with the GoalCreator
//works as a drone ordinarily would with 4 propeller points where thrust is applied
public class DroneSimception3 : Simception
{
    private int inputVolume = 3 + 1 + 3;
    private int outputVolume = 4; //4 propellers
    private int breadth = 5;
    private int height = 9;

    private NeuralNetwork neuralNetwork;

    private float[] inputs;
    private float[] outputs;
    private Rigidbody rb;

    private DroneGoalCreator3 goalCreator;
    private Vector3 goalPosition = Vector3.zero;

    [SerializeField] float maxThrust = 15f;
    [SerializeField] Transform[] props = new Transform[4];

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
        goalPosition = new Vector3(transform.position.x, goalPosition.y, goalPosition.z);
    }

    private void FixedUpdate()
    {
        inputs[0] = goalPosition.x - transform.position.x;
        inputs[1] = goalPosition.y - transform.position.y;
        inputs[2] = goalPosition.z - transform.position.z;
        inputs[3] = rb.velocity.magnitude;
        inputs[4] = transform.eulerAngles.x;
        inputs[5] = transform.eulerAngles.y;
        inputs[6] = transform.eulerAngles.z;

        outputs = neuralNetwork.ForwardPass(inputs);

        ProcessThrustOutput();
    }

    private void ProcessThrustOutput()
    {
        for(int i = 0; i < outputs.Length; i++)
        {
            float thrust = outputs[i];

            if(thrust > maxThrust)
            {
                thrust = maxThrust;
            }
            if(thrust < 0f)
            {
                thrust = 0f;
            }

            rb.AddForceAtPosition(transform.up * thrust, props[i].position, ForceMode.Force);

            lastInputs = inputs;
            lastPosition = this.transform.position;
        }
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
        //MutateNeuralNetwork(100f, 0.5f);
    }

    public override void MutateNeuralNetwork(float neuronChance, float standardDeviation)
    {
        neuralNetwork.Mutate(neuronChance, standardDeviation);
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        this.goalCreator = (DroneGoalCreator3)goalCreator;
    }
    public override float[] GetLastInputs()
    {
        return lastInputs;
    }

    public override float CalculateLastReward()
    {
        float prevDist = Vector3.Distance(goalPosition, lastPosition);
        float currDist = Vector3.Distance(goalPosition, this.transform.position);
        return prevDist - currDist; //will be positive if currDist is smaller than prevDist
    }

    public override int[] GetNetworkExtents()
    {
        return new int[] { inputVolume, outputVolume, breadth, height };
    }
}
