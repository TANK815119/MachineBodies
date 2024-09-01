using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//this class first takes in the relative position of all the limbs
//these values are fed into the neural netwoek
//then the output is taken and physically applied, like msucles, to the torque of the limbs
public class SimceptionAction : MonoBehaviour
{
    [SerializeField] private Vector2Int hiddenLayerWidthByHeight = new Vector2Int(4, 32);
    [SerializeField] private string limbTag = "BodyGib";

    private NeuralNetwork neuralNet;
    private Transform trans;
    private Vector3[] inputRotations; //inputs rotation of limb relative to its parent
    private Vector3[] inputAngularVelocity; //inputs angular velocity of limb
    private Vector3[] outputTorques; //outputs torque

    //bodily references
    private GameObject[] limbArr;
    private Transform[] transArr;
    private Rigidbody[] rigidArr;

    public void StartUp()
    {
        //set up reference to transform
        //number of children(may have to rework to get children of children so this may grow)
        trans = this.gameObject.transform;
        int childNumber = countAllLimbChildren(trans, limbTag);

        //set of vectors to store inputs and the list of actual limb references
        inputRotations = new Vector3[childNumber];
        inputAngularVelocity = new Vector3[childNumber];
        outputTorques = new Vector3[childNumber];

        transArr = new Transform[childNumber];
        limbArr = new GameObject[childNumber];
        rigidArr = new Rigidbody[childNumber];
        fillArraysWithChildren(trans);
    }

    public void FixedUpdate()
    {
        //I first fill the two input vector arrays with their corresponding values from the scene
        //I then take the resulting filled vector arrays and unpackagethem into a megainput float array
        //I then feed these "neuronised" float values and feed them into the input of the neural network
        //this may be inneficient as I am packaging and then unpackaging, but it makes intuitve sense to me
        //I take the raw float output and turn them into vectors
        //i use these vectors to apply torque to every single limb
        
        updateVectors();
        float[] megaInput = neuroniseInputs();
        float[] rawOutput = neuralNet.ForwardPass(megaInput);
        outputTorques = outputReader(rawOutput);
        for(int i = 0; i < limbArr.Length; i++)
        {
            rigidArr[i].AddTorque(outputTorques[i], ForceMode.Impulse);
        }
    }

    //turn float array into compiled vec array
    private static Vector3[] outputReader(float[] rawOutput)
    {
        Vector3[] outputTorqueArr = new Vector3[rawOutput.Length / 3];
        for(int i = 0; i < rawOutput.Length; i+=3) //a step of 3 covers a vector3
        {
            Vector3 currVec = new Vector3(rawOutput[i], rawOutput[i + 1], rawOutput[i + 2]); //compile raw values into vec
            outputTorqueArr[i / 3] = currVec; //add to vec array that is a third smaller
        }
        return outputTorqueArr;
    }

    //simply update all the vector arrays of rotation and velocity values for neuroniseation
    private void updateVectors()
    {
        for(int i = 0; i < limbArr.Length; i++)
        {
            inputRotations[i] = transArr[i].eulerAngles;
            inputAngularVelocity[i] = rigidArr[i].angularVelocity;
        }
    }

    //puts all the values of those pesky vector3 arrays together
    private float[] neuroniseInputs()
    {
        //package up the vecotr arrays to make this code more expandable
        Vector3[][] megaVecArray = {inputAngularVelocity, inputRotations}; //i should probably move this out and make it a parameter
        float[] megaInput = new float[inputAngularVelocity.Length * 3 + inputRotations.Length * 3];//length of output
        int megaArrIndex = 0;//index for assigning values to megaInput Array

        //open up megaVecArray to acces to two vectors arrays
        for(int megaVecIndex = 0; megaVecIndex < megaVecArray.Length; megaVecIndex++)
        {
            Vector3[] currVectorArr = megaVecArray[megaVecIndex]; //assigning to variable to make more readable
            //open up this vector array
            for (int vecIndex = 0; vecIndex < currVectorArr.Length; vecIndex++)
            {
                //crack open and append the 3 values of the vector3 to the megainput array
                Vector3 currVector = currVectorArr[vecIndex];
                megaInput[megaArrIndex] = currVector.x; megaArrIndex++;
                megaInput[megaArrIndex] = currVector.y; megaArrIndex++;
                megaInput[megaArrIndex] = currVector.z; megaArrIndex++;
            }
        }
        return megaInput;
    }

    //count the TOTAL of ALL children with specific tag
    private static int countAllLimbChildren(Transform parent, string limbTag)
    {
        int childCount = 0;
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if(child.CompareTag(limbTag))
            {
                childCount++;
            }
            childCount += countAllLimbChildren(child, limbTag);
        }
        return childCount;
    }

    //fills the transform, rigidbody, object reference arrays with ALL children(childofchild)
    private int childIndex = 0;
    private void fillArraysWithChildren(Transform parent)
    {
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if(child.CompareTag(limbTag))
            {
                transArr[childIndex] = child;
                limbArr[childIndex] = child.gameObject;
                rigidArr[childIndex] = child.GetComponent<Rigidbody>();
                childIndex++;
            }
            fillArraysWithChildren(child);
        }
    }

    public void SetNeuralNetwork(int[] networkDimensions, Layer[] layerArr, bool mutate)
    {
        neuralNet = new NeuralNetwork(networkDimensions, layerArr);
        if (mutate)
        {
            neuralNet.Mutate(3f, 0.1f);
        }
    }

    public void CreateNeuralNetwork()
    {
        Debug.Log("Created anew! lol");
        //setting up shape of neural network
        //the arrays are getting multiplied by 3 because they're vectors
        neuralNet = new NeuralNetwork(inputAngularVelocity.Length * 3 + inputRotations.Length * 3,
                                      outputTorques.Length * 3,
                                      hiddenLayerWidthByHeight.x,
                                      hiddenLayerWidthByHeight.y);
        neuralNet.Mutate(3f, 0.1f);
    }

    public NeuralNetwork GetNeuralNetwork()
    {
        return neuralNet;
    }
}
