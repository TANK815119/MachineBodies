using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class GeneticAlgorithm : MonoBehaviour
{
    [SerializeField] private GameObject dummy;
    [SerializeField] private float coolDownTimer = 3f;
    [SerializeField] private int dummyVolume = 3;
    [SerializeField] private float spaceBetweenDummies = 4f;
    [SerializeField] private float spawnHeight = 2f;
    [SerializeField] private TextMeshProUGUI genText;
    [SerializeField] private GameObject floor;
    [SerializeField] private float timeSpeed = 1f;

    private float time = 0;
    private float cooldown = 0;
    private bool resetable = true;
    private float genStartPosititon;
    private int generation;

    //seperate arrays for each of these components because GetComponent is expensive(i hear)
    private GameObject[] dummyArr;
    private ScoreCounterAirtime[] scoreCounterArr;
    private SimceptionAction[] simActionArr;

    public void Start()
    {
        dummyArr = new GameObject[dummyVolume];
        scoreCounterArr = new ScoreCounterAirtime[dummyVolume];
        simActionArr = new SimceptionAction[dummyVolume];

        genStartPosititon = (dummyVolume / 2) * spaceBetweenDummies * -1;
        genText.text = "Generation: zero";
    }

    public void FixedUpdate()
    {
        Time.timeScale = timeSpeed;

        time += 1 * Time.fixedDeltaTime;

        if (resetable == true)
        {
            resetDummies();
        }

        //need this so it doesnt reset a billion times in one second
        if (resetable == false)
        {
            cooldown -= 1 * Time.fixedDeltaTime;
            if (cooldown <= 0)
            {
                resetable = true;
            }
        }
    }
    private void resetDummies()
    {
        bool firstGeneration = true;
        int bestNetSkim = 1; //hardcoded to skim the three best
        NeuralNetwork[] bestNetworkArr = new NeuralNetwork[bestNetSkim];
        //find the 3 most succesful neural network
        if (dummyArr[0] != null) //check if empty
        {
            firstGeneration = false;

            for (int netInd = 0; netInd < bestNetSkim; netInd++)
            {
                bestNetworkArr[netInd] = simActionArr[0].GetNeuralNetwork();
                float bestScore = scoreCounterArr[0].score;
                for (int i = 0; i < dummyVolume; i++)
                {
                    NeuralNetwork currNNet = simActionArr[i].GetNeuralNetwork();
                    if (scoreCounterArr[i].score >= bestScore && !currNNet.Equals(bestNetworkArr[0]) //checks its not previous nn
                        && !currNNet.Equals(bestNetworkArr[0]) && !currNNet.Equals(bestNetworkArr[0]))
                    {
                        bestScore = scoreCounterArr[i].score;
                        bestNetworkArr[netInd] = currNNet;
                    }
                }
                Debug.Log("Best Score: " + bestScore);
                if(bestScore >= coolDownTimer) //hardcoded to be withing 3 seconds of max
                {
                    Debug.Log("Time expanded!");
                    coolDownTimer++;
                }
            }
        }

        //this is the actual resetting part
        for (int i = 0; i < dummyVolume; i++)
        {
            //destroy previous dummy
            if (dummyArr[i] != null)
            {
                Destroy(dummyArr[i]);
            }

            //make dummy
            Vector3 position = new Vector3(genStartPosititon + i * spaceBetweenDummies, spawnHeight, 0);
            Vector3 rotation = new Vector3(0, 0, 0);
            GameObject thisDummy = Instantiate(dummy, position, Quaternion.Euler(rotation));
            dummyArr[i] = thisDummy;

            //give score counter
            ScoreCounterAirtime sc = thisDummy.GetComponent<ScoreCounterAirtime>();
            sc.floor = floor; //assigning this after might mess me up as the collisiondetectors may lack a floor assignment in Start()
            scoreCounterArr[i] = sc;

            if(!firstGeneration)
            {
                //set the neural network to a copy of the most succesful(mutated inside Start())(neural network inside simception)
                //I hope the fucking references here work, because if they dont, nothing will happen and it will be hard to tell
                simActionArr[i] = dummyArr[i].GetComponent<SimceptionAction>();
                simActionArr[i].StartUp();
                simActionArr[i].SetNeuralNetwork(bestNetworkArr[i % bestNetSkim].CopyNetworkDimensions(), bestNetworkArr[i % bestNetSkim].CopyLayers(), true);
            }
            else
            {
                //do nothing, the awake was called and made a default neural network, but store neural network references
                simActionArr[i] = dummyArr[i].GetComponent<SimceptionAction>();
                simActionArr[i].StartUp();
                simActionArr[i].CreateNeuralNetwork();
            }
        }

        //update generation text
        generation++;
        genText.text = "Generation: " + generation;

        //set up cooldown
        resetable = false;
        cooldown = coolDownTimer;
    }
}
