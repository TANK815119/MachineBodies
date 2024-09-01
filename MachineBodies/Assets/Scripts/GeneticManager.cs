using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class GeneticManager : MonoBehaviour
{
    //creature prefab
    [SerializeField] private GameObject creature;
    [SerializeField] private GoalCreator goalCreator;

    //simulation parameters
    [SerializeField] private int creatureVolume = 10;
    [SerializeField] private float creatureSpaceing = 4f;
    [SerializeField] private float simLengthMax = 3f;
    [SerializeField] private float simulationSpeed = 1f;
    
    //text
    [SerializeField] private TextMeshProUGUI genText;

    //mutation
    [SerializeField] private int reproducingCreatures = 3;
    [SerializeField] private float globalMutationChance = 5f; //%
    [SerializeField] private float neuronMutationChance = 3f; //%
    [SerializeField] private float mutationMagnitude = 0.1f; //Standard Deviation
    [SerializeField] private float reproductionSuccesBias = 3f;

    private GameObject[] creatures;

    private float simLength = 0;
    private int generation = 0;

    // Start is called before the first frame update
    void Start()
    {
        creatures = new GameObject[creatureVolume];

        genText.text = "Generation: zero";

        //generate first batch at random
        if(goalCreator != null)
        {
            goalCreator.GenerateGoal();
        }

        creatures = RepopulateCreatures(null);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        simLength += Time.fixedDeltaTime;

        if (simLength > simLengthMax)
        {
            AdvanceGeneration();

            simLength = 0f;

            //update generation text
            generation++;
            genText.text = "Generation: " + generation;
        }
    }

    private void AdvanceGeneration()
    {
        //collect the best few neural networks(currently hard codded for 3)
        NeuralNetwork[] bestNeuralNetworks = FindBestNeuralNetworks(reproducingCreatures);

        //get rid of the previous generation of creatures after copying the best
        CullPreviousCreatures();

        //create a new goal for the creatures
        if(goalCreator != null)
        {
            goalCreator.GenerateGoal();
        }

        //create new creatures with the neural networks of the sucesful creatures
        creatures = RepopulateCreatures(bestNeuralNetworks);

        Time.timeScale = simulationSpeed;
    }

    private NeuralNetwork[] FindBestNeuralNetworks(int selectionVolume) //finds selectionVolume number of the best neuralnetworks
    {

        //gather all scoreCounters into a list
        ScoreCounter[] scoreCounters = new ScoreCounter[creatures.Length];
        for(int i = 0; i < creatures.Length; i++)
        {
            if(creatures[i].TryGetComponent(out ScoreCounter scoreCounter))
            {
                scoreCounters[i] = scoreCounter;
            }
            else
            {
                Debug.LogError("Score counter not found during evaluation");
            }
        }

        //sort the scoreCounters with bubble sort greatest to least
        bool sorted = false;
        while(!sorted)
        {
            sorted = true;
            for(int i = 0; i < scoreCounters.Length - 1; i++)
            {
                if (scoreCounters[i].GetScore() < scoreCounters[i + 1].GetScore())
                {
                    ScoreCounter temp = scoreCounters[i];
                    scoreCounters[i] = scoreCounters[i + 1];
                    scoreCounters[i + 1] = temp;
                    sorted = false;
                }
            }
        }

        //report the best scores
        Debug.Log("Best scores: " + scoreCounters[0].GetScore() + ", " + scoreCounters[1].GetScore() + ", " + scoreCounters[2].GetScore());

        //harvest the first few selectionVolume of NeuralNetworks from previous sort
        NeuralNetwork[] bestNeuralNetworks = new NeuralNetwork[selectionVolume];
        for(int i = 0; i < selectionVolume; i++)
        {
            if(scoreCounters[i].gameObject.TryGetComponent(out Simception simception))
            {
                bestNeuralNetworks[i] = simception.GetNeuralNetwork().DeepCopyNeuralNetwork(); //deep copy at this point may not be necessary, but makes the NN safe if original is deleted
                simception.Startup();

            }
            else
            {
                Debug.LogError("Simception could not be found upon repopulation");
            }
        }

        return bestNeuralNetworks;
    }
    
    private GameObject[] RepopulateCreatures(NeuralNetwork[] bestNeuralNetworks) //spawns and configures a new generation
    {
        GameObject[] newCreatures = new GameObject[creatures.Length];

        //loop through the creatures
        for(int i = 0; i < creatureVolume; i++)
        {
            //spawn creatures from prefab
            Vector3 spawnPosition = transform.position + Vector3.right * i * creatureSpaceing;
            GameObject newCreature = Instantiate(creature, spawnPosition, Quaternion.identity);

            //configure the ScoreCounter and Simception of the new creature given the new neural networks and persistent goalCreator
            ConfigureScoreCounter(newCreature);
            ConfigureSimception(bestNeuralNetworks, newCreature);

            newCreatures[i] = newCreature;
        }

        return newCreatures;
    }

    private void ConfigureScoreCounter(GameObject newCreature)
    {
        //synchronize and activate the ScoreCounter's goal with the GoalCreator
        if (newCreature.TryGetComponent(out ScoreCounter scoreCounter))
        {
            if(goalCreator != null)
            {
                scoreCounter.SetGoalCreator(goalCreator);
                scoreCounter.Startup();
            }
        }
        else
        {
            Debug.LogError("ScoreCounter could not be found upon repopulation");
        }
        
    }

    private void ConfigureSimception(NeuralNetwork[] bestNeuralNetworks, GameObject newCreature)
    {
        //synchronize and activate the simception's inputs with the GoalCreator and deal with NeuralNetwork generation
        if (newCreature.TryGetComponent(out Simception simception))
        {
            //if there are no best Neural networks, create new ones at random
            if (bestNeuralNetworks == null)
            {
                simception.GenerateRandomNeuralNetwork();
            }
            else //otherwise give creature a brain by copying the most succesful of the previous generation
            {
                //int NNIndex = Random.Range(0, bestNeuralNetworks.Length);
                int NNIndex = PowerRandom(bestNeuralNetworks.Length, reproductionSuccesBias);
                simception.SetNeuralNetwork(bestNeuralNetworks[NNIndex].DeepCopyNeuralNetwork());

                //attmpt global mutation
                if (Random.Range(0f, 100f) < globalMutationChance)
                {
                    simception.MutateNeuralNetwork(neuronMutationChance, mutationMagnitude);
                }
            }

            //useful call akin to Awake or Start in Monobehvior
            simception.SetGoalCreator(goalCreator);
            simception.Startup();

            //set goalCreator
            if (goalCreator != null)
            {
                simception.SetGoalCreator(goalCreator);
            }
        }
        else
        {
            Debug.LogError("Simception could not be found upon repopulation");
        }
    }

    private static int PowerRandom(int maxIndex, float power = 2.0f) //any values above 1 favors lower numbers
    {
        // Generate a random float between 0 and 1
        float randomValue = UnityEngine.Random.value;

        // Apply the inverse of the power-law distribution
        float scaledValue = Mathf.Pow(randomValue, power);

        // Scale it to the desired range [0, maxIndex)
        int selectedIndex = Mathf.FloorToInt(scaledValue * maxIndex);

        return selectedIndex;
    }

    private void CullPreviousCreatures() //destroy the previous generation
    {
        //loop through the creatures
        for (int i = 0; i < creatureVolume; i++)
        {
            //destroy previous dummy
            Destroy(creatures[i]);
        }
    }
}
