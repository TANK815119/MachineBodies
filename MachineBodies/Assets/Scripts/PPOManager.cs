using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PPOManager : MonoBehaviour
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

    //PPO stuff
    private List<Experience> experiences;
    private Simception[] simceptions;
    private NeuralNetwork_PPO neuralNetworkPPO;
    [SerializeField] private float policyLearningRate = 0.003f;
    [SerializeField] private float valueLearningRate = 0.0003f;

    private GameObject[] creatures;

    private float simLength = 0;
    private int generation = 0;

    // Start is called before the first frame update
    void Start()
    {
        //make the PPO neural network form scratch using parameters in the simception
        if (creature.TryGetComponent(out Simception simception))
        {

        }
        else
        {
            Debug.LogError("the creature has no simception");
        }

        int[] netExtents = simception.GetNetworkExtents();
        Debug.Log("Training with " + simception.GetType());
        neuralNetworkPPO = new NeuralNetwork_PPO(netExtents[0], netExtents[1], netExtents[2], netExtents[3]);
        neuralNetworkPPO.InnitializeNeuralNetworksHe();

        experiences = new List<Experience>();
        simceptions = new Simception[creatureVolume];
        creatures = new GameObject[creatureVolume];

        genText.text = "Generation: zero";

        //generate first batch at random
        if (goalCreator != null)
        {
            goalCreator.GenerateGoal();
        }

        creatures = RepopulateCreatures(neuralNetworkPPO.GetPolicyNeuralNetwork());
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Time.timeScale = simulationSpeed;

        //collect collective experiences
        for (int i = 0; i < simceptions.Length; i++)
        {
            if(simceptions[i] != null && simceptions[i].GetLastInputs() != null)
            {
                Experience newExperience = new Experience(simceptions[i].GetLastInputs(), simceptions[i].CalculateLastReward());
                experiences.Add(newExperience);
            }
        }

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
        //train the neural network on all the collected data
        neuralNetworkPPO.Train(experiences);
        NeuralNetwork newPolicyNetwork = neuralNetworkPPO.GetPolicyNeuralNetwork();

        //set learning rates
        neuralNetworkPPO.SetLearningRates(policyLearningRate, valueLearningRate);

        //get rid of the previous generation of creatures after copying the best
        CullPreviousCreatures();

        //create a new goal for the creatures
        if (goalCreator != null)
        {
            goalCreator.GenerateGoal();
        }

        //create new creatures with the neural networks of the sucesful creatures
        creatures = RepopulateCreatures(newPolicyNetwork);
        
        //reset time and experiences
        Time.timeScale = simulationSpeed;
        experiences = new List<Experience>();
    }

    private GameObject[] RepopulateCreatures(NeuralNetwork newNeuralNetwork) //spawns and configures a new generation
    {
        GameObject[] newCreatures = new GameObject[creatures.Length];

        //loop through the creatures
        for (int i = 0; i < creatureVolume; i++)
        {
            //spawn creatures from prefab
            Vector3 spawnPosition = transform.position + Vector3.right * i * creatureSpaceing;
            GameObject newCreature = Instantiate(creature, spawnPosition, Quaternion.identity);

            //configure the ScoreCounter and Simception of the new creature given the new neural network and persistent goalCreator
            ConfigureScoreCounter(newCreature);
            ConfigureSimception(newNeuralNetwork, newCreature, i);

            newCreatures[i] = newCreature;
        }

        return newCreatures;
    }

    private void ConfigureScoreCounter(GameObject newCreature)
    {
        //synchronize and activate the ScoreCounter's goal with the GoalCreator
        if (newCreature.TryGetComponent(out ScoreCounter scoreCounter))
        {
            if (goalCreator != null)
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

    private void ConfigureSimception(NeuralNetwork newNeuralNetwork, GameObject newCreature, int creatureIndex)
    {
        //synchronize and activate the simception's inputs with the GoalCreator and deal with NeuralNetwork generation
        if (newCreature.TryGetComponent(out Simception simception))
        {
            simception.SetNeuralNetwork(newNeuralNetwork);

            //useful call akin to Awake or Start in Monobehvior
            simception.SetGoalCreator(goalCreator);
            simception.Startup();

            //set goalCreator
            if (goalCreator != null)
            {
                simception.SetGoalCreator(goalCreator);
            }

            //make simception accesible
            simceptions[creatureIndex] = simception;
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
