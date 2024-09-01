using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

//every 3 seconds
//delete the previous 3 dummies
//instantiate the next 3 dummies
public class SimpleReinstantiator : MonoBehaviour
{
    [SerializeField] private GameObject dummy;
    [SerializeField] private float coolDownTimer = 3f;
    [SerializeField] private int dummyVolume = 3;
    [SerializeField] private float spawnHeight = 2f;
    [SerializeField] private TextMeshProUGUI genText;
    private GameObject[] dummyArr;

    private float time = 0;
    private float cooldown = 0;
    private bool resetable = true;
    private float spaceBetweenDummies = 2f;
    private float genStartPosititon;
    private int generation;

    public void Start()
    {
        dummyArr = new GameObject[dummyVolume];
        genStartPosititon = (dummyVolume / 2) * spaceBetweenDummies * -1;
        genText.text = "Generation: zero";
    }

    public void FixedUpdate()
    {
        time += 1 * Time.fixedDeltaTime;
        
        if(resetable == true)
        {
            resetDummies();
        }

        //need this so it doesnt reset a billion times in one second
        if (resetable == false)
        {
            cooldown -= 1 * Time.fixedDeltaTime;
            if(cooldown <= 0)
            {
                resetable = true;
            }
        }
    }
    private void resetDummies()
    {
        for(int i = 0; i < dummyVolume; i++)
        {
            //destroy previous dummy
            if(dummyArr[i] != null)
            {
                Destroy(dummyArr[i]);
            }

            //make dummy
            Vector3 position = new Vector3(genStartPosititon + i * spaceBetweenDummies, spawnHeight, 0);
            Vector3 rotation = new Vector3(0, 0, 0);
            GameObject thisDummy = Instantiate(dummy, position, Quaternion.Euler(rotation));
            dummyArr[i] = thisDummy;
        }

        //update generation text
        generation++;
        genText.text = "Generation: " + generation;

        //set up cooldown
        resetable = false;
        cooldown = coolDownTimer;
    }
}
