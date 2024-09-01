using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScoreCounterAirtime : MonoBehaviour
{
    public float score;
    public GameObject floor;
    [SerializeField] private GameObject[] negativeTouchGibs;
    private CollisionDetector[] colDetArr;
    private bool hasTouched = false;

    public void Start()
    {
        //make array to store all the collision detector components
        colDetArr = new CollisionDetector[negativeTouchGibs.Length];
        for (int i = 0; i < negativeTouchGibs.Length; i++)
        {
            //assign collision detectors to every gib and give them floor data
            colDetArr[i] = negativeTouchGibs[i].AddComponent<CollisionDetector>(); //if Start calls before the next line, i'm fucked
            colDetArr[i].floor = floor;
        }
    }
    public void FixedUpdate()
    {
        //score = negativeTouchGibs[0].transform.position.z;
        if (!hasTouched)
        {
            score += 1 * Time.fixedDeltaTime; //postive value for not touching equal to seconds
            for (int i = 0; i < negativeTouchGibs.Length; i++)
            {
                if (colDetArr[i].touchingFloor)
                {
                    hasTouched = true;
                }
            }
        }
    }
}
