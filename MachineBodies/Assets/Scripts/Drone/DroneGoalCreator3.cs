using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneGoalCreator3 : GoalCreator
{
    [SerializeField] private Transform goalMarker;

    private Vector3 goalPosition = Vector3.zero;

    public override void GenerateGoal()
    {
        //generate a new goal position
        //if (goalPosition.z == 0)
        //{
        //    goalPosition = new Vector3(0f, Random.Range(-5f, 5f), Random.Range(-5f, 5f));
        //}
        //else if (goalPosition.z > 0)
        //{
        //    goalPosition = new Vector3(0f, Random.Range(-5f, 5f), Random.Range(-5f, 0f));
        //}
        //else
        //{
        //    goalPosition = new Vector3(0f, Random.Range(-5f, 5f), Random.Range(0f, 5f));
        //}
        //goalPosition = new Vector3(0f, Random.Range(3f, 3f), 0f);
        goalPosition = new Vector3(0f, 0f, 0f);

        goalMarker.position = new Vector3(0f, goalPosition.y, goalPosition.z);
    } 

    public Vector3 GetGoal()
    {
        return goalPosition;
    }
}
