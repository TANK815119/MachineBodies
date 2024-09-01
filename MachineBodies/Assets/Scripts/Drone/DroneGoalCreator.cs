using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneGoalCreator : GoalCreator
{
    [SerializeField] private Transform goalMarker;

    private Vector3 goalPosition = Vector3.zero;

    public override void GenerateGoal()
    {
        //generate a new goal position
        goalPosition = new Vector3(0f, Random.Range(1f, 5f), 0f);

        goalMarker.position = new Vector3(0f, goalPosition.y, 0f);
    } 

    public Vector3 GetGoal()
    {
        return goalPosition;
    }
}
