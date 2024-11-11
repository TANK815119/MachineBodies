using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalCreator_Ragdoll_1 : GoalCreator
{
    [SerializeField] private GameObject floor;

    public override void GenerateGoal()
    {
        //do nothing, we already have the floor
    }

    public GameObject GetFloor()
    {
        return floor;
    }
}
