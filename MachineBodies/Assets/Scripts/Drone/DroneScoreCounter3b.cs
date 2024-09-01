using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneScoreCounter3b : ScoreCounter
{
    private float airtime = 0f;
    private bool inAir = true;

    private DroneGoalCreator3 goalCreator;
    private Vector3 goalPosition;
    private float distanceIntegral = 0;

    public override void Startup()
    {
        goalPosition = goalCreator.GetGoal();
        goalPosition.x = transform.position.x;
    }

    private void OnCollisionStay(Collision collision)
    {
        inAir = false;
    }

    private void OnCollisionExit(Collision collision)
    {
        inAir = true;
    }

    private void FixedUpdate()
    {
        if (inAir)
        {
            airtime += Time.fixedDeltaTime;
        }

        distanceIntegral = -1 * Vector3.Distance(goalPosition, transform.position);
    }

    public override float GetScore()
    {
        return  distanceIntegral;
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        this.goalCreator = (DroneGoalCreator3)goalCreator;
    }
}
