using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneScoreCounter3 : ScoreCounter
{
    private float airtime = 0f;
    private bool inAir = true;
    private float yDistIntegral = 0;

    private DroneGoalCreator goalCreator;
    private Vector3 goalPosition;

    public override void Startup()
    {
        goalPosition = goalCreator.GetGoal();
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

        yDistIntegral += -1 * Mathf.Abs(goalPosition.y - transform.position.y) * Time.fixedDeltaTime;
    }

    public override float GetScore()
    {
        return airtime + yDistIntegral;
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        this.goalCreator = (DroneGoalCreator)goalCreator;
    }
}
