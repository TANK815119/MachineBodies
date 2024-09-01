using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneScoreCounter2 : ScoreCounter
{
    private float airtime = 0f;
    private bool inAir = true;
    private float yDistIntegral = 0;

    public override void Startup()
    {
        //throw new System.NotImplementedException();
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

        yDistIntegral += -1 * Mathf.Abs(2 - transform.position.y) * Time.fixedDeltaTime;
    }

    public override float GetScore()
    {
        return airtime * 3f + yDistIntegral;
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        throw new System.NotImplementedException();
    }
}
