using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneScoreCounter : ScoreCounter
{
    private float airtime = 0f;
    private bool inAir = true;

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
        if(inAir)
        {
            airtime += Time.fixedDeltaTime;
        }
    }

    public override float GetScore()
    {
        return airtime * 10f;
    }

    public override void SetGoalCreator(GoalCreator goalCreator)
    {
        throw new System.NotImplementedException();
    }
}
