using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class ScoreCounter : MonoBehaviour
{
    public abstract void Startup();

    public abstract float GetScore();

    public abstract void SetGoalCreator(GoalCreator goalCreator);
}
