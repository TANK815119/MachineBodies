using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Experience
{
    public float[] State { get; private set; }
    public float[] Output { get; private set; }
    public float[] Action { get; private set; }
    public float[] OldOutput { get; private set; }
    public float[] OldAction { get; private set; }
    public float Reward { get; private set; }
    //public float[] NextState { get; private set; }
    //public bool Done { get; private set; }

    public Experience(float[] state, float[] output, float[] action, float[] oldOutput, float[] oldAction, float reward) //NOTE: Not yet implemented with heuristics
    {
        State = state;
        Output = output;
        Action = action;
        OldOutput = oldOutput;
        OldAction = oldAction;
        Reward = reward;
    }

    public Experience(float[] state, float reward)
    {
        State = state;
        //Action = action;
        Reward = reward;
        //NextState = nextState;
        //Done = done;
    }
    
}
