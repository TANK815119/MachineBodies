using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class Experience
{
    public float[] State { get; private set; }
    public float Action { get; private set; }
    public float Reward { get; private set; }
    public float[] NextState { get; private set; }
    public bool Done { get; private set; }

    public Experience(float[] state, float action, float reward, float[] nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}
