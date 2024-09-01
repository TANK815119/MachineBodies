using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionDetector : MonoBehaviour
{
    public GameObject floor;
    public bool touchingFloor = false;

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.Equals(floor))
        {
            touchingFloor = true;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.Equals(floor))
        {
            touchingFloor = false;
        }
    }
}
