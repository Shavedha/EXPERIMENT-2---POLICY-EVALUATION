# EXPERIMENT 2: POLICY EVALUATION

## AIM
To develop a python program to evaluate the performance of a given policy.

## PROBLEM STATEMENT
The problem statement defines a Stochastic Bandit walk environment with five states excluding the Goal state and the hole state.
### State Space:
{0(HOLE),1,2,3,4,5,6(GOAL)}<br>
Thus it includes 2 terminal states(0 and 6) and 5 non-terminal states.
### Action Space:
Two actions 0 and 1 are possible,<br>
{0(LEFT),1(RIGHT)}
### Reward Function:
* Reaches Goal state: +1
* Otherwise: 0
### Tranisition Probability:
* 50% - Agent moves in the desired direction
* 33.33% - Agent stays in the same state
* 16.66% - Agent movies in orthogonal direction 

## POLICY EVALUATION FUNCTION
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s] [pi(s)]:
          V[s] +=prob * (reward + gamma * prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()

    return V
```
## OUTPUT
### Policy 1
* Policy
  
  <img width="539" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/88092b7e-a85c-4145-aaf2-a7f131d9c09b">

* Success Probability
  
  <img width="479" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/f07d7e93-40d2-4286-80c1-3383768a2bb7">

* State value Function
  
  <img width="551" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/0a6e3cda-041c-4438-b0e3-46d5c4d0b631">

* Array

  <img width="417" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/556ca075-10bc-4272-9097-79a768d500e1">


### Policy 2
* Policy
  
  <img width="568" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/ecff1ccc-7f7f-4c1e-bf5d-9d7f755ca3df">

* Success Probability
  
  <img width="537" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/38bef75d-1718-401c-8343-8f8ea0b7561e">

* State Value Function
  
  <img width="535" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/5568a885-dbd3-425e-a1d4-307ab2d4e140">
  
* Array
  
  <img width="423" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/4b15a9ea-2209-44ba-b283-884b2558a5c9">
  
### Comparison

<img width="362" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/b4d35b43-761c-4503-b7ed-4e9966c4005f">
<br>
<img width="314" alt="image" src="https://github.com/Shavedha/EXPERIMENT-2---POLICY-EVALUATION/assets/93427376/060d4cf6-2a7c-4f0b-8796-678108c66495">

### Inference
By comparing the state value functions of both the policies we can see that <br> Policy 2 is better in terms of reaching the Goal State.

## RESULT
Thus a program is executed to evaluate the performance of a policy.

