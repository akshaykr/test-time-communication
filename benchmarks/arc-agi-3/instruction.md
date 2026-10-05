# ARC-AGI-3 Interactive Control

## Context

You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

## Fairness rule: no internet or public solutions

This is a closed-book ARC-AGI-3 run. Do not use the internet, web search,
browsers, search engines, public ARC Prize pages, scorecards, recordings,
replays, GitHub repositories, or any other online source to find examples,
hints, action traces, or solutions.

You may use only the task instructions, local files already present in the
environment, `arcctl` observations/actions/scores, and notes produced during
this run. Do not use shell commands such as `curl`, `wget`, Python HTTP
libraries, or similar tools to fetch external web content.

## Turn

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

Call exactly one action. After each action, observe the next Frame, 
then continue turn by turn until WIN, or the action budget is exhausted.

## Stopping rule

Do not end your session just because you have a partial solution, a theory, a
promising state, a shared note, or a dead end. Those are not stopping
conditions.

You stop only if your own session output proves one of the following:

1. `target_reached` is `true` when target-level mode is configured.
2. The current level action budget is exhausted, with
   `action_budget.current_level_actions_remaining` equal to `0`.

If `target_reached` is `false`, `state` is `NOT_FINISHED`, and current-level
actions remain, your session is not done. Your next turn must call another
`arcctl` command and continue from the current state. After GAME_OVER, call
arcctl reset and continue playing while budget remains.

## Terminal control

`arcctl` is the terminal interface for this task. It controls one ARC-AGI-3
environment session for the configured benchmark game and translates shell
commands into ARC-AGI-3 actions.

If this task is run with multiple agents, each agent's `arcctl` controls that
agent's own ARC session. Shared notes, leaderboards, or scorecard entries from
other agents are only hints. They do not clear your session. Before stopping,
verify with your own `arcctl observe` or `arcctl score` output that your own
session has advanced `levels_completed`; when target-level mode is configured,
your own session must report `target_reached: true`.

Start the game and inspect the state:

```bash
arcctl reset
arcctl observe
arcctl actions
```

Use `arcctl observe` to inspect the current state, score, frame, available
actions, and action budget. Use `arcctl actions` to see which actions are
currently available. Then call exactly one action at a time:

| ARC-AGI-3 action | `arcctl` command | Meaning |
| --- | --- | --- |
| `RESET` | `arcctl reset` or `arcctl step RESET` | Start or restart a game. Must be called first when `NOT_PLAYED` or after `GAME_OVER` to play again. Reset is charged to the same scorecard/session. |
| `ACTION1` | `arcctl step ACTION1` or `arcctl step 1` | Send this simple input action (1, W, Up). |
| `ACTION2` | `arcctl step ACTION2` or `arcctl step 2` | Send this simple input action (2, S, Down). |
| `ACTION3` | `arcctl step ACTION3` or `arcctl step 3` | Send this simple input action (3, A, Left). |
| `ACTION4` | `arcctl step ACTION4` or `arcctl step 4` | Send this simple input action (4, D, Right). |
| `ACTION5` | `arcctl step ACTION5` or `arcctl step 5` | Send this simple input action (5, Enter, Spacebar, Delete). |
| `ACTION6` | `arcctl step ACTION6 --data '{"x": 12, "y": 34}'` | Send this complex input action (6, Click, Point) with `x` and `y` coordinates. |

Use `arcctl score` to inspect current official scorecard progress. Both reset and action consume the action budget and does NOT restore the budget.

## Notes

1. `arcctl observe` prints JSON with a compact grid frame. Increase `--max-frames`, `--max-rows`, or `--max-cols` if you need more of the observation.
2. ARC-AGI-3 action budgets are enforced per level, and the game total is the sum of those level budgets. You cannot spend later-level budget while still on the current level.

| `action_budget` field | Meaning |
| --- | --- |
| `current_level_budget` | Maximum actions allowed on the current level. |
| `current_level_actions_used` | Actions already spent on the current level. |
| `current_level_actions_remaining` | Actions still available before the current level is exhausted. |
| `max_actions` | Total budget across all levels in the game. |
| `actions_used` | Total actions charged to the ARC scorecard/session. |
| `actions_remaining` | Total game budget still unused; later-level budget is only usable after reaching those levels. |

3. If you are cooperating with multiple agents, your final performance is measured by the average performance of each agent. Note that budget is enforced per agent, so one agent may fail to proceed to the next level after budget exhaustion even if the other succeeds.