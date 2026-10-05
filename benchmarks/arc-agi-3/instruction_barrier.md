# Sync barrier (active in this run)

This game runs with a sync barrier so no agent races ahead of its teammates.
`arcctl` responses include a `barrier` object, e.g. `"barrier": {"open": false,
"your_level": 1, "your_round": 1, "blocking_level": 0, "blocking_round": 3}`.
Check it cheaply with `arcctl barrier`, a lightweight status poll that returns
only this object — no grid frame, budget, or scorecard payload.

- `barrier.open` is `true` (or there is no `barrier` field): play normally.
- `barrier.open` is `false`: you are ahead of your teammates. Only `arcctl step`
  and `arcctl RESET` are refused — no action is taken and no budget charged.
  Every other command still works: any shell command, all file I/O, and the
  read-only `arcctl barrier` / `observe` / `actions` / `score`. You cannot
  advance the game until the barrier opens.

A closed barrier is not a pause and not a stopping condition — keep your session
alive. While it is closed, coordinate with your teammates, then re-check with
`arcctl barrier` (poll this lightweight check rather than `arcctl observe`, so
you don't re-pull the full grid each time). The barrier opens automatically once
your teammates reach your phase (or leave the game), and your next `arcctl step`
then succeeds.
