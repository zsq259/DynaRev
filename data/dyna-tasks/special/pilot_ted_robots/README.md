# Pilot: TED Robots Talks

Batch0 seed #4 (`om2w_bb5d90e6`, accepted), domain `ted.com`.

## Base task (feasible)

`Find the most viewed TED talk on the topic of robots that lasts between 12 and 18 minutes.`

Oracle 2026-09-13: `/talks` opens; Topics / Duration / Sort are real facets. `Robots` = 109; `Robots ∩ 12-18 minutes` = 38; `Robots ∩ 6-12 minutes` nonempty. Most viewed under the 12-18 ∩ robots view: *The astounding athletic power of quadcopters* (15:50).

## Construction

| variant | i0 | i1 |
|---------|----|----|
| additive | TED talk on the topic of robots | add 12-18 minutes + most viewed |
| override | topic of robots + 6-12 minutes | replace duration → 12-18 + most viewed |
| conditional | topic of robots + 12-18 + most viewed | if no 12-18 match, switch to 6-12 + most viewed |

Wording keeps upstream **on the topic of robots**; constraint c2 is `Topic must be robots (via TED Topics facet)` so keyword search `q=robots` is not the intended path.

No extra answer-format constraints beyond the upstream task.

`start_url`: `https://www.ted.com/talks`
