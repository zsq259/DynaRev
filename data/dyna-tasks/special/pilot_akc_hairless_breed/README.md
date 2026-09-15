# Pilot: AKC Hairless Dog Breeds

Batch0 seed #3 (`om2w_eb2db4b7`, accepted), domain `akc.org`.

## Base task (feasible)

`Find a hairless dog breed with medium barking.`

Dropped upstream `energetic` / Activity Level: oracle showed `hairless ∩ activity` is empty on current AKC facets.

## Construction

Follows pipeline Step1–3 style + Formal4 three-variant layout (one core revision each):

| variant | i0 | i1 |
|---------|----|----|
| additive | hairless dog breed | add medium barking |
| override | hairless + low barking | replace barking → medium |
| conditional | hairless + medium barking | if no medium match, any barking OK |

No extra answer-format constraints beyond Step1 extractions.

`start_url`: `https://www.akc.org/dog-breeds/`
