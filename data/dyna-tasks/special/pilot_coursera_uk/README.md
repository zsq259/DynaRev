# Pilot: Coursera UK programming courses

Formal4 **M2W2** family draft (`coursera_uk`), domain `coursera.org`. **Rejected 2026-09-16** (runner Chromium: Upgrade Your Browser). Not a Batch0 seed.

## Base task (upstream)

I want to learn computer programming skills on Coursera. Could you please recommend a course related to programming and algorithms (i.e., with skill tag computer programming), as well as a Python-related course (i.e., with skill tag Python programming), both offered by UK universities and with a rating of 4.5 or higher? Please provide the homepage links for these courses on Coursera.

## Construction

Multi-turn. Override and conditional occupy **one** turn (rating 4.5 vs 4.0, pair `lsd_012`); the two-course + homepage-link request is a later additive turn. UK university and Computer Programming stay in i0 across variants.

| variant | i0 | i1 | i2 |
|---------|----|----|----|
| additive | Computer Programming courses on Coursera from UK universities | add rating 4.5+ | one algorithms course + one Python course + homepage links |
| override | UK Computer Programming courses rated 4.0+ | replace rating → 4.5+, not 4.0 | same as additive i2 |
| conditional | UK Computer Programming courses rated 4.5+ | if fewer than 2 listed, switch to 4.0+ | same as additive i2 |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) |
|---------|----------------|----------------|
| additive | Coursera search for Computer Programming courses from UK universities is established | 4.5+ rating criterion is in effect on that search |
| override | Coursera search for Computer Programming courses from UK universities rated 4.0+ is established | same as additive m2 |
| conditional | Coursera search for Computer Programming courses from UK universities rated 4.5+ is established | working search follows the rating branch (keep 4.5 after the default count is known, or move to 4.0+) |

`start_url`: `https://www.coursera.org/courses`

## Oracle note

Not live-counted. Coursera catalog typically has Skills / Partner / Ratings facets; UK is a partner-country constraint, not always a single checkbox. Conditional “fewer than 2” is expected **false** if several UK 4.5 Computer Programming courses appear, analogous to Cleveland’s default branch. Recheck on the public catalog before accepting the seed.
