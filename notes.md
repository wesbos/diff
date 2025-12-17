In order to make a good challenge, we need to avoid pixel pushing.

Common use cases for pixel pushing are when there is a mix of:

- unknown font size
- unknown padding
- obtuse centering or alignment

Ideas on making things less about pixel pushing:

- pre-set font sizes?
- all units should be set in px to avoid half pixels with em/rems
- any explicit gaps, padding, margins should always be set in increments of 5px
- any other values (borderers, radius ) must be a clean pixel count (1,2,3,4...) or an increment of 5% (50%, 55%...)

## Problems?

1. straight up background color gets you to 93%. Is that a problem if both people have that advantage?
2. Literally nothing is already a 50%. Need to test if we stretch this from 0 - 100
