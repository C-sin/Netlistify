
.subckt 000039_output
* === component bounding boxes (pixel coords) ===
* i0 type=current bbox=(204,75,243,114) orientation=R0
*   pin0_rect=(204,75,243,94)
*   pin1_rect=(204,94,243,114)
* i1 type=current bbox=(98,251,137,290) orientation=R0
*   pin0_rect=(98,251,137,270)
*   pin1_rect=(98,270,137,290)
* c2 type=capacity bbox=(295,212,326,235) orientation=None
*   pin0_rect=(295,212,326,223)
*   pin1_rect=(295,223,326,235)
* gnd3 type=gnd bbox=(298,252,323,285) orientation=None
*   pin0_rect=(298,252,323,285)
* gnd4 type=gnd bbox=(105,304,129,338) orientation=None
*   pin0_rect=(105,304,129,338)
* c5 type=capacity bbox=(155,153,176,185) orientation=None
*   pin0_rect=(155,153,165,185)
*   pin1_rect=(165,153,176,185)
i0 net0 net2 i
i1 net1 gnd i
c2 net2 net2 c
c5 net1 net2 c
.ends
