
.subckt 000397_output
* === component bounding boxes (pixel coords) ===
* i0 type=current bbox=(162,253,203,294) orientation=R0
*   pin0_rect=(162,253,203,273)
*   pin1_rect=(162,273,203,294)
* q1 type=npn bbox=(150,144,188,225) orientation=R0
*   pin0_rect=(175,144,188,171)
*   pin1_rect=(150,171,162,198)
*   pin2_rect=(175,198,188,225)
* q2 type=npn bbox=(284,180,320,257) orientation=R0
*   pin0_rect=(308,180,320,205)
*   pin1_rect=(284,205,296,231)
*   pin2_rect=(308,231,320,257)
* gnd3 type=gnd bbox=(167,317,197,342) orientation=None
*   pin0_rect=(167,317,197,342)
i0 net3 gnd i
q1 net0 net1 net2 npn
q2 net0 net2 net4 npn
.ends
