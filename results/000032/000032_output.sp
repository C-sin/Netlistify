
.subckt 000032_output
* === component bounding boxes (pixel coords) ===
* m0 type=pmos bbox=(190,67,235,112) orientation=R0
*   pin0_rect=(223,100,235,112)
*   pin1_rect=(185,82,190,97)
*   pin2_rect=(223,67,235,78)
*   pin3_rect=(223,67,235,78)
* m1 type=pmos bbox=(24,67,70,112) orientation=MY
*   pin0_rect=(24,100,35,112)
*   pin1_rect=(70,82,75,97)
*   pin2_rect=(24,67,35,78)
*   pin3_rect=(24,67,35,78)
* m2 type=nmos bbox=(190,232,235,277) orientation=R0
*   pin0_rect=(223,232,235,243)
*   pin1_rect=(185,247,190,262)
*   pin2_rect=(223,265,235,277)
*   pin3_rect=(223,265,235,277)
* m3 type=nmos bbox=(116,177,162,222) orientation=R0
*   pin0_rect=(150,177,162,188)
*   pin1_rect=(111,192,116,207)
*   pin2_rect=(150,210,162,222)
*   pin3_rect=(150,210,162,222)
* m4 type=nmos bbox=(22,231,70,277) orientation=MY
*   pin0_rect=(22,231,34,242)
*   pin1_rect=(70,246,75,261)
*   pin2_rect=(22,265,34,277)
*   pin3_rect=(22,265,34,277)
* r5 type=resistor bbox=(217,312,245,347) orientation=None
*   pin0_rect=(217,312,245,329)
*   pin1_rect=(217,329,245,347)
* gnd6 type=gnd bbox=(218,361,243,394) orientation=None
*   pin0_rect=(218,361,243,394)
* gnd7 type=gnd bbox=(15,286,41,320) orientation=None
*   pin0_rect=(15,286,41,320)
m0 net1 net1 net0 net0 pmos4
m1 net4 net1 net0 net0 pmos4
m2 net1 net4 net5 net5 nmos4
m3 net2 net1 net4 net4 nmos4
m4 net4 net4 gnd gnd nmos4
r5 net5 gnd r
.ends
