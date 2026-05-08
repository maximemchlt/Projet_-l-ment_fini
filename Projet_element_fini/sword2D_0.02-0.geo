// Gmsh project created on Fri May 08 14:14:28 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0.02, 0, 0, 1.0};
//+
Point(2) = {-0.02, 0, 0, 1.0};
//+
Point(3) = {0, 0.0025, 0, 1.0};
//+
Point(4) = {0, -0.0025, 0, 1.0};
//+
Line(1) = {2, 4};
//+
Line(2) = {4, 1};
//+
Line(3) = {1, 3};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {4, 3, 2, 1} = 20 Using Progression 1;
//+
Transfinite Surface {1};
