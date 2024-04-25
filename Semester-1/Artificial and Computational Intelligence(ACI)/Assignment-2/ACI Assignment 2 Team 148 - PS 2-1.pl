% List only the BITS (Name) of active contributors in this assignment:
% 1. 2023aa05152 - Shruti S Kumar
% 2. 2023aa05072 - Peyala Samarasimha Reddy
% 3. 2023aa05195 - Viswanath Hemanth Chadalawada
% 4. 2023aa05229 - POLAVARAPU SATYA DURGA LALITHA RAO SARMA
% 5. 2023aa05930 - Sridhar K


% Decision Tree Rules
predict_water_source(Distance_from_Lake, WaterSource) :-
    Distance_from_Lake < 10,
    WaterSource = lake.

predict_water_source(Rainfall, Distance_from_River, Distance_from_Lake, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River < 8,
    Rainfall < 200,
    WaterSource = river.

predict_water_source(Rainfall, Distance_from_River, Distance_from_Lake, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River < 8,
    Rainfall >= 200,
    WaterSource = rain.

predict_water_source(Rainfall, Distance_from_River, Distance_from_Lake, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall >= 150,
    WaterSource = rain.

predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall < 150,
    SandyAquifer = no,
    Distance_from_Lake >= 14,
    WaterSource = rain.

predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, Distance_from_Beach, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall < 150,
    SandyAquifer = no,
    Distance_from_Lake < 14,
    Distance_from_Beach >= 0,
    WaterSource = lake.

predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, Distance_from_Beach, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall < 150,
    SandyAquifer = yes,
    Distance_from_Beach >= 5,
    WaterSource = groundwater.

predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, Distance_from_Beach, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall < 150,
    SandyAquifer = yes,
    Distance_from_Beach < 5,
    Distance_from_River >= 20,
    WaterSource = rain.

predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, Distance_from_Beach, WaterSource) :-
    Distance_from_Lake >= 10,
    Distance_from_River >= 8,
    Rainfall < 150,
    SandyAquifer = yes,
    Distance_from_Beach < 5,
    Distance_from_River < 20,
    WaterSource = river.

% User Input
example_prediction(WaterSource) :-
    write('Enter rainfall amount (mm): '),
    read(Rainfall),
    write('Is there a sandy aquifer? (yes/no): '),
    read(SandyAquifer),
    write('Enter distance from river (km): '),
    read(Distance_from_River),
    write('Enter distance from lake (km): '),
    read(Distance_from_Lake),
    write('Enter distance from beach (km): '),
    read(Distance_from_Beach),
    predict_water_source(Rainfall, SandyAquifer, Distance_from_River, Distance_from_Lake, Distance_from_Beach, WaterSource).