library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package nn_pkg is

    constant CROP      : integer := 10;
    constant C1        : integer := 4;
    constant C2        : integer := 8;
    constant C3        : integer := 16;
    constant NEIGHBORS : integer := 8;
    constant CLASSES   : integer := 2;
    constant FLAT      : integer := 64;
    constant MAC_UNITS : integer := 64;

    constant RAW_W     : integer := 10;
    constant RAW_I     : integer := 10;
    constant SUM_W     : integer := 16;
    constant SUM_I     : integer := 16;
    constant IN_W      : integer := 10;
    constant IN_I      : integer := 1;
    constant NB_W      : integer := 10;
    constant NB_I      : integer := 1;

    constant N_LAYERS  : integer := 6;
    constant N_CONV    : integer := 3;

    constant WEIGHT_W : integer_vector(0 to N_LAYERS-1) := (7, 6, 5, 7, 7, 5);
    constant WEIGHT_I : integer_vector(0 to N_LAYERS-1) := (1, 2, 2, 2, 1, 1);
    constant ACT_W : integer_vector(0 to N_LAYERS-1) := (10, 9, 8, 10, 10, 8);
    constant ACT_I : integer_vector(0 to N_LAYERS-1) := (1, 1, 3, 5, 5, 5);
    constant ACC_W : integer_vector(0 to N_LAYERS-1) := (18, 18, 17, 18, 18, 16);
    constant ACC_I : integer_vector(0 to N_LAYERS-1) := (3, 5, 7, 8, 7, 7);
    constant LAYER_MACS : integer_vector(0 to N_LAYERS-1) := (3600, 7200, 4608, 512, 128, 16);

    constant WEIGHT_W_MAX  : integer := 7;
    constant ACC_W_MAX     : integer := 18;
    constant ACT_W_MAX     : integer := 10;

end package nn_pkg;
