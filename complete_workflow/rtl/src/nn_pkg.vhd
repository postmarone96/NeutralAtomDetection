library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package nn_pkg is

    constant CROP       : integer := 10;
    constant MAX_CHANNEL : integer := 16;
    constant NEIGHBORS  : integer := 8;
    constant CLASSES    : integer := 2;
    constant MAX_MAC_UNITS : integer := 100;

    constant RAW_W     : integer := 10;
    constant IN_W      : integer := 8;
    constant IN_I      : integer := 1;
    constant NB_W      : integer := 9;
    constant NB_I      : integer := 1;
    constant CAT_W     : integer := 9;
    constant CAT_I     : integer := 5;

    constant N_LAYERS  : integer := 6;
    constant N_CONV    : integer := 3;

    constant ACT_W : integer_vector(0 to N_LAYERS-1) := (8, 9, 6, 9, 9, 8);
    constant ACT_I : integer_vector(0 to N_LAYERS-1) := (1, 1, 3, 5, 5, 5);
    constant ACC_W : integer_vector(0 to N_LAYERS-1) := (14, 16, 17, 15, 16, 15);
    constant ACC_I : integer_vector(0 to N_LAYERS-1) := (3, 5, 7, 8, 7, 7);

    constant WEIGHT_W_MAX   : integer := 6;
    constant ACC_W_MAX      : integer := 17;
    constant ACT_W_MAX      : integer := 9;

end package nn_pkg;
