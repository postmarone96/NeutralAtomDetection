library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package weights_pkg is

    constant WEIGHT_BITS      : integer := 8;
    constant WEIGHT_SLOTS_PER_WORD : integer := 16;
    constant WEIGHT_ROM_BITS  : integer := 128;
    constant WEIGHT_ROM_DEPTH : integer := 268;
    constant MAX_LAYER_INPUT_COUNT : integer := 64;

    constant LAYER_INPUT_COUNT : integer_vector(0 to 6-1) := (1, 4, 8, 64, 16, 8);

end package weights_pkg;
