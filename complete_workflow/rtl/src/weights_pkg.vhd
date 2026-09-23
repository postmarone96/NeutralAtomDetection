library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package weights_pkg is

    constant WEIGHT_BITS      : integer := 8;
    constant WEIGHTS_PER_WORD : integer := 9;
    constant WEIGHT_ROM_BITS  : integer := 72;
    constant BIAS_ROM_BITS    : integer := 32;
    constant WEIGHT_ROM_DEPTH : integer := 238;
    constant WEIGHT_VALUE_COUNT : integer := 2132;
    constant BIAS_ROM_DEPTH   : integer := 46;

    constant WEIGHT_BASE : integer_vector(0 to 6-1) := (0, 36, 324, 1476, 1988, 2116);
    constant WEIGHT_COUNT : integer_vector(0 to 6-1) := (36, 288, 1152, 512, 128, 16);
    constant WEIGHT_WORD_BASE : integer_vector(0 to 6-1) := (0, 4, 36, 164, 221, 236);
    constant WEIGHT_WORD_COUNT : integer_vector(0 to 6-1) := (4, 32, 128, 57, 15, 2);
    constant WEIGHT_LAST_WORD_COUNT : integer_vector(0 to 6-1) := (9, 9, 9, 8, 2, 7);
    constant BIAS_BASE : integer_vector(0 to 6-1) := (0, 4, 12, 28, 36, 44);
    constant BIAS_COUNT : integer_vector(0 to 6-1) := (4, 8, 16, 8, 8, 2);

end package weights_pkg;
