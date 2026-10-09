library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.nn_pkg.all;

package activation_map_pkg is

    constant KERNEL_TAP_COUNT : integer := 9;
    constant CONV_OUTPUT_BLOCK_COUNT : integer_vector(0 to N_CONV-1) := (4, 2, 1);
    constant MAX_OUTPUT_BLOCK_COUNT : integer := 4;

end package activation_map_pkg;
