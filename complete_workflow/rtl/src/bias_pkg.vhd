library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package bias_pkg is

    constant BIAS_BITS        : integer := 32;
    constant BIAS_ROM_BITS    : integer := 512;
    constant BIAS_ROM_DEPTH   : integer := 10;

end package bias_pkg;
