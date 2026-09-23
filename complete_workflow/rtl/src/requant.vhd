library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.fixed_float_types.all;

entity requant is
    generic (
        IN_W: integer;
        IN_I: integer;
        OUT_W: integer;
        OUT_I: integer
    );
    port (
        input : in sfixed(IN_I - 1 downto IN_I - IN_W);
        output: out sfixed(OUT_I - 1 downto OUT_I - OUT_W)
    );
end entity;

architecture rtl of requant is
begin
    output <= (others => '0')
        when input(input'left) = '1'
        else resize(
            input,
            output'left,
            output'right,
            fixed_saturate,
            fixed_round
        );
end architecture;