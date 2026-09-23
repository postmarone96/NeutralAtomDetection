library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use work.helpers_pkg.all;

entity mac_array is
    generic(
        ACT_W_MAX:      integer;
        WEIGHT_W_MAX:   integer;
        ACC_W_MAX:      integer;
        MAX_MAC_UNITS:  integer;
        MAC_UNITS:      integer
    );

    port(
        clk:        in std_logic;
        rst:        in std_logic;
        clear:      in std_logic;
        valid:      in std_logic;

        activation: in signed_vector(0 to MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        weight:     in signed_vector(0 to MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);

        accum:      out signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of mac_array is 
    signal accum_temp: signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);

begin

    -- assertion 
    assert MAC_UNITS <= MAX_MAC_UNITS
        report "MAC units greater than Max MAC unit count"
        severity failure;
    
    -- MAC operations
    process(clk, rst)
    begin
        if rst = '1' then
            accum_temp <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if valid = '1' and clear = '0' then
                for i in 0 to MAC_UNITS - 1 loop
                    accum_temp(i) <= accum_temp(i) + resize(weight(i) * activation(i), ACC_W_MAX);
                end loop;
            elsif clear = '1' then
                    accum_temp <= (others => (others => '0'));
            end if;
        end if;
    end process;

    -- accumulator output
    accum <= accum_temp;
    
end architecture;
