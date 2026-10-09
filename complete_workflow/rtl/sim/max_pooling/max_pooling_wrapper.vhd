library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity max_pooling_wrapper is
    port (
        conv_index          : in integer range 0 to N_CONV - 1 := 0;
        activation_in_flat  : in std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
                              := (others => '0');
        activation_out_flat : out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of max_pooling_wrapper is
    signal activation_in_lanes  : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal activation_out_lanes : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    lanes : for lane in 0 to MAX_MAC_UNITS - 1 generate
        activation_in_lanes(lane) <= signed(
            activation_in_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
        );
        activation_out_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(activation_out_lanes(lane));
    end generate;

    dut : entity work.max_pooling
        port map (
            conv_index     => conv_index,
            activation_in  => activation_in_lanes,
            activation_out => activation_out_lanes
        );
end architecture;
