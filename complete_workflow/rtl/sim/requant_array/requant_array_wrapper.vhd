library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity requant_array_wrapper is
    port (
        layer_index          : in integer range 0 to N_LAYERS - 1;
        input_valid          : in std_logic;
        accum_data_flat      : in std_logic_vector(MAX_MAC_UNITS * ACC_W_MAX - 1 downto 0);
        output_valid         : out std_logic;
        activation_data_flat : out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of requant_array_wrapper is
    signal accum_lanes      : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
    signal activation_lanes : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    lanes : for lane in 0 to MAX_MAC_UNITS - 1 generate
        accum_lanes(lane) <= signed(
            accum_data_flat((lane + 1) * ACC_W_MAX - 1 downto lane * ACC_W_MAX)
        );
        activation_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(activation_lanes(lane));
    end generate;

    dut : entity work.requant_array
        port map (
            layer_index     => layer_index,
            input_valid     => input_valid,
            accum_data      => accum_lanes,
            output_valid    => output_valid,
            activation_data => activation_lanes
        );
end architecture;
