library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity layer_output_wrapper is
    port (
        clk                : in std_logic;
        rst                : in std_logic;
        input_valid        : in std_logic;
        layer_index        : in integer range 0 to N_LAYERS - 1;
        output_block_index : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
        input_data_flat    : in std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0);

        output_valid       : out std_logic;
        output_data_flat   : out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of layer_output_wrapper is
    signal input_lanes  : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal output_lanes : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    lanes : for lane in 0 to MAX_MAC_UNITS - 1 generate
        input_lanes(lane) <= signed(
            input_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
        );
        output_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(output_lanes(lane));
    end generate;

    dut : entity work.layer_output
        port map (
            clk                => clk,
            rst                => rst,
            input_valid        => input_valid,
            layer_index        => layer_index,
            output_block_index => output_block_index,
            input_data         => input_lanes,
            output_valid       => output_valid,
            output_data        => output_lanes
        );
end architecture;
