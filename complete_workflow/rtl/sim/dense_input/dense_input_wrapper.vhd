library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity dense_input_wrapper is
    port (
        clk                  : in std_logic;
        rst                  : in std_logic;
        read_valid           : in std_logic;
        read_data_flat       : in std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0);
        neighbor_data_flat   : in std_logic_vector(NEIGHBORS * NB_W - 1 downto 0);
        layer_index          : in integer range 0 to N_LAYERS - 1;
        input_index          : in integer range 0 to MAX_LAYER_INPUT_COUNT - 1;

        dense_input_valid    : out std_logic;
        dense_input_data_flat: out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of dense_input_wrapper is
    signal read_lanes     : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal neighbor_lanes : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
    signal output_lanes   : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    activation_lanes : for lane in 0 to MAX_MAC_UNITS - 1 generate
        read_lanes(lane) <= signed(
            read_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
        );
        dense_input_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(output_lanes(lane));
    end generate;

    unpack_neighbors : for lane in 0 to NEIGHBORS - 1 generate
        neighbor_lanes(lane) <= signed(
            neighbor_data_flat((lane + 1) * NB_W - 1 downto lane * NB_W)
        );
    end generate unpack_neighbors;

    dut : entity work.dense_input
        port map (
            clk               => clk,
            rst               => rst,
            read_valid        => read_valid,
            read_data         => read_lanes,
            neighbor_data     => neighbor_lanes,
            layer_index       => layer_index,
            input_index       => input_index,
            dense_input_valid => dense_input_valid,
            dense_input_data  => output_lanes
        );
end architecture;
