library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity conv_input_wrapper is
    port (
        clk              : in  std_logic;
        rst              : in  std_logic;
        read_valid       : in  std_logic;
        read_data_flat   : in  std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0);
        input_channel    : in  integer range 0 to MAX_CHANNEL - 1;
        conv_index       : in  integer range 0 to N_CONV - 1;
        tap_index        : in  integer range 0 to KERNEL_TAP_COUNT - 1;
        conv_input_valid : out std_logic;
        conv_input_flat  : out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of conv_input_wrapper is
    signal read_data_lanes : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal conv_input_lanes : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    lane_adapter : for lane in 0 to MAX_MAC_UNITS - 1 generate
        read_data_lanes(lane) <= signed(
            read_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
        );

        conv_input_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(conv_input_lanes(lane));
    end generate lane_adapter;

    dut : entity work.conv_input
        port map (
            clk              => clk,
            rst              => rst,
            read_valid       => read_valid,
            read_data        => read_data_lanes,
            input_channel    => input_channel,
            conv_index       => conv_index,
            tap_index        => tap_index,
            conv_input_valid => conv_input_valid,
            conv_input       => conv_input_lanes
        );
end architecture;
