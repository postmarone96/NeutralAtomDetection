library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.bias_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity biases_handler_wrapper is
    port (
        conv_enable        : in std_logic;
        layer_index        : in integer range 0 to N_LAYERS - 1;
        output_block_index : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;

        bias_row           : out integer range 0 to BIAS_ROM_DEPTH - 1;
        bias_out_flat      : out std_logic_vector(MAX_MAC_UNITS * ACC_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of biases_handler_wrapper is
    signal selected_row  : integer range 0 to BIAS_ROM_DEPTH - 1;
    signal bias_rom_data : signed(BIAS_ROM_BITS - 1 downto 0);
    signal bias_lanes    : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
begin
    dut : entity work.biases_handler
        port map (
            conv_enable        => conv_enable,
            layer_index        => layer_index,
            output_block_index => output_block_index,
            bias_row           => selected_row,
            bias_rom_data      => bias_rom_data,
            bias_out           => bias_lanes
        );

    memory : entity work.bias_rom
        generic map (
            ROM_DEPTH   => BIAS_ROM_DEPTH,
            WORD_LENGTH => BIAS_ROM_BITS,
            FILE_NAME   => "biases.mem"
        )
        port map (
            addr     => selected_row,
            data_out => bias_rom_data
        );

    bias_row <= selected_row;

    flatten : for lane in 0 to MAX_MAC_UNITS - 1 generate
        bias_out_flat((lane + 1) * ACC_W_MAX - 1 downto lane * ACC_W_MAX)
            <= std_logic_vector(bias_lanes(lane));
    end generate;
end architecture;
