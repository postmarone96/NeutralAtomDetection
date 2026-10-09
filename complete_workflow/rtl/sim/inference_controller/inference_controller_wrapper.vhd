library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity inference_controller_wrapper is
    port (
        clk                      : in std_logic;
        rst                      : in std_logic;
        start                    : in std_logic;

        busy                     : out std_logic;
        done                     : out std_logic;
        result_valid             : out std_logic;
        conv_enable              : out std_logic;
        layer_index              : out integer range 0 to N_LAYERS - 1;
        input_index              : out integer range 0 to MAX_LAYER_INPUT_COUNT - 1;
        output_block_index       : out integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
        tap_index                : out integer range 0 to KERNEL_TAP_COUNT - 1;

        activation_read_enable   : out std_logic;
        activation_write_enable  : out std_logic;
        weight_read_enable       : out std_logic;
        mac_load_bias            : out std_logic;
        mac_valid                : out std_logic;
        layer_output_input_valid : out std_logic
    );
end entity;

architecture rtl of inference_controller_wrapper is
    signal activation_valid   : std_logic := '0';
    signal activation_ram_valid : std_logic := '0';
    signal weights_valid      : std_logic := '0';
    signal layer_output_valid : std_logic := '0';
    signal mac_input_valid    : std_logic := '0';
    signal mac_product_valid  : std_logic := '0';
    signal mac_busy           : std_logic;

    signal selected_layer       : integer range 0 to N_LAYERS - 1;
    signal selected_block       : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
    signal act_read_enable      : std_logic;
    signal weight_enable        : std_logic;
    signal mac_enable           : std_logic;
    signal layer_block_valid    : std_logic;

    function block_count(layer : integer) return integer is
    begin
        if layer < N_CONV then
            return CONV_OUTPUT_BLOCK_COUNT(layer);
        end if;
        return 1;
    end function;
begin
    dut : entity work.inference_controller
        port map (
            clk                      => clk,
            rst                      => rst,
            start                    => start,
            activation_valid         => activation_valid,
            weights_valid            => weights_valid,
            layer_output_valid       => layer_output_valid,
            mac_busy                 => mac_busy,
            busy                     => busy,
            done                     => done,
            result_valid             => result_valid,
            conv_enable              => conv_enable,
            layer_index              => selected_layer,
            input_index              => input_index,
            output_block_index       => selected_block,
            tap_index                => tap_index,
            activation_read_enable   => act_read_enable,
            activation_write_enable  => activation_write_enable,
            weight_read_enable       => weight_enable,
            mac_load_bias            => mac_load_bias,
            mac_valid                => mac_enable,
            layer_output_input_valid => layer_block_valid
        );

    layer_index              <= selected_layer;
    output_block_index       <= selected_block;
    activation_read_enable   <= act_read_enable;
    weight_read_enable       <= weight_enable;
    mac_valid                <= mac_enable;
    layer_output_input_valid <= layer_block_valid;
    mac_busy                 <= mac_input_valid or mac_product_valid;

    -- Timing model for act_ram + conv_input/dense_input, weight_rom and mac_array.
    process (clk, rst)
    begin
        if rst = '1' then
            activation_ram_valid <= '0';
            activation_valid     <= '0';
            weights_valid        <= '0';
            layer_output_valid   <= '0';
            mac_input_valid      <= '0';
            mac_product_valid    <= '0';
        elsif rising_edge(clk) then
            activation_ram_valid <= act_read_enable;
            activation_valid     <= activation_ram_valid;
            weights_valid        <= weight_enable;
            mac_input_valid      <= mac_enable;
            mac_product_valid    <= mac_input_valid;

            layer_output_valid <= '0';
            if layer_block_valid = '1'
                and selected_block = block_count(selected_layer) - 1 then
                layer_output_valid <= '1';
            end if;
        end if;
    end process;
end architecture;
