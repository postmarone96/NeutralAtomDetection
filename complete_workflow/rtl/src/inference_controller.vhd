library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity inference_controller is
    port (
        clk                      : in std_logic;
        rst                      : in std_logic;
        start                    : in std_logic;

        activation_valid         : in std_logic;
        weights_valid            : in std_logic;
        layer_output_valid       : in std_logic;
        mac_busy                 : in std_logic;

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

architecture rtl of inference_controller is
    type controller_state is (
        idle_state,
        activation_read_state,
        weight_read_state,
        mac_state,
        emit_block_state,
        wait_layer_output_state
    );

    signal state_reg        : controller_state := idle_state;
    signal layer_reg        : integer range 0 to N_LAYERS - 1 := 0;
    signal input_reg        : integer range 0 to MAX_LAYER_INPUT_COUNT - 1 := 0;
    signal output_block_reg : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1 := 0;
    signal tap_reg          : integer range 0 to KERNEL_TAP_COUNT - 1 := 0;
    signal busy_reg         : std_logic := '0';
    signal done_reg         : std_logic := '0';
    signal result_valid_reg : std_logic := '0';

    function layer_block_count(selected_layer : integer) return integer is
    begin
        if selected_layer < N_CONV then
            return CONV_OUTPUT_BLOCK_COUNT(selected_layer);
        end if;
        return 1;
    end function;

    function layer_tap_count(selected_layer : integer) return integer is
    begin
        if selected_layer < N_CONV then
            return KERNEL_TAP_COUNT;
        end if;
        return 1;
    end function;
begin
    process (clk, rst)
    begin
        if rst = '1' then
            state_reg         <= idle_state;
            layer_reg         <= 0;
            input_reg         <= 0;
            output_block_reg  <= 0;
            tap_reg           <= 0;
            busy_reg          <= '0';
            done_reg          <= '0';
            result_valid_reg  <= '0';
        elsif rising_edge(clk) then
            done_reg         <= '0';
            result_valid_reg <= '0';

            case state_reg is
                when idle_state =>
                    busy_reg <= '0';
                    if start = '1' then
                        layer_reg         <= 0;
                        input_reg         <= 0;
                        output_block_reg  <= 0;
                        tap_reg           <= 0;
                        busy_reg          <= '1';
                        state_reg         <= activation_read_state;
                    end if;

                when activation_read_state =>
                    state_reg <= weight_read_state;

                when weight_read_state =>
                    state_reg <= mac_state;

                when mac_state =>
                    if activation_valid = '1' and weights_valid = '1' then
                        if tap_reg < layer_tap_count(layer_reg) - 1 then
                            tap_reg   <= tap_reg + 1;
                            state_reg <= activation_read_state;
                        elsif input_reg < LAYER_INPUT_COUNT(layer_reg) - 1 then
                            tap_reg   <= 0;
                            input_reg <= input_reg + 1;
                            state_reg <= activation_read_state;
                        else
                            tap_reg   <= 0;
                            input_reg <= 0;
                            state_reg <= emit_block_state;
                        end if;
                    end if;

                when emit_block_state =>
                    if mac_busy = '0' then
                        if output_block_reg < layer_block_count(layer_reg) - 1 then
                            output_block_reg <= output_block_reg + 1;
                            state_reg        <= activation_read_state;
                        else
                            state_reg <= wait_layer_output_state;
                        end if;
                    end if;

                when wait_layer_output_state =>
                    if layer_output_valid = '1' then
                        if layer_reg = N_LAYERS - 1 then
                            busy_reg         <= '0';
                            done_reg         <= '1';
                            result_valid_reg <= '1';
                            state_reg         <= idle_state;
                        else
                            layer_reg         <= layer_reg + 1;
                            input_reg         <= 0;
                            output_block_reg  <= 0;
                            tap_reg           <= 0;
                            state_reg         <= activation_read_state;
                        end if;
                    end if;
            end case;
        end if;
    end process;

    process (all)
    begin
        activation_read_enable   <= '0';
        activation_write_enable  <= '0';
        weight_read_enable       <= '0';
        mac_load_bias            <= '0';
        mac_valid                <= '0';
        layer_output_input_valid <= '0';

        case state_reg is
            when activation_read_state =>
                activation_read_enable <= '1';

            when weight_read_state =>
                weight_read_enable <= '1';

            when mac_state =>
                if activation_valid = '1' and weights_valid = '1' then
                    mac_valid <= '1';
                    if input_reg = 0 and tap_reg = 0 then
                        mac_load_bias <= '1';
                    end if;
                end if;

            when emit_block_state =>
                if mac_busy = '0' then
                    layer_output_input_valid <= '1';
                end if;

            when wait_layer_output_state =>
                if layer_output_valid = '1' and layer_reg < N_LAYERS - 1 then
                    activation_write_enable <= '1';
                end if;

            when others =>
                null;
        end case;
    end process;

    conv_enable        <= '1' when layer_reg < N_CONV else '0';
    layer_index        <= layer_reg;
    input_index        <= input_reg;
    output_block_index <= output_block_reg;
    tap_index          <= tap_reg;

    busy         <= busy_reg;
    done         <= done_reg;
    result_valid <= result_valid_reg;
end architecture;
