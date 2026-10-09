library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity inference_core_wrapper is
    port (
        clk                : in std_logic;
        rst                : in std_logic;
        load_input         : in std_logic;
        start              : in std_logic;
        input_data_flat    : in std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0);
        neighbor_data_flat : in std_logic_vector(NEIGHBORS * NB_W - 1 downto 0);
        busy               : out std_logic;
        result_valid       : out std_logic;
        result_data_flat   : out std_logic_vector(MAX_MAC_UNITS * ACT_W_MAX - 1 downto 0);
        decision           : out std_logic
    );
end entity;

architecture rtl of inference_core_wrapper is
    signal input_lanes    : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal neighbor_lanes : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
    signal result_lanes   : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
begin
    activation_lanes : for lane in 0 to MAX_MAC_UNITS - 1 generate
        input_lanes(lane) <= signed(
            input_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
        );
        result_data_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(result_lanes(lane));
    end generate;

    neighbor_values : for lane in 0 to NEIGHBORS - 1 generate
        neighbor_lanes(lane) <= signed(
            neighbor_data_flat((lane + 1) * NB_W - 1 downto lane * NB_W)
        );
    end generate;

    dut : entity work.inference_core
        generic map (
            WEIGHT_FILE => "weights.mem",
            BIAS_FILE   => "biases.mem"
        )
        port map (
            clk           => clk,
            rst           => rst,
            load_input    => load_input,
            start         => start,
            input_data    => input_lanes,
            neighbor_data => neighbor_lanes,
            busy          => busy,
            result_valid  => result_valid,
            result_data   => result_lanes,
            decision      => decision
        );
end architecture;
