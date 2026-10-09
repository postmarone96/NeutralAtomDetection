library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.preprocess_pkg.all;

entity detector_core_wrapper is
    port (
        clk                : in std_logic;
        rst                : in std_logic;
        frame_write_enable : in std_logic;
        frame_write_addr   : in integer range 0 to FRAME_PIXELS - 1;
        frame_write_data   : in unsigned(RAW_W - 1 downto 0);
        start              : in std_logic;
        busy               : out std_logic;
        result_valid       : out std_logic;
        result_site_index  : out integer range 0 to SITE_COUNT - 1;
        result_logits_flat : out std_logic_vector(CLASSES * ACT_W_MAX - 1 downto 0);
        result_decision    : out std_logic;
        preprocess_cycles  : out unsigned(31 downto 0);
        cnn_cycles         : out unsigned(31 downto 0);
        pipeline_cycles    : out unsigned(31 downto 0);
        done               : out std_logic
    );
end entity;

architecture rtl of detector_core_wrapper is
    signal result_logits : signed_vector(0 to CLASSES - 1)(ACT_W_MAX - 1 downto 0);
begin
    flatten_logits : for lane in 0 to CLASSES - 1 generate
        result_logits_flat((lane + 1) * ACT_W_MAX - 1 downto lane * ACT_W_MAX)
            <= std_logic_vector(result_logits(lane));
    end generate;

    dut : entity work.detector_core
        generic map (
            WEIGHT_FILE => "weights.mem",
            BIAS_FILE   => "biases.mem"
        )
        port map (
            clk                => clk,
            rst                => rst,
            frame_write_enable => frame_write_enable,
            frame_write_addr   => frame_write_addr,
            frame_write_data   => frame_write_data,
            start              => start,
            busy               => busy,
            result_valid       => result_valid,
            result_site_index  => result_site_index,
            result_logits      => result_logits,
            result_decision    => result_decision,
            preprocess_cycles  => preprocess_cycles,
            cnn_cycles         => cnn_cycles,
            pipeline_cycles    => pipeline_cycles,
            done               => done
        );
end architecture;
