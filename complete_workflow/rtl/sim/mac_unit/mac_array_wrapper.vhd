library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity mac_array_wrapper is
    generic(
        ACT_W_MAX:      positive;
        WEIGHT_W_MAX:   positive;
        ACC_W_MAX:      positive;
        MAX_MAC_UNITS:  positive;
        MAC_UNITS:      positive
    );

    port(
        clk:             in  std_logic;
        rst:             in  std_logic;
        load_bias:       in  std_logic;
        valid:           in  std_logic;

        activation_flat: in  std_logic_vector(MAC_UNITS * ACT_W_MAX - 1 downto 0);
        weight_flat:     in  std_logic_vector(MAC_UNITS * WEIGHT_W_MAX - 1 downto 0);
        bias_flat:       in  std_logic_vector(MAX_MAC_UNITS * ACC_W_MAX - 1 downto 0);
        busy:            out std_logic;
        accum_flat:      out std_logic_vector(MAX_MAC_UNITS * ACC_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of mac_array_wrapper is
    signal activation_lanes: signed_vector(0 to MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal weight_lanes:     signed_vector(0 to MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
    signal bias_lanes:       signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
    signal accum_lanes:      signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
begin
    unpack_active_lanes: for i in 0 to MAC_UNITS - 1 generate
        activation_lanes(i) <= signed(
            activation_flat((i + 1) * ACT_W_MAX - 1 downto i * ACT_W_MAX)
        );
        weight_lanes(i) <= signed(
            weight_flat((i + 1) * WEIGHT_W_MAX - 1 downto i * WEIGHT_W_MAX)
        );
    end generate unpack_active_lanes;

    unpack_bias_lanes: for i in 0 to MAX_MAC_UNITS - 1 generate
        bias_lanes(i) <= signed(
            bias_flat((i + 1) * ACC_W_MAX - 1 downto i * ACC_W_MAX)
        );
    end generate unpack_bias_lanes;

    pack_accumulators: for i in 0 to MAX_MAC_UNITS - 1 generate
        accum_flat((i + 1) * ACC_W_MAX - 1 downto i * ACC_W_MAX)
            <= std_logic_vector(accum_lanes(i));
    end generate pack_accumulators;

    dut: entity work.mac_array
        generic map(
            ACT_W_MAX     => ACT_W_MAX,
            WEIGHT_W_MAX  => WEIGHT_W_MAX,
            ACC_W_MAX     => ACC_W_MAX,
            MAX_MAC_UNITS => MAX_MAC_UNITS,
            MAC_UNITS     => MAC_UNITS
        )
        port map(
            clk        => clk,
            rst        => rst,
            load_bias  => load_bias,
            valid      => valid,
            activation => activation_lanes,
            weight     => weight_lanes,
            bias       => bias_lanes,
            busy       => busy,
            accum      => accum_lanes
        );
end architecture;
