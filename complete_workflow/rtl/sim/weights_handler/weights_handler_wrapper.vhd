library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.activation_map_pkg.all;
use work.weights_pkg.all;

entity weights_handler_wrapper is
  port (
    clk                : in std_logic;
    rst                : in std_logic;
    read_enable        : in std_logic;
    layer_index        : in integer range 0 to N_LAYERS - 1;
    output_block_index : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
    input_index        : in integer range 0 to MAX_LAYER_INPUT_COUNT - 1;
    tap_index          : in integer range 0 to KERNEL_TAP_COUNT - 1;
    weights_valid      : out std_logic;
    mac_weights_flat   : out std_logic_vector(MAX_MAC_UNITS * WEIGHT_W_MAX - 1 downto 0)
  );
end entity;

architecture rtl of weights_handler_wrapper is
  signal conv_enable     : std_logic;
  signal rom_read_enable : std_logic;
  signal rom_read_valid  : std_logic;
  signal rom_row         : integer range 0 to WEIGHT_ROM_DEPTH - 1;
  signal rom_row_data    : signed(WEIGHT_ROM_BITS - 1 downto 0);
  signal mac_weights     : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
begin
  conv_enable <= '1' when layer_index < N_CONV else '0';

  dut : entity work.weights_handler
    port map (
      clk                => clk,
      conv_enable        => conv_enable,
      layer_index        => layer_index,
      input_index        => input_index,
      output_block_index => output_block_index,
      tap_index          => tap_index,
      read_enable        => read_enable,
      rom_read_enable    => rom_read_enable,
      read_valid         => rom_read_valid,
      rom_row            => rom_row,
      rom_row_data       => rom_row_data,
      weights            => mac_weights,
      weights_valid      => weights_valid
    );

  memory : entity work.weight_rom
    generic map (
      ROM_DEPTH   => WEIGHT_ROM_DEPTH,
      WORD_LENGTH => WEIGHT_ROM_BITS,
      FILE_NAME   => "weights.mem"
    )
    port map (
      clk                => clk,
      read_enable        => rom_read_enable,
      addr               => rom_row,
      read_valid         => rom_read_valid,
      data_out           => rom_row_data
    );

  flatten : for lane in 0 to MAX_MAC_UNITS - 1 generate
    mac_weights_flat((lane + 1) * WEIGHT_W_MAX - 1 downto lane * WEIGHT_W_MAX)
      <= std_logic_vector(mac_weights(lane));
  end generate;
end architecture;
