library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.activation_map_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity weights_handler is
  port (
    clk : in std_logic;

    conv_enable : in std_logic; --  '1' if conv, '0' if dense
    layer_index : in integer range 0 to N_LAYERS - 1; -- layer and conv index
    -- also input_channel for conv layers and input_layer for dense layers
    input_index : in integer range 0 to MAX_LAYER_INPUT_COUNT - 1;
    -- Conv0 block = 1 input channel × 9 taps = 9 rows
    -- Conv1 block = 4 input channels × 9 taps = 36 rows
    -- Conv2 block = 8 input channels × 9 taps = 72 rows
    output_block_index : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
    tap_index          : in integer range 0 to KERNEL_TAP_COUNT - 1;

    -- ROM
    read_enable     : in std_logic;
    rom_read_enable : out std_logic;
    read_valid     : in std_logic;
    rom_row        : out integer range 0 to WEIGHT_ROM_DEPTH - 1;
    rom_row_data   : in signed(WEIGHT_ROM_BITS - 1 downto 0);

    -- outputs
    weights       : out signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
    weights_valid : out std_logic
  );
end entity;

architecture rtl of weights_handler is
  signal selected_conv_enable        : std_logic                                     := '0';
  signal selected_layer_index        : integer range 0 to N_LAYERS - 1               := 0;
  signal selected_output_block_index : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1 := 0;

  signal unpacked_slots : signed_vector(0 to WEIGHT_SLOTS_PER_WORD - 1)(WEIGHT_W_MAX - 1 downto 0);
  signal conv0_weights  : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
  signal conv1_weights  : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
  signal conv2_weights  : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
  signal dense8_weights : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
  signal dense2_weights : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
begin
  -- remember which operation requested the synchronous rom row.
  process (clk)
  begin
    if rising_edge(clk) then
      if read_enable = '1' then
        selected_conv_enable        <= conv_enable;
        selected_layer_index        <= layer_index;
        selected_output_block_index <= output_block_index;
      end if;
    end if;
  end process;

  -- Fixed row layout for this six-layer model.  Keeping the layer cases
  -- explicit prevents general integer multipliers and package-array muxes.
  process (all)
  begin
    rom_row <= 0;

    if conv_enable = '1' then
      case layer_index is
        when 0 =>
          case output_block_index is
            when 0      => rom_row <=  0 + tap_index;
            when 1      => rom_row <=  9 + tap_index;
            when 2      => rom_row <= 18 + tap_index;
            when others => rom_row <= 27 + tap_index;
          end case;

        when 1 =>
          case output_block_index is
            when 0 =>
              case input_index is
                when 0      => rom_row <= 36 + tap_index;
                when 1      => rom_row <= 45 + tap_index;
                when 2      => rom_row <= 54 + tap_index;
                when others => rom_row <= 63 + tap_index;
              end case;
            when others =>
              case input_index is
                when 0      => rom_row <= 72 + tap_index;
                when 1      => rom_row <= 81 + tap_index;
                when 2      => rom_row <= 90 + tap_index;
                when others => rom_row <= 99 + tap_index;
              end case;
          end case;

        when 2 =>
          case input_index is
            when 0      => rom_row <= 108 + tap_index;
            when 1      => rom_row <= 117 + tap_index;
            when 2      => rom_row <= 126 + tap_index;
            when 3      => rom_row <= 135 + tap_index;
            when 4      => rom_row <= 144 + tap_index;
            when 5      => rom_row <= 153 + tap_index;
            when 6      => rom_row <= 162 + tap_index;
            when others => rom_row <= 171 + tap_index;
          end case;

        when others =>
          rom_row <= 0;
      end case;
    else
      case layer_index is
        when 0      => rom_row <= input_index;
        when 1      => rom_row <= 36  + input_index;
        when 2      => rom_row <= 108 + input_index;
        when 3      => rom_row <= 180 + input_index;
        when 4      => rom_row <= 244 + input_index;
        when others => rom_row <= 260 + input_index;
      end case;
    end if;
  end process;

  rom_read_enable <= read_enable;

  -- Every slice bound below is static, so the ROM word is only wiring.
  unpack_rom_slots : for slot in 0 to WEIGHT_SLOTS_PER_WORD - 1 generate
    unpacked_slots(slot) <= resize(
      rom_row_data((slot + 1) * WEIGHT_BITS - 1 downto slot * WEIGHT_BITS),
      WEIGHT_W_MAX
    );
  end generate;

  -- Conv0: one output channel per block, broadcast one slot to 100 positions.
  map_conv0 : for lane in 0 to 99 generate
    conv0_weights(lane) <= unpacked_slots(0);
  end generate;

  -- Conv1: four output channels per block, each covering 25 positions.
  map_conv1 : for slot in 0 to 3 generate
    map_conv1_positions : for position in 0 to 24 generate
      conv1_weights(slot * 25 + position) <= unpacked_slots(slot);
    end generate;
  end generate;

  -- Conv2: sixteen output channels, each covering four positions.
  map_conv2 : for slot in 0 to 15 generate
    map_conv2_positions : for position in 0 to 3 generate
      conv2_weights(slot * 4 + position) <= unpacked_slots(slot);
    end generate;
  end generate;
  clear_conv2_unused : for lane in 64 to MAX_MAC_UNITS - 1 generate
    conv2_weights(lane) <= (others => '0');
  end generate;

  -- Dense0/Dense1 use eight lanes; Dense2 uses two lanes.
  map_dense8 : for lane in 0 to 7 generate
    dense8_weights(lane) <= unpacked_slots(lane);
  end generate;
  clear_dense8_unused : for lane in 8 to MAX_MAC_UNITS - 1 generate
    dense8_weights(lane) <= (others => '0');
  end generate;

  map_dense2 : for lane in 0 to 1 generate
    dense2_weights(lane) <= unpacked_slots(lane);
  end generate;
  clear_dense2_unused : for lane in 2 to MAX_MAC_UNITS - 1 generate
    dense2_weights(lane) <= (others => '0');
  end generate;

  -- Select the fixed lane map that belongs to the synchronous ROM request.
  process (all)
  begin
    weights <= (others => (others => '0'));

    if read_valid = '1' then
      if selected_conv_enable = '1' then
        case selected_layer_index is
          when 0      => weights <= conv0_weights;
          when 1      => weights <= conv1_weights;
          when 2      => weights <= conv2_weights;
          when others => null;
        end case;
      else
        case selected_layer_index is
          when 3 | 4  => weights <= dense8_weights;
          when 5      => weights <= dense2_weights;
          when others => null;
        end case;
      end if;
    end if;
  end process;

  weights_valid <= read_valid;
end architecture;
