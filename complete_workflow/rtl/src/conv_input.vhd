library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.activation_map_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity conv_input is
  port (
    clk               : in std_logic;
    rst               : in std_logic;

    read_valid        : in std_logic;
    read_data         : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

    input_channel     : in integer range 0 to MAX_CHANNEL - 1;
    conv_index        : in integer range 0 to N_CONV - 1;
    tap_index         : in integer range 0 to KERNEL_TAP_COUNT - 1;
    
    conv_input_valid  : out std_logic;
    conv_input        : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
  );
end entity;

architecture rtl of conv_input is
    signal internal_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                           := (others => (others => '0'));
begin
  process (clk, rst)
    variable next_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    variable channel_25  : signed_vector(0 to 24)(ACT_W_MAX - 1 downto 0);
    variable channel_4   : signed_vector(0 to 3)(ACT_W_MAX - 1 downto 0);
  begin
    if rst = '1' then
      conv_input_valid <= '0';
    elsif rising_edge(clk) then
      conv_input_valid <= '0';

      if read_valid = '1' then
        next_buffer := (others => (others => '0'));
        channel_25  := (others => (others => '0'));
        channel_4   := (others => (others => '0'));

        -- Each case below has fixed loop bounds and fixed source indices.  The
        -- synthesis tool therefore builds wiring and small muxes instead of a
        -- general spatial-map ROM, multiplier, and 100-to-1 lane selector.
        case conv_index is
          when 0 =>
            assert input_channel = 0
              report "input channel is invalid for convolution 0"
              severity failure;

            case tap_index is
              when 0 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index > 0 and column_index > 0 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index - 1) * 10 + column_index - 1);
                    end if;
                  end loop;
                end loop;

              when 1 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index > 0 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index - 1) * 10 + column_index);
                    end if;
                  end loop;
                end loop;

              when 2 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index > 0 and column_index < 9 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index - 1) * 10 + column_index + 1);
                    end if;
                  end loop;
                end loop;

              when 3 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if column_index > 0 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data(row_index * 10 + column_index - 1);
                    end if;
                  end loop;
                end loop;

              when 4 =>
                for position_index in 0 to 99 loop
                  next_buffer(position_index) := read_data(position_index);
                end loop;

              when 5 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if column_index < 9 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data(row_index * 10 + column_index + 1);
                    end if;
                  end loop;
                end loop;

              when 6 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index < 9 and column_index > 0 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index + 1) * 10 + column_index - 1);
                    end if;
                  end loop;
                end loop;

              when 7 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index < 9 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index + 1) * 10 + column_index);
                    end if;
                  end loop;
                end loop;

              when 8 =>
                for row_index in 0 to 9 loop
                  for column_index in 0 to 9 loop
                    if row_index < 9 and column_index < 9 then
                      next_buffer(row_index * 10 + column_index) :=
                        read_data((row_index + 1) * 10 + column_index + 1);
                    end if;
                  end loop;
                end loop;
            end case;

          when 1 =>
            assert input_channel < 4
              report "input channel is invalid for convolution 1"
              severity failure;

            case input_channel is
              when 0      => channel_25 := read_data(0 to 24);
              when 1      => channel_25 := read_data(25 to 49);
              when 2      => channel_25 := read_data(50 to 74);
              when 3      => channel_25 := read_data(75 to 99);
              when others => null;
            end case;

            case tap_index is
              when 0 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index > 0 and column_index > 0 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index - 1) * 5 + column_index - 1);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 1 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index > 0 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index - 1) * 5 + column_index);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 2 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index > 0 and column_index < 4 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index - 1) * 5 + column_index + 1);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 3 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if column_index > 0 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25(row_index * 5 + column_index - 1);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 4 =>
                for output_channel in 0 to 3 loop
                  for position_index in 0 to 24 loop
                    next_buffer(output_channel * 25 + position_index) := channel_25(position_index);
                  end loop;
                end loop;

              when 5 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if column_index < 4 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25(row_index * 5 + column_index + 1);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 6 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index < 4 and column_index > 0 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index + 1) * 5 + column_index - 1);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 7 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index < 4 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index + 1) * 5 + column_index);
                      end if;
                    end loop;
                  end loop;
                end loop;

              when 8 =>
                for output_channel in 0 to 3 loop
                  for row_index in 0 to 4 loop
                    for column_index in 0 to 4 loop
                      if row_index < 4 and column_index < 4 then
                        next_buffer(output_channel * 25 + row_index * 5 + column_index) :=
                          channel_25((row_index + 1) * 5 + column_index + 1);
                      end if;
                    end loop;
                  end loop;
                end loop;
            end case;

          when 2 =>
            assert input_channel < 8
              report "input channel is invalid for convolution 2"
              severity failure;

            case input_channel is
              when 0      => channel_4 := read_data(0 to 3);
              when 1      => channel_4 := read_data(4 to 7);
              when 2      => channel_4 := read_data(8 to 11);
              when 3      => channel_4 := read_data(12 to 15);
              when 4      => channel_4 := read_data(16 to 19);
              when 5      => channel_4 := read_data(20 to 23);
              when 6      => channel_4 := read_data(24 to 27);
              when 7      => channel_4 := read_data(28 to 31);
              when others => null;
            end case;

            case tap_index is
              when 0 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4 + 3) := channel_4(0);
                end loop;

              when 1 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4 + 2) := channel_4(0);
                  next_buffer(output_channel * 4 + 3) := channel_4(1);
                end loop;

              when 2 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4 + 2) := channel_4(1);
                end loop;

              when 3 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4 + 1) := channel_4(0);
                  next_buffer(output_channel * 4 + 3) := channel_4(2);
                end loop;

              when 4 =>
                for output_channel in 0 to 15 loop
                  for position_index in 0 to 3 loop
                    next_buffer(output_channel * 4 + position_index) := channel_4(position_index);
                  end loop;
                end loop;

              when 5 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4)     := channel_4(1);
                  next_buffer(output_channel * 4 + 2) := channel_4(3);
                end loop;

              when 6 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4 + 1) := channel_4(2);
                end loop;

              when 7 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4)     := channel_4(2);
                  next_buffer(output_channel * 4 + 1) := channel_4(3);
                end loop;

              when 8 =>
                for output_channel in 0 to 15 loop
                  next_buffer(output_channel * 4) := channel_4(3);
                end loop;
            end case;
        end case;

        internal_buffer <= next_buffer;
        conv_input_valid <= '1';
      end if;
    end if;
  end process;

  conv_input <= internal_buffer;
end architecture;
