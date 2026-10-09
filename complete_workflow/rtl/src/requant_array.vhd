library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity requant_array is
    port (
        layer_index     : in integer range 0 to N_LAYERS - 1;
        input_valid     : in std_logic;
        accum_data      : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);

        output_valid    : out std_logic;
        activation_data : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of requant_array is
begin
    output_valid <= input_valid;

    process (all)
    begin
        activation_data <= (others => (others => '0'));
        if input_valid = '1' then
            for lane in 0 to MAX_MAC_UNITS - 1 loop
                case layer_index is
                    when 0 =>
                        activation_data(lane) <= resize(
                            requantize_relu(accum_data(lane), 4, ACT_W(0)), ACT_W_MAX
                        );
                    when 1 =>
                        activation_data(lane) <= resize(
                            requantize_relu(accum_data(lane), 3, ACT_W(1)), ACT_W_MAX
                        );
                    when 2 =>
                        activation_data(lane) <= resize(
                            requantize_relu(accum_data(lane), 7, ACT_W(2)), ACT_W_MAX
                        );
                    when 3 =>
                        activation_data(lane) <= resize(
                            requantize_relu(accum_data(lane), 3, ACT_W(3)), ACT_W_MAX
                        );
                    when 4 =>
                        activation_data(lane) <= resize(
                            requantize_relu(accum_data(lane), 5, ACT_W(4)), ACT_W_MAX
                        );
                    when 5 =>
                        activation_data(lane) <= resize(
                            requantize_signed(accum_data(lane), 5, ACT_W(5)), ACT_W_MAX
                        );
                end case;
            end loop;
        end if;
    end process;
end architecture;
