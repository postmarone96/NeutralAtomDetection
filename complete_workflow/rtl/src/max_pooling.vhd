library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity max_pooling is
    port (
        conv_index     : in integer range 0 to N_CONV - 1;
        activation_in  : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        activation_out : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of max_pooling is
begin
    process (all)
    begin
        activation_out <= (others => (others => '0'));

        if conv_index = 0 then
            for output_row in 0 to 4 loop
                for output_column in 0 to 4 loop
                    activation_out(output_row * 5 + output_column) <=
                        maximum_of_four(
                            activation_in(output_row * 20 + output_column * 2),
                            activation_in(output_row * 20 + output_column * 2 + 1),
                            activation_in(output_row * 20 + output_column * 2 + 10),
                            activation_in(output_row * 20 + output_column * 2 + 11)
                        );
                end loop;
            end loop;

        elsif conv_index = 1 then
            for channel in 0 to 3 loop
                for output_row in 0 to 1 loop
                    for output_column in 0 to 1 loop
                        activation_out(channel * 4 + output_row * 2 + output_column) <=
                            maximum_of_four(
                                activation_in(channel * 25 + output_row * 10 + output_column * 2),
                                activation_in(channel * 25 + output_row * 10 + output_column * 2 + 1),
                                activation_in(channel * 25 + output_row * 10 + output_column * 2 + 5),
                                activation_in(channel * 25 + output_row * 10 + output_column * 2 + 6),
                                round_half => true
                            );
                    end loop;
                end loop;
            end loop;

        else
            activation_out <= activation_in;
        end if;
    end process;
end architecture;
