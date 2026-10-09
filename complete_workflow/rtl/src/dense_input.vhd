library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use ieee.fixed_float_types.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity dense_input is
    port (
        clk               : in std_logic;
        rst               : in std_logic;
        read_valid        : in std_logic;
        read_data         : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        neighbor_data     : in signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);

        layer_index       : in integer range 0 to N_LAYERS - 1;
        input_index       : in integer range 0 to MAX_LAYER_INPUT_COUNT - 1;

        dense_input_valid : out std_logic;
        dense_input_data  : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of dense_input is
    signal internal_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                           := (others => (others => '0'));

    function cast_neighbor(value : signed(NB_W - 1 downto 0)) return signed is
        variable input_value  : sfixed(NB_I - 1 downto NB_I - NB_W);
        variable output_value : sfixed(CAT_I - 1 downto CAT_I - CAT_W);
    begin
        input_value := to_sfixed(std_logic_vector(value), input_value'high, input_value'low);
        output_value := resize(
            input_value,
            output_value'high,
            output_value'low,
            fixed_saturate,
            fixed_round
        );
        return signed(to_slv(output_value));
    end function;
begin
    process (clk, rst)
        variable selected_value : signed(ACT_W_MAX - 1 downto 0);
    begin
        if rst = '1' then
            internal_buffer  <= (others => (others => '0'));
            dense_input_valid <= '0';
        elsif rising_edge(clk) then
            dense_input_valid <= '0';

            if read_valid = '1' then
                internal_buffer <= (others => (others => '0'));

                case layer_index is
                    when 3 =>
                        case input_index is
                            when 0  => selected_value := read_data(0);
                            when 1  => selected_value := read_data(1);
                            when 2  => selected_value := read_data(2);
                            when 3  => selected_value := read_data(3);
                            when 4  => selected_value := read_data(4);
                            when 5  => selected_value := read_data(5);
                            when 6  => selected_value := read_data(6);
                            when 7  => selected_value := read_data(7);
                            when 8  => selected_value := read_data(8);
                            when 9  => selected_value := read_data(9);
                            when 10 => selected_value := read_data(10);
                            when 11 => selected_value := read_data(11);
                            when 12 => selected_value := read_data(12);
                            when 13 => selected_value := read_data(13);
                            when 14 => selected_value := read_data(14);
                            when 15 => selected_value := read_data(15);
                            when 16 => selected_value := read_data(16);
                            when 17 => selected_value := read_data(17);
                            when 18 => selected_value := read_data(18);
                            when 19 => selected_value := read_data(19);
                            when 20 => selected_value := read_data(20);
                            when 21 => selected_value := read_data(21);
                            when 22 => selected_value := read_data(22);
                            when 23 => selected_value := read_data(23);
                            when 24 => selected_value := read_data(24);
                            when 25 => selected_value := read_data(25);
                            when 26 => selected_value := read_data(26);
                            when 27 => selected_value := read_data(27);
                            when 28 => selected_value := read_data(28);
                            when 29 => selected_value := read_data(29);
                            when 30 => selected_value := read_data(30);
                            when 31 => selected_value := read_data(31);
                            when 32 => selected_value := read_data(32);
                            when 33 => selected_value := read_data(33);
                            when 34 => selected_value := read_data(34);
                            when 35 => selected_value := read_data(35);
                            when 36 => selected_value := read_data(36);
                            when 37 => selected_value := read_data(37);
                            when 38 => selected_value := read_data(38);
                            when 39 => selected_value := read_data(39);
                            when 40 => selected_value := read_data(40);
                            when 41 => selected_value := read_data(41);
                            when 42 => selected_value := read_data(42);
                            when 43 => selected_value := read_data(43);
                            when 44 => selected_value := read_data(44);
                            when 45 => selected_value := read_data(45);
                            when 46 => selected_value := read_data(46);
                            when 47 => selected_value := read_data(47);
                            when 48 => selected_value := read_data(48);
                            when 49 => selected_value := read_data(49);
                            when 50 => selected_value := read_data(50);
                            when 51 => selected_value := read_data(51);
                            when 52 => selected_value := read_data(52);
                            when 53 => selected_value := read_data(53);
                            when 54 => selected_value := read_data(54);
                            when 55 => selected_value := read_data(55);
                            when 56 => selected_value := read_data(56);
                            when 57 => selected_value := read_data(57);
                            when 58 => selected_value := read_data(58);
                            when 59 => selected_value := read_data(59);
                            when 60 => selected_value := read_data(60);
                            when 61 => selected_value := read_data(61);
                            when 62 => selected_value := read_data(62);
                            when 63 => selected_value := read_data(63);
                            when others =>
                                assert false
                                    report "input index is invalid for dense layer 0"
                                    severity failure;
                                selected_value := (others => '0');
                        end case;
                        for lane in 0 to 7 loop
                            internal_buffer(lane) <= selected_value;
                        end loop;

                    -- Dense 1
                    when 4 =>
                        case input_index is
                            when 0  => selected_value := read_data(0);
                            when 1  => selected_value := read_data(1);
                            when 2  => selected_value := read_data(2);
                            when 3  => selected_value := read_data(3);
                            when 4  => selected_value := read_data(4);
                            when 5  => selected_value := read_data(5);
                            when 6  => selected_value := read_data(6);
                            when 7  => selected_value := read_data(7);
                            when 8  => selected_value := resize(cast_neighbor(neighbor_data(0)), ACT_W_MAX);
                            when 9  => selected_value := resize(cast_neighbor(neighbor_data(1)), ACT_W_MAX);
                            when 10 => selected_value := resize(cast_neighbor(neighbor_data(2)), ACT_W_MAX);
                            when 11 => selected_value := resize(cast_neighbor(neighbor_data(3)), ACT_W_MAX);
                            when 12 => selected_value := resize(cast_neighbor(neighbor_data(4)), ACT_W_MAX);
                            when 13 => selected_value := resize(cast_neighbor(neighbor_data(5)), ACT_W_MAX);
                            when 14 => selected_value := resize(cast_neighbor(neighbor_data(6)), ACT_W_MAX);
                            when 15 => selected_value := resize(cast_neighbor(neighbor_data(7)), ACT_W_MAX);
                            when others =>
                                assert false
                                    report "input index is invalid for dense layer 1"
                                    severity failure;
                                selected_value := (others => '0');
                        end case;
                        for lane in 0 to 7 loop
                            internal_buffer(lane) <= selected_value;
                        end loop;

                    -- Dense 2
                    when 5 =>
                        case input_index is
                            when 0 => selected_value := read_data(0);
                            when 1 => selected_value := read_data(1);
                            when 2 => selected_value := read_data(2);
                            when 3 => selected_value := read_data(3);
                            when 4 => selected_value := read_data(4);
                            when 5 => selected_value := read_data(5);
                            when 6 => selected_value := read_data(6);
                            when 7 => selected_value := read_data(7);
                            when others =>
                                assert false
                                    report "input index is invalid for dense layer 2"
                                    severity failure;
                                selected_value := (others => '0');
                        end case;
                        for lane in 0 to 1 loop
                            internal_buffer(lane) <= selected_value;
                        end loop;

                    when others =>
                        assert false
                            report "dense input received a convolution layer index"
                            severity failure;
                end case;

                dense_input_valid <= '1';
            end if;
        end if;
    end process;

    dense_input_data <= internal_buffer;
end architecture;
