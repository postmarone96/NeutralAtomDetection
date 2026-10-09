library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity layer_output is
    port (
        clk                : in std_logic;
        rst                : in std_logic;
        input_valid        : in std_logic;
        layer_index        : in integer range 0 to N_LAYERS - 1;
        output_block_index : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
        input_data         : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

        output_valid       : out std_logic;
        output_data        : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of layer_output is
    signal output_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                         := (others => (others => '0'));
begin
    process (clk, rst)
        variable next_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    begin
        if rst = '1' then
            output_buffer <= (others => (others => '0'));
            output_data   <= (others => (others => '0'));
            output_valid  <= '0';
        elsif rising_edge(clk) then
            output_valid <= '0';

            if input_valid = '1' then
                case layer_index is
                    -- Conv 0: four blocks of 25 pooled activations.
                    when 0 =>
                        case output_block_index is
                            when 0 =>
                                next_buffer := (others => (others => '0'));
                                for lane in 0 to 24 loop
                                    next_buffer(lane) := input_data(lane);
                                end loop;
                            when 1 =>
                                next_buffer := output_buffer;
                                for lane in 0 to 24 loop
                                    next_buffer(25 + lane) := input_data(lane);
                                end loop;
                            when 2 =>
                                next_buffer := output_buffer;
                                for lane in 0 to 24 loop
                                    next_buffer(50 + lane) := input_data(lane);
                                end loop;
                            when 3 =>
                                next_buffer := output_buffer;
                                for lane in 0 to 24 loop
                                    next_buffer(75 + lane) := input_data(lane);
                                end loop;
                                output_data  <= next_buffer;
                                output_valid <= '1';
                            when others =>
                                assert false
                                    report "output block index is invalid for conv 0"
                                    severity failure;
                                next_buffer := output_buffer;
                        end case;
                        output_buffer <= next_buffer;

                    -- Conv 1: two blocks of 16 pooled activations.
                    when 1 =>
                        case output_block_index is
                            when 0 =>
                                next_buffer := (others => (others => '0'));
                                for lane in 0 to 15 loop
                                    next_buffer(lane) := input_data(lane);
                                end loop;
                            when 1 =>
                                next_buffer := output_buffer;
                                for lane in 0 to 15 loop
                                    next_buffer(16 + lane) := input_data(lane);
                                end loop;
                                output_data  <= next_buffer;
                                output_valid <= '1';
                            when others =>
                                assert false
                                    report "output block index is invalid for conv 1"
                                    severity failure;
                                next_buffer := output_buffer;
                        end case;
                        output_buffer <= next_buffer;

                    -- Conv 2: one block of 64 activations.
                    when 2 =>
                        assert output_block_index = 0
                            report "output block index is invalid for conv 2"
                            severity failure;
                        next_buffer := (others => (others => '0'));
                        for lane in 0 to 63 loop
                            next_buffer(lane) := input_data(lane);
                        end loop;
                        output_buffer <= next_buffer;
                        output_data   <= next_buffer;
                        output_valid  <= '1';

                    -- Dense 0 and dense 1: one block of eight activations.
                    when 3 | 4 =>
                        assert output_block_index = 0
                            report "output block index is invalid for dense layer"
                            severity failure;
                        next_buffer := (others => (others => '0'));
                        for lane in 0 to 7 loop
                            next_buffer(lane) := input_data(lane);
                        end loop;
                        output_buffer <= next_buffer;
                        output_data   <= next_buffer;
                        output_valid  <= '1';

                    -- Final dense layer: one block of two logits.
                    when 5 =>
                        assert output_block_index = 0
                            report "output block index is invalid for final dense layer"
                            severity failure;
                        next_buffer := (others => (others => '0'));
                        for lane in 0 to 1 loop
                            next_buffer(lane) := input_data(lane);
                        end loop;
                        output_buffer <= next_buffer;
                        output_data   <= next_buffer;
                        output_valid  <= '1';

                    when others =>
                        assert false
                            report "layer index is invalid"
                            severity failure;
                end case;
            end if;
        end if;
    end process;
end architecture;
