library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.activation_map_pkg.all;
use work.bias_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity biases_handler is
    port(
        conv_enable         : in std_logic;
        layer_index         : in integer range 0 to N_LAYERS - 1;
        output_block_index  : in integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;

        bias_row            : out integer range 0 to BIAS_ROM_DEPTH - 1;
        bias_rom_data       : in signed(BIAS_ROM_BITS - 1 downto 0);
        bias_out            : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of biases_handler is
begin
    -- Every legal layer/block combination maps to a fixed ROM row.  Keeping
    -- these choices explicit prevents a general-purpose integer adder and
    -- package-array mux from being inferred for this ten-row memory.
    process (all)
    begin
        if conv_enable = '1' then
            case layer_index is
                when 0 =>
                    case output_block_index is
                        when 0      => bias_row <= 0;
                        when 1      => bias_row <= 1;
                        when 2      => bias_row <= 2;
                        when others => bias_row <= 3;
                    end case;

                when 1 =>
                    case output_block_index is
                        when 0      => bias_row <= 4;
                        when 1      => bias_row <= 5;
                        when 2      => bias_row <= 6;
                        when others => bias_row <= 7;
                    end case;

                when 2 =>
                    case output_block_index is
                        when 0      => bias_row <= 6;
                        when 1      => bias_row <= 7;
                        when 2      => bias_row <= 8;
                        when others => bias_row <= 9;
                    end case;

                -- These combinations are not issued by the controller.
                when 3      => bias_row <= 7;
                when 4      => bias_row <= 8;
                when others => bias_row <= 9;
            end case;
        else
            case layer_index is
                when 0      => bias_row <= 0;
                when 1      => bias_row <= 4;
                when 2      => bias_row <= 6;
                when 3      => bias_row <= 7;
                when 4      => bias_row <= 8;
                when others => bias_row <= 9;
            end case;
        end if;
    end process;

    process (all)
    begin
        bias_out <= (others => (others => '0'));

        if conv_enable = '1' then
            case layer_index is
                -- Conv 0: each row contains one bias for all 100 positions.
                when 0 =>
                    for lane in 0 to 99 loop
                        bias_out(lane) <= resize(
                            bias_rom_data(BIAS_BITS - 1 downto 0),
                            ACC_W_MAX
                        );
                    end loop;

                -- Conv 1: four bias slots, each covering 25 positions.
                when 1 =>
                    if output_block_index < 2 then
                        for slot in 0 to 3 loop
                            for position in 0 to 24 loop
                                bias_out(slot * 25 + position) <= resize(
                                    bias_rom_data(
                                        (slot + 1) * BIAS_BITS - 1 downto
                                        slot * BIAS_BITS
                                    ),
                                    ACC_W_MAX
                                );
                            end loop;
                        end loop;
                    end if;

                -- Conv 2: sixteen bias slots, each covering four positions.
                when 2 =>
                    if output_block_index = 0 then
                        for slot in 0 to 15 loop
                            for position in 0 to 3 loop
                                bias_out(slot * 4 + position) <= resize(
                                    bias_rom_data(
                                        (slot + 1) * BIAS_BITS - 1 downto
                                        slot * BIAS_BITS
                                    ),
                                    ACC_W_MAX
                                );
                            end loop;
                        end loop;
                    end if;

                when others =>
                    null;
            end case;
        else
            -- Dense layers use one fixed slot per active output lane.  The
            -- layer 0..2 cases retain the entity's behavior for mismatched
            -- conv_enable inputs while keeping every slice static.
            case layer_index is
                when 0 =>
                    for lane in 0 to 3 loop
                        bias_out(lane) <= resize(
                            bias_rom_data(
                                (lane + 1) * BIAS_BITS - 1 downto
                                lane * BIAS_BITS
                            ),
                            ACC_W_MAX
                        );
                    end loop;

                when 1 | 3 | 4 =>
                    for lane in 0 to 7 loop
                        bias_out(lane) <= resize(
                            bias_rom_data(
                                (lane + 1) * BIAS_BITS - 1 downto
                                lane * BIAS_BITS
                            ),
                            ACC_W_MAX
                        );
                    end loop;

                when 2 =>
                    for lane in 0 to 15 loop
                        bias_out(lane) <= resize(
                            bias_rom_data(
                                (lane + 1) * BIAS_BITS - 1 downto
                                lane * BIAS_BITS
                            ),
                            ACC_W_MAX
                        );
                    end loop;

                when others =>
                    for lane in 0 to 1 loop
                        bias_out(lane) <= resize(
                            bias_rom_data(
                                (lane + 1) * BIAS_BITS - 1 downto
                                lane * BIAS_BITS
                            ),
                            ACC_W_MAX
                        );
                    end loop;
            end case;
        end if;
    end process;
end architecture;
