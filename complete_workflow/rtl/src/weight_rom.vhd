library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;

entity weight_rom is
    generic (
        ROM_DEPTH   : positive;
        WORD_LENGTH : positive;
        FILE_NAME   : string
    );
    port (
        clk         : in std_logic;
        read_enable : in std_logic;
        addr        : in integer range 0 to ROM_DEPTH - 1;

        read_valid  : out std_logic := '0';
        data_out    : out signed(WORD_LENGTH - 1 downto 0)
    );
end entity;

architecture rtl of weight_rom is
    attribute rom_style : string;

    signal rom_block : signed_vector(0 to ROM_DEPTH - 1)(WORD_LENGTH - 1 downto 0)
                     := init_signed_rom_from_file(FILE_NAME, ROM_DEPTH, WORD_LENGTH);
    attribute rom_style of rom_block : signal is "block";
begin
    process (clk)
    begin
        if rising_edge(clk) then
            read_valid <= read_enable;
            if read_enable = '1' then
                data_out <= rom_block(addr);
            end if;
        end if;
    end process;
end architecture;
