library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;

entity bias_rom is
    generic (
        ROM_DEPTH   : positive;
        WORD_LENGTH : positive;
        FILE_NAME   : string
    );
    port (
        addr     : in integer range 0 to ROM_DEPTH - 1;
        data_out : out signed(WORD_LENGTH - 1 downto 0)
    );
end entity;

architecture rtl of bias_rom is
    attribute rom_style : string;

    signal rom_data : signed_vector(0 to ROM_DEPTH - 1)(WORD_LENGTH - 1 downto 0)
                    := init_signed_rom_from_file(FILE_NAME, ROM_DEPTH, WORD_LENGTH);
    attribute rom_style of rom_data : signal is "distributed";
begin
    data_out <= rom_data(addr);
end architecture;
