library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity weight_rom is
    generic(
        WEIGHT_ROM_DEPTH: integer;
        WEIGHT_ROM_BITS: integer
    );
    port(
        clk:        in std_logic;
        addr:       in integer range 0 to WEIGHT_ROM_DEPTH - 1;
        data_out:   out signed(WEIGHT_ROM_BITS - 1 downto 0)
    );
end entity;

architecture rtl of weight_rom is
    -- init ROM type 
    type rom_type is array(0 to WEIGHT_ROM_DEPTH - 1) of signed(WEIGHT_ROM_BITS - 1 downto 0);

    -- funciton to read from mem file
    impure function init_rom_from_file(file_name: string) return rom_type is
        file mem_file       : text open read_mode is file_name;
        variable mem_line   : line;
        variable temp_word  : signed(WEIGHT_ROM_BITS - 1 downto 0);
        variable temp_rom   : rom_type := (others => (others => '0'));
    begin
        for i in rom_type'range loop
            if not endfile(mem_file) then
                readline(mem_file, mem_line);
                hread(mem_line, temp_word);
                temp_mem(i) := temp_word;
            else
                exit;
            end if;
        end loop;
        return temp_mem;
    end function;

    -- ROM instance
    signal rom_block: rom_type := init_rom_from_file("weights.mem");
begin
    process(clk)
    begin
        if rising_edge(clk) then 
            data_out <= rom_block(addr);
        end if;
    end process;
end architecture;