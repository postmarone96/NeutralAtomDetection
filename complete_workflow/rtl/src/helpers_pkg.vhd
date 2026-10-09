library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

package helpers_pkg is

    type signed_vector is array(integer range <>) of signed;

    function minimum_integer(a, b : integer) return integer;
    function maximum_integer(a, b : integer) return integer;

    function maximum_of_four(
        value_0    : signed;
        value_1    : signed;
        value_2    : signed;
        value_3    : signed;
        round_half : boolean := false
    ) return signed;

    function requantize_relu(
        raw_value    : signed;
        shift_count  : positive;
        output_width : positive
    ) return signed;

    function requantize_signed(
        raw_value    : signed;
        shift_count  : positive;
        output_width : positive
    ) return signed;

    impure function init_signed_rom_from_file(
        source_file : string;
        rom_depth   : positive;
        word_length : positive
    ) return signed_vector;

end package helpers_pkg;

package body helpers_pkg is

    function minimum_integer(a, b : integer) return integer is
    begin
        if a < b then return a; end if;
        return b;
    end function;

    function maximum_integer(a, b : integer) return integer is
    begin
        if a > b then return a; end if;
        return b;
    end function;

    function round_shift_one(value : signed) return signed is
        variable rounded : signed(value'range);
    begin
        rounded := shift_right(value, 1);
        if value(0) = '1' and value(1) = '1' then
            rounded := rounded + 1;
        end if;
        return rounded;
    end function;

    function maximum_of_four(
        value_0    : signed;
        value_1    : signed;
        value_2    : signed;
        value_3    : signed;
        round_half : boolean := false
    ) return signed is
        variable ge_01, ge_02, ge_03, ge_12, ge_13, ge_23 : std_logic;
        variable select_0, select_1, select_2, select_3 : std_logic;
        variable candidate_0, candidate_1, candidate_2, candidate_3 : signed(value_0'range);
        variable mask_0, mask_1, mask_2, mask_3 : signed(value_0'range);
    begin
        ge_01 := '1' when value_0 >= value_1 else '0';
        ge_02 := '1' when value_0 >= value_2 else '0';
        ge_03 := '1' when value_0 >= value_3 else '0';
        ge_12 := '1' when value_1 >= value_2 else '0';
        ge_13 := '1' when value_1 >= value_3 else '0';
        ge_23 := '1' when value_2 >= value_3 else '0';

        select_0 := ge_01 and ge_02 and ge_03;
        select_1 := not ge_01 and ge_12 and ge_13;
        select_2 := not ge_02 and not ge_12 and ge_23;
        select_3 := not ge_03 and not ge_13 and not ge_23;

        candidate_0 := value_0;
        candidate_1 := value_1;
        candidate_2 := value_2;
        candidate_3 := value_3;
        if round_half then
            candidate_0 := round_shift_one(value_0);
            candidate_1 := round_shift_one(value_1);
            candidate_2 := round_shift_one(value_2);
            candidate_3 := round_shift_one(value_3);
        end if;

        mask_0 := (others => select_0);
        mask_1 := (others => select_1);
        mask_2 := (others => select_2);
        mask_3 := (others => select_3);
        return (candidate_0 and mask_0) or (candidate_1 and mask_1)
            or (candidate_2 and mask_2) or (candidate_3 and mask_3);
    end function;

    function requantize_relu(
        raw_value    : signed;
        shift_count  : positive;
        output_width : positive
    ) return signed is
        constant maximum : signed(output_width - 1 downto 0)
            := to_signed(2**(output_width - 1) - 1, output_width);
        variable rounded : unsigned(output_width - 1 downto 0);
    begin
        if raw_value(raw_value'high) = '1' then
            return to_signed(0, output_width);
        end if;

        rounded := '0' & unsigned(
            raw_value(shift_count + output_width - 2 downto shift_count)
        );
        if raw_value(shift_count - 1) = '1'
            and (unsigned(raw_value(shift_count - 2 downto 0)) /= 0
                 or raw_value(shift_count) = '1') then
            rounded := rounded + 1;
        end if;

        if unsigned(raw_value(raw_value'high downto shift_count + output_width - 1)) /= 0
            or rounded(rounded'high) = '1' then
            return maximum;
        end if;
        return signed(rounded);
    end function;

    function requantize_signed(
        raw_value    : signed;
        shift_count  : positive;
        output_width : positive
    ) return signed is
        constant maximum : signed(output_width - 1 downto 0)
            := to_signed(2**(output_width - 1) - 1, output_width);
        constant minimum : signed(output_width - 1 downto 0)
            := to_signed(-2**(output_width - 1), output_width);
        constant negative_extension : signed(raw_value'high downto shift_count + output_width - 1)
            := (others => '1');
        variable rounded : signed(output_width downto 0);
    begin
        rounded := resize(
            raw_value(shift_count + output_width - 1 downto shift_count),
            rounded'length
        );
        if raw_value(shift_count - 1) = '1'
            and (unsigned(raw_value(shift_count - 2 downto 0)) /= 0
                 or raw_value(shift_count) = '1') then
            rounded := rounded + 1;
        end if;

        if raw_value(raw_value'high) = '0' then
            if unsigned(raw_value(raw_value'high downto shift_count + output_width - 1)) /= 0
                or rounded(output_width - 1) = '1' then
                return maximum;
            end if;
        elsif raw_value(raw_value'high downto shift_count + output_width - 1) /= negative_extension then
            return minimum;
        end if;
        return rounded(output_width - 1 downto 0);
    end function;

    impure function init_signed_rom_from_file(
        source_file : string;
        rom_depth   : positive;
        word_length : positive
    ) return signed_vector is
        file mem_file      : text open read_mode is source_file;
        variable mem_line  : line;
        variable temp_word : std_logic_vector(word_length - 1 downto 0);
        variable temp_rom  : signed_vector(0 to rom_depth - 1)(word_length - 1 downto 0)
                           := (others => (others => '0'));
    begin
        for address in temp_rom'range loop
            exit when endfile(mem_file);
            readline(mem_file, mem_line);
            hread(mem_line, temp_word);
            temp_rom(address) := signed(temp_word);
        end loop;
        return temp_rom;
    end function;

end package body helpers_pkg;
