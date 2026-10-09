library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

entity tb_lut_ram is
end entity;

architecture sim of tb_lut_ram is
    constant CLK_PERIOD      : time := 10 ns;
    function first_value(lane : integer) return integer is
    begin
        return lane - 50;
    end function;

    function second_value(lane : integer) return integer is
    begin
        if lane mod 2 = 0 then
            return 100 + lane / 2;
        end if;
        return -(100 + lane / 2);
    end function;

    constant NEIGHBOR_VALUES : integer_vector(0 to NEIGHBORS - 1) := (
        7, -9, 15, -21, 31, -45, 63, -81
    );

    signal clk             : std_logic := '0';
    signal write_data      : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                           := (others => (others => '0'));
    signal write_enable    : std_logic := '0';
    signal read_enable     : std_logic := '0';
    signal read_valid      : std_logic;
    signal read_data       : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal nb_in           : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0)
                           := (others => (others => '0'));
    signal nb_out          : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
    signal nb_load         : std_logic := '0';

    -- Scalar status signals are the only objects Cocotb needs to inspect.
    signal tb_done         : std_logic := '0';
    signal tb_passed       : std_logic := '0';
begin
    clk <= not clk after CLK_PERIOD / 2 when tb_done = '0' else '0';

    dut : entity work.act_ram
        port map (
            clk             => clk,
            write_data      => write_data,
            write_enable    => write_enable,
            read_enable     => read_enable,
            read_valid      => read_valid,
            read_data       => read_data,
            nb_in           => nb_in,
            nb_out          => nb_out,
            nb_load         => nb_load
        );

    stimulus : process
    begin
        -- Write the first activation word and load the neighbours.
        wait until falling_edge(clk);
        for lane in 0 to MAX_MAC_UNITS - 1 loop
            write_data(lane) <= to_signed(first_value(lane), ACT_W_MAX);
        end loop;
        for neighbor in 0 to NEIGHBORS - 1 loop
            nb_in(neighbor) <= to_signed(NEIGHBOR_VALUES(neighbor), NB_W);
        end loop;
        write_enable <= '1';
        nb_load      <= '1';
        wait until rising_edge(clk);

        -- Read the stored activation word.
        wait until falling_edge(clk);
        write_enable    <= '0';
        nb_load         <= '0';
        read_enable     <= '1';
        wait until rising_edge(clk);
        wait for 1 ns;

        assert read_valid = '1'
            report "read_valid was not asserted for shelf 0"
            severity failure;
        for lane in 0 to MAX_MAC_UNITS - 1 loop
            assert read_data(lane) = to_signed(first_value(lane), ACT_W_MAX)
                report "First activation word mismatch at lane " & integer'image(lane)
                severity failure;
        end loop;

        -- With no request, valid drops and the registered data holds.
        wait until falling_edge(clk);
        read_enable <= '0';
        wait until rising_edge(clk);
        wait for 1 ns;
        assert read_valid = '0'
            report "read_valid did not clear when read_enable was low"
            severity failure;
        for lane in 0 to MAX_MAC_UNITS - 1 loop
            assert read_data(lane) = to_signed(first_value(lane), ACT_W_MAX)
                report "read_data did not hold its previous value at lane " & integer'image(lane)
                severity failure;
        end loop;

        for neighbor in 0 to NEIGHBORS - 1 loop
            assert nb_out(neighbor) = to_signed(NEIGHBOR_VALUES(neighbor), NB_W)
                report "Neighbour mismatch at index " & integer'image(neighbor)
                severity failure;
        end loop;

        -- Overwrite the activation word and read the new layer back.
        wait until falling_edge(clk);
        for lane in 0 to MAX_MAC_UNITS - 1 loop
            write_data(lane) <= to_signed(second_value(lane), ACT_W_MAX);
        end loop;
        write_enable <= '1';
        wait until rising_edge(clk);

        wait until falling_edge(clk);
        write_enable    <= '0';
        read_enable     <= '1';
        wait until rising_edge(clk);
        wait for 1 ns;

        assert read_valid = '1'
            report "read_valid was not asserted after overwrite"
            severity failure;
        for lane in 0 to MAX_MAC_UNITS - 1 loop
            assert read_data(lane) = to_signed(second_value(lane), ACT_W_MAX)
                report "Overwritten activation word mismatch at lane " & integer'image(lane)
                severity failure;
        end loop;

        tb_passed <= '1';
        tb_done   <= '1';
        wait;
    end process;
end architecture;
