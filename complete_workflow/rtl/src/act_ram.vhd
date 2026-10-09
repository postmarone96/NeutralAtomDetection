library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;


entity act_ram is
    port(
        clk             : in std_logic;
        write_data      : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

        write_enable    : in  std_logic;
        read_enable     : in  std_logic;
        read_valid      : out std_logic;   

        read_data       : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

        nb_in           : in signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
        nb_out          : out signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
        nb_load         : in std_logic
    );
end entity;

architecture rtl of act_ram is 
    signal buffer_nb    : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0) := (others =>(others=>'0'));
    signal read_reg     : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0) := (others =>(others=>'0'));

    signal activation_buffer : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                             := (others => (others => '0'));

begin
    -- A completed layer overwrites the previous 100-lane activation vector.
    process(clk)
    begin
        if rising_edge(clk) then
            if write_enable = '1' then
                activation_buffer <= write_data;
            end if;

            if read_enable = '1' then
                read_reg <= activation_buffer;
            end if;

            read_valid <= read_enable;
        end if;
    end process;

    -- nb process
    process(clk)
    begin
        if rising_edge(clk) then
            if nb_load = '1' then
                buffer_nb <= nb_in;
            end if;
        end if;
    end process;

    -- read output
    read_data <= read_reg;

    -- nb output
    nb_out <= buffer_nb;

end architecture;
