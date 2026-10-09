library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;

entity mac_array is
    generic(
        ACT_W_MAX:      integer;
        WEIGHT_W_MAX:   integer;
        ACC_W_MAX:      integer;
        MAX_MAC_UNITS:  integer;
        MAC_UNITS:      integer
    );

    port(
        clk         : in std_logic;
        rst         : in std_logic;
        valid       : in std_logic;
        load_bias   : in std_logic;

        bias        : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);

        activation  : in signed_vector(0 to MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        weight      : in signed_vector(0 to MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);

        busy        : out std_logic;
        accum       : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
    );
end entity;

architecture rtl of mac_array is
    signal activation_reg : signed_vector(0 to MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                          := (others => (others => '0'));
    signal weight_reg     : signed_vector(0 to MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0)
                          := (others => (others => '0'));
    signal bias_reg       : signed_vector(0 to MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
                          := (others => (others => '0'));
    signal product_reg    : signed_vector(0 to MAC_UNITS - 1)(ACT_W_MAX + WEIGHT_W_MAX - 1 downto 0)
                          := (others => (others => '0'));
    signal accum_temp     : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
                          := (others => (others => '0'));

    signal input_valid   : std_logic := '0';
    signal input_load    : std_logic := '0';
    signal product_valid : std_logic := '0';
    signal product_load  : std_logic := '0';

    attribute use_dsp : string;
    attribute use_dsp of product_reg : signal is "yes";
    attribute use_dsp of accum_temp  : signal is "yes";

begin

    -- assertion
    assert MAC_UNITS <= MAX_MAC_UNITS
        report "MAC units greater than Max MAC unit count"
        severity failure;

    -- MAC operations
    process(clk)
        variable addend : signed(ACC_W_MAX - 1 downto 0);
    begin
        if rising_edge(clk) then
            assert not (valid = '1' and load_bias = '1'
                        and input_valid = '1' and input_load = '1')
                report "mac_array needs at least two products per sum"
                severity failure;

            for i in 0 to MAC_UNITS - 1 loop
                activation_reg(i) <= activation(i);
                weight_reg(i)     <= weight(i);
                product_reg(i)    <= activation_reg(i) * weight_reg(i);
                if valid = '1' and load_bias = '1' then
                    bias_reg(i) <= bias(i);
                end if;
            end loop;

            if rst = '1' then
                input_valid   <= '0';
                input_load    <= '0';
                product_valid <= '0';
                product_load  <= '0';
                accum_temp    <= (others => (others => '0'));
            else
                input_valid   <= valid;
                input_load    <= load_bias;
                product_valid <= input_valid;
                product_load  <= input_load;

                if product_valid = '1' then
                    for i in 0 to MAC_UNITS - 1 loop
                        if product_load = '1' then
                            addend := bias_reg(i);
                        else
                            addend := accum_temp(i);
                        end if;
                        accum_temp(i) <= addend + resize(product_reg(i), ACC_W_MAX);
                    end loop;
                end if;
            end if;
        end if;
    end process;

    busy <= input_valid or product_valid;

    -- accumulator output
    accum <= accum_temp;

end architecture;
