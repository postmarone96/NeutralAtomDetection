library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

-- Fast protocol stub used only by the XRT wrapper test.  It checks that all
-- 784 pixels arrive in address order, then emits deterministic indexed results.
entity detector_core is
    generic (
        WEIGHT_FILE : string := "weights.mem";
        BIAS_FILE   : string := "biases.mem"
    );
    port (
        clk                 : in  std_logic;
        rst                 : in  std_logic;
        frame_write_enable  : in  std_logic;
        frame_write_addr    : in  integer range 0 to 783;
        frame_write_data    : in  unsigned(RAW_W - 1 downto 0);
        start               : in  std_logic;
        busy                : out std_logic;
        result_valid        : out std_logic;
        result_site_index   : out integer range 0 to 8;
        result_logits       : out signed_vector(0 to 1)(ACT_W_MAX - 1 downto 0);
        result_decision     : out std_logic;
        preprocess_cycles   : out unsigned(31 downto 0);
        cnn_cycles          : out unsigned(31 downto 0);
        pipeline_cycles     : out unsigned(31 downto 0);
        done                : out std_logic
    );
end entity;

architecture simulation of detector_core is
    signal pixel_count : integer range 0 to 784 := 0;
    signal running     : std_logic := '0';
    signal wait_count  : integer range 0 to 2 := 0;
    signal site        : integer range 0 to 8 := 0;
begin
    busy <= running;
    preprocess_cycles <= to_unsigned(812, 32);
    cnn_cycles        <= to_unsigned(858, 32);
    pipeline_cycles   <= to_unsigned(8550, 32);

    process (clk)
    begin
        if rising_edge(clk) then
            result_valid <= '0';
            done <= '0';

            if rst = '1' then
                pixel_count     <= 0;
                running         <= '0';
                wait_count      <= 0;
                site            <= 0;
                result_valid    <= '0';
                result_site_index <= 0;
                result_logits   <= (others => (others => '0'));
                result_decision <= '0';
                done            <= '0';
            else
                if frame_write_enable = '1' then
                    assert frame_write_addr = pixel_count
                        report "frame pixel address is not sequential" severity failure;
                    assert frame_write_data = to_unsigned(pixel_count mod 1024, RAW_W)
                        report "frame pixel data mismatch" severity failure;
                    pixel_count <= pixel_count + 1;
                end if;

                if start = '1' then
                    assert pixel_count = 784
                        report "detector started before all pixels were loaded" severity failure;
                    running    <= '1';
                    site       <= 0;
                    wait_count <= 2;
                elsif running = '1' then
                    if wait_count = 0 then
                        result_valid      <= '1';
                        result_site_index <= site;
                        result_logits(0)  <= to_signed(site - 4, ACT_W_MAX);
                        result_logits(1)  <= to_signed(20 + site, ACT_W_MAX);
                        if site mod 2 = 0 then
                            result_decision <= '0';
                        else
                            result_decision <= '1';
                        end if;

                        if site = 8 then
                            done    <= '1';
                            running <= '0';
                        else
                            site       <= site + 1;
                            wait_count <= 2;
                        end if;
                    else
                        wait_count <= wait_count - 1;
                    end if;
                end if;
            end if;
        end if;
    end process;
end architecture;
