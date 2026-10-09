library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.preprocess_pkg.all;

entity detector_core is
    generic (
        WEIGHT_FILE : string := "weights.mem";
        BIAS_FILE   : string := "biases.mem"
    );
    port (
        clk                : in std_logic;
        rst                : in std_logic;

        frame_write_enable : in std_logic;
        frame_write_addr   : in integer range 0 to FRAME_PIXELS - 1;
        frame_write_data   : in unsigned(RAW_W - 1 downto 0);
        start              : in std_logic;

        busy               : out std_logic;
        result_valid       : out std_logic;
        result_site_index  : out integer range 0 to SITE_COUNT - 1;
        result_logits      : out signed_vector(0 to CLASSES - 1)(ACT_W_MAX - 1 downto 0);
        result_decision    : out std_logic;
        preprocess_cycles  : out unsigned(31 downto 0);
        cnn_cycles         : out unsigned(31 downto 0);
        pipeline_cycles    : out unsigned(31 downto 0);
        done               : out std_logic
    );
end entity;

architecture rtl of detector_core is
    type state_t is (IDLE, WAIT_SITE, LOAD_INPUT, START_INFERENCE, WAIT_RESULT);
    signal state : state_t := IDLE;

    signal preprocess_start        : std_logic;
    signal preprocess_site_ready   : std_logic;
    signal preprocess_busy         : std_logic;
    signal preprocess_site_valid   : std_logic;
    signal preprocess_done         : std_logic;
    signal preprocess_site_index   : unsigned(SITE_INDEX_W - 1 downto 0);
    signal preprocess_crop         : signed_vector(0 to CROP_PIXELS - 1)(IN_W - 1 downto 0);
    signal preprocess_neighbors    : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
    signal input_crop_reg          : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                                   := (others => (others => '0'));
    signal input_neighbors_reg     : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0)
                                   := (others => (others => '0'));

    signal inference_load_input    : std_logic;
    signal inference_start         : std_logic;
    signal inference_busy          : std_logic;
    signal inference_result_valid  : std_logic;
    signal inference_result_data   : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal inference_decision      : std_logic;

    signal accepted_site_index     : integer range 0 to SITE_COUNT - 1 := 0;
    signal done_reg                : std_logic := '0';
    signal preprocess_count, cnn_count, pipeline_count : unsigned(31 downto 0) := (others => '0');
    signal preprocess_counting, cnn_counting, pipeline_counting : std_logic := '0';
    attribute DONT_TOUCH : string;
    attribute DONT_TOUCH of input_crop_reg, input_neighbors_reg : signal is "TRUE";
begin
    preprocess_start      <= '1' when state = IDLE and start = '1' else '0';
    preprocess_site_ready <= '1' when state = WAIT_SITE and preprocess_site_valid = '1' else '0';
    inference_load_input  <= '1' when state = LOAD_INPUT else '0';
    inference_start       <= '1' when state = START_INFERENCE else '0';

    busy              <= '1' when state /= IDLE else '0';
    done              <= done_reg;
    result_valid      <= inference_result_valid when state = WAIT_RESULT else '0';
    result_site_index <= accepted_site_index;
    result_logits(0)  <= inference_result_data(0);
    result_logits(1)  <= inference_result_data(1);
    result_decision   <= inference_decision;
    preprocess_cycles <= preprocess_count;
    cnn_cycles        <= cnn_count;
    pipeline_cycles   <= pipeline_count;

    process (clk)
    begin
        if rising_edge(clk) then
            if rst = '0' and preprocess_site_ready = '1' then
                input_crop_reg <= (others => (others => '0'));
                for pixel in 0 to CROP_PIXELS - 1 loop
                    input_crop_reg(pixel) <= resize(preprocess_crop(pixel), ACT_W_MAX);
                end loop;
                input_neighbors_reg <= preprocess_neighbors;
            end if;
        end if;
    end process;

    -- Elapsed clock periods: the accepted start edge is zero. A registered
    -- valid is observed on the following edge, so retain the previous count
    -- on that edge. These monitors do not add stages to the datapath.
    latency_counters : process (clk, rst)
    begin
        if rst = '1' then
            preprocess_count    <= (others => '0');
            cnn_count           <= (others => '0');
            pipeline_count      <= (others => '0');
            preprocess_counting <= '0';
            cnn_counting        <= '0';
            pipeline_counting   <= '0';
        elsif rising_edge(clk) then
            if state = IDLE and frame_write_enable = '1' and frame_write_addr = 0 then
                preprocess_count    <= (others => '0');
                cnn_count           <= (others => '0');
                pipeline_count      <= (others => '0');
                preprocess_counting <= '1';
                cnn_counting        <= '0';
                pipeline_counting   <= '1';
            else
                if preprocess_counting = '1' then
                    if preprocess_site_valid = '1' then
                        preprocess_counting <= '0';
                    else
                        preprocess_count <= preprocess_count + 1;
                    end if;
                end if;

                if pipeline_counting = '1' then
                    if state = WAIT_RESULT and inference_result_valid = '1'
                        and accepted_site_index = SITE_COUNT - 1
                    then
                        pipeline_counting <= '0';
                    else
                        pipeline_count <= pipeline_count + 1;
                    end if;
                end if;

                if inference_start = '1' then
                    cnn_count    <= (others => '0');
                    cnn_counting <= '1';
                elsif cnn_counting = '1' then
                    if inference_result_valid = '1' then
                        cnn_counting <= '0';
                    else
                        cnn_count <= cnn_count + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    input_preprocessing : entity work.preprocess
        port map (
            clk               => clk,
            rst               => rst,
            frame_pixel_valid => frame_write_enable,
            frame_pixel_addr  => to_unsigned(frame_write_addr, FRAME_ADDR_W),
            frame_pixel_data  => frame_write_data,
            start             => preprocess_start,
            site_ready        => preprocess_site_ready,
            busy              => preprocess_busy,
            site_valid        => preprocess_site_valid,
            done              => preprocess_done,
            site_index        => preprocess_site_index,
            crop_data         => preprocess_crop,
            neighbor_data     => preprocess_neighbors
        );

    neural_network : entity work.inference_core
        generic map (
            WEIGHT_FILE => WEIGHT_FILE,
            BIAS_FILE   => BIAS_FILE
        )
        port map (
            clk           => clk,
            rst           => rst,
            load_input    => inference_load_input,
            start         => inference_start,
            input_data    => input_crop_reg,
            neighbor_data => input_neighbors_reg,
            busy          => inference_busy,
            result_valid  => inference_result_valid,
            result_data   => inference_result_data,
            decision      => inference_decision
        );

    process (clk, rst)
    begin
        if rst = '1' then
            state               <= IDLE;
            accepted_site_index <= 0;
            done_reg            <= '0';
        elsif rising_edge(clk) then
            done_reg <= '0';

            case state is
                when IDLE =>
                    if start = '1' then
                        state <= WAIT_SITE;
                    end if;

                when WAIT_SITE =>
                    if preprocess_site_valid = '1' then
                        accepted_site_index <= to_integer(preprocess_site_index);
                        state               <= LOAD_INPUT;
                    end if;

                when LOAD_INPUT =>
                    state <= START_INFERENCE;

                when START_INFERENCE =>
                    state <= WAIT_RESULT;

                when WAIT_RESULT =>
                    if inference_result_valid = '1' then
                        if accepted_site_index = SITE_COUNT - 1 then
                            done_reg <= '1';
                            state    <= IDLE;
                        else
                            state <= WAIT_SITE;
                        end if;
                    end if;
            end case;
        end if;
    end process;
end architecture;
