library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.preprocess_pkg.all;

entity preprocess is
    port (
        clk               : in std_logic;
        rst               : in std_logic;

        frame_pixel_valid : in std_logic;
        frame_pixel_addr  : in unsigned(FRAME_ADDR_W - 1 downto 0);
        frame_pixel_data  : in unsigned(RAW_W - 1 downto 0);

        start             : in std_logic;
        site_ready        : in std_logic;

        busy              : out std_logic;
        site_valid        : out std_logic;
        site_index        : out unsigned(SITE_INDEX_W - 1 downto 0);
        crop_data         : out signed_vector(0 to CROP_PIXELS - 1)(IN_W - 1 downto 0);
        neighbor_data     : out signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);
        done              : out std_logic
    );
end entity;

architecture rtl of preprocess is
    type state_t is (
        IDLE,
        NEIGHBOR_MULTIPLY,
        NEIGHBOR_FINISH,
        PRESENT_SITES
    );
    type crop_row_t is array(0 to CROP_PIXELS - 1) of
        signed(IN_W - 1 downto 0);
    type crop_memory_t is array(0 to SITE_COUNT - 1) of crop_row_t;
    type site_sum_t is array(0 to SITE_COUNT - 1) of
        integer range 0 to CROP_PIXELS * ((2 ** RAW_W) - 1);
    type neighbor_code_t is array(0 to SITE_COUNT - 1) of
        signed(NB_W - 1 downto 0);

    signal state            : state_t := IDLE;
    signal next_state       : state_t := IDLE;
    signal crop_rows        : crop_memory_t :=
                              (others => (others => (others => '0')));
    signal site_sums        : site_sum_t := (others => 0);

    -- One pipeline register lets a new frame pixel arrive every clock while
    -- keeping the normalization lookup off the external input path.
    signal pixel_valid_reg  : std_logic := '0';
    signal raw_pixel_reg    : unsigned(RAW_W - 1 downto 0) := (others => '0');
    signal pixel_x_reg      : integer range 0 to FRAME_WIDTH - 1 := 0;
    signal pixel_y_reg      : integer range 0 to FRAME_HEIGHT - 1 := 0;
    signal stream_x         : integer range 0 to FRAME_WIDTH - 1 := 0;
    signal stream_y         : integer range 0 to FRAME_HEIGHT - 1 := 0;
    signal normalized_pixel : signed(IN_W - 1 downto 0);

    signal quantize_site    : integer range 0 to SITE_COUNT - 1 := 0;
    signal selected_site_sum : signed(CROP_SUM_W downto 0) := (others => '0');
    signal neighbor_product : signed(CROP_SUM_W + NB_MULTIPLIER_W downto 0)
                            := (others => '0');
    signal neighbor_codes   : neighbor_code_t := (others => (others => '0'));
    signal presented_site   : integer range 0 to SITE_COUNT - 1 := 0;
    signal done_reg         : std_logic := '0';

    function round_shift_even(value : signed; amount : positive) return signed is
        variable quotient  : signed(value'range);
        variable remainder : unsigned(amount - 1 downto 0);
        variable halfway   : unsigned(amount - 1 downto 0) := (others => '0');
    begin
        quotient := shift_right(value, amount);
        remainder := unsigned(value(amount - 1 downto 0));
        halfway(amount - 1) := '1';
        if remainder > halfway
            or (remainder = halfway and quotient(0) = '1')
        then
            quotient := quotient + 1;
        end if;
        return quotient;
    end function;

    function finish_neighbor_quantization(
        affine : signed(NB_AFFINE_W - 1 downto 0)
    ) return signed is
        variable rounded : signed(NB_AFFINE_W - 1 downto 0);
        variable result  : signed(NB_W - 1 downto 0);
    begin
        rounded := round_shift_even(affine, NB_AFFINE_FRACTION);

        if rounded > to_signed((2 ** (NB_W - 1)) - 1, rounded'length) then
            result := to_signed((2 ** (NB_W - 1)) - 1, NB_W);
        elsif rounded < to_signed(-(2 ** (NB_W - 1)), rounded'length) then
            result := to_signed(-(2 ** (NB_W - 1)), NB_W);
        else
            result := resize(rounded, NB_W);
        end if;
        return result;
    end function;

begin
    busy <= '0' when state = IDLE else '1';
    done <= done_reg;
    site_valid <= '1' when state = PRESENT_SITES else '0';
    site_index <= to_unsigned(presented_site, SITE_INDEX_W);
    normalized_pixel <= to_signed(
        RAW_TO_INPUT(to_integer(raw_pixel_reg)),
        IN_W
    );

    controller : process(all)
    begin
        next_state <= state;

        case state is
            when IDLE =>
                if start = '1' then
                    next_state <= NEIGHBOR_MULTIPLY;
                end if;

            when NEIGHBOR_MULTIPLY =>
                next_state <= NEIGHBOR_FINISH;

            when NEIGHBOR_FINISH =>
                if quantize_site = SITE_COUNT - 1 then
                    next_state <= PRESENT_SITES;
                else
                    next_state <= NEIGHBOR_MULTIPLY;
                end if;

            when PRESENT_SITES =>
                if site_ready = '1' and presented_site = SITE_COUNT - 1 then
                    next_state <= IDLE;
                end if;
        end case;
    end process;

    state_register : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state <= IDLE;
            else
                state <= next_state;
            end if;
        end if;
    end process;

    output_mux : process(all)
        variable source_site : integer;
    begin
        crop_data <= (others => (others => '0'));
        neighbor_data <= (others => (others => '0'));

        if state = PRESENT_SITES then
            for pixel in 0 to CROP_PIXELS - 1 loop
                crop_data(pixel) <= crop_rows(presented_site)(pixel);
            end loop;

            for slot in 0 to NEIGHBORS - 1 loop
                source_site := NEIGHBOR_SITE(presented_site * NEIGHBORS + slot);
                if source_site < 0 then
                    neighbor_data(slot) <= to_signed(MISSING_NEIGHBOR_CODE, NB_W);
                else
                    neighbor_data(slot) <= neighbor_codes(source_site);
                end if;
            end loop;
        end if;
    end process;

    crop_banks : for site in 0 to SITE_COUNT - 1 generate
        process(clk)
            variable local_x     : integer range 0 to CROP_SIDE - 1;
            variable local_y     : integer range 0 to CROP_SIDE - 1;
            variable local_pixel : integer range 0 to CROP_PIXELS - 1;
        begin
            if rising_edge(clk) then
                if rst = '1' then
                    site_sums(site) <= 0;
                else
                    -- Address zero marks the beginning of a complete new frame.
                    if state = IDLE
                        and frame_pixel_valid = '1'
                        and to_integer(frame_pixel_addr) = 0
                    then
                        site_sums(site) <= 0;
                    elsif pixel_valid_reg = '1'
                        and pixel_x_reg >= CROP_X_START(site)
                        and pixel_x_reg < CROP_X_START(site) + CROP_SIDE
                        and pixel_y_reg >= CROP_Y_START(site)
                        and pixel_y_reg < CROP_Y_START(site) + CROP_SIDE
                    then
                        local_x := pixel_x_reg - CROP_X_START(site);
                        local_y := pixel_y_reg - CROP_Y_START(site);
                        local_pixel := local_y * CROP_SIDE + local_x;

                        crop_rows(site)(local_pixel) <= normalized_pixel;
                        site_sums(site) <=
                            site_sums(site) + to_integer(raw_pixel_reg);
                    end if;
                end if;
            end if;
        end process;
    end generate;

    datapath : process(clk)
        variable affine_value : signed(NB_AFFINE_W - 1 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                pixel_valid_reg  <= '0';
                raw_pixel_reg    <= (others => '0');
                pixel_x_reg      <= 0;
                pixel_y_reg      <= 0;
                stream_x         <= 0;
                stream_y         <= 0;
                quantize_site    <= 0;
                selected_site_sum <= (others => '0');
                neighbor_product <= (others => '0');
                neighbor_codes   <= (others => (others => '0'));
                presented_site   <= 0;
                done_reg         <= '0';
            else
                done_reg <= '0';
                pixel_valid_reg <= '0';

                if state = IDLE and frame_pixel_valid = '1' then
                    assert to_integer(frame_pixel_addr) =
                           stream_y * FRAME_WIDTH + stream_x
                        report "preprocess frame pixels must arrive in row-major order"
                        severity failure;

                    raw_pixel_reg   <= frame_pixel_data;
                    pixel_x_reg     <= stream_x;
                    pixel_y_reg     <= stream_y;
                    pixel_valid_reg <= '1';

                    if stream_x = FRAME_WIDTH - 1 then
                        stream_x <= 0;
                        if stream_y = FRAME_HEIGHT - 1 then
                            stream_y <= 0;
                        else
                            stream_y <= stream_y + 1;
                        end if;
                    else
                        stream_x <= stream_x + 1;
                    end if;
                end if;

                case state is
                    when IDLE =>
                        if start = '1' then
                            quantize_site  <= 0;
                            selected_site_sum <= to_signed(site_sums(0), selected_site_sum'length);
                            presented_site <= 0;
                        end if;

                    when NEIGHBOR_MULTIPLY =>
                        neighbor_product <=
                            selected_site_sum
                            * to_signed(NB_MULTIPLIER, NB_MULTIPLIER_W);

                    when NEIGHBOR_FINISH =>
                        affine_value := resize(neighbor_product, affine_value'length)
                                        - NB_OFFSET;
                        neighbor_codes(quantize_site) <=
                            finish_neighbor_quantization(affine_value);

                        if quantize_site = SITE_COUNT - 1 then
                            presented_site <= 0;
                        else
                            selected_site_sum <= to_signed(
                                site_sums(quantize_site + 1), selected_site_sum'length
                            );
                            quantize_site <= quantize_site + 1;
                        end if;

                    when PRESENT_SITES =>
                        if site_ready = '1' then
                            if presented_site = SITE_COUNT - 1 then
                                done_reg <= '1';
                            else
                                presented_site <= presented_site + 1;
                            end if;
                        end if;
                end case;
            end if;
        end if;
    end process;
end architecture;
