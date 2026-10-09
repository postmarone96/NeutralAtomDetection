library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.helpers_pkg.all;
use work.nn_pkg.all;

-- XRT-facing RTL kernel for one 28 x 28 raw frame.
--
-- Burst reads stage the entire frame before feeding detector_core continuously.
-- A beat contains 32 little-endian uint16 pixels, using bits 9:0 per pixel.
-- The final frame read is a 32-byte narrow beat.
-- detector_core emits nine indexed results, which are buffered here and
-- written in one 512-bit beat after inference has completed.  Buffering all
-- results makes the detector independent of AXI write-response latency.
entity neutral_atom_kernel is
    port (
        ap_clk                  : in  std_logic;
        ap_rst_n                : in  std_logic;
        interrupt               : out std_logic;

        -- AXI4-Lite control slave.
        s_axi_control_AWADDR    : in  std_logic_vector(6 downto 0);
        s_axi_control_AWVALID   : in  std_logic;
        s_axi_control_AWREADY   : out std_logic;
        s_axi_control_WDATA     : in  std_logic_vector(31 downto 0);
        s_axi_control_WSTRB     : in  std_logic_vector(3 downto 0);
        s_axi_control_WVALID    : in  std_logic;
        s_axi_control_WREADY    : out std_logic;
        s_axi_control_BRESP     : out std_logic_vector(1 downto 0);
        s_axi_control_BVALID    : out std_logic;
        s_axi_control_BREADY    : in  std_logic;
        s_axi_control_ARADDR    : in  std_logic_vector(6 downto 0);
        s_axi_control_ARVALID   : in  std_logic;
        s_axi_control_ARREADY   : out std_logic;
        s_axi_control_RDATA     : out std_logic_vector(31 downto 0);
        s_axi_control_RRESP     : out std_logic_vector(1 downto 0);
        s_axi_control_RVALID    : out std_logic;
        s_axi_control_RREADY    : in  std_logic;

        -- AXI4 global-memory master, 512-bit data and 64-bit addresses.
        m_axi_gmem_AWADDR       : out std_logic_vector(63 downto 0);
        m_axi_gmem_AWLEN        : out std_logic_vector(7 downto 0);
        m_axi_gmem_AWSIZE       : out std_logic_vector(2 downto 0);
        m_axi_gmem_AWBURST      : out std_logic_vector(1 downto 0);
        m_axi_gmem_AWLOCK       : out std_logic;
        m_axi_gmem_AWCACHE      : out std_logic_vector(3 downto 0);
        m_axi_gmem_AWPROT       : out std_logic_vector(2 downto 0);
        m_axi_gmem_AWQOS        : out std_logic_vector(3 downto 0);
        m_axi_gmem_AWVALID      : out std_logic;
        m_axi_gmem_AWREADY      : in  std_logic;
        m_axi_gmem_WDATA        : out std_logic_vector(511 downto 0);
        m_axi_gmem_WSTRB        : out std_logic_vector(63 downto 0);
        m_axi_gmem_WLAST        : out std_logic;
        m_axi_gmem_WVALID       : out std_logic;
        m_axi_gmem_WREADY       : in  std_logic;
        m_axi_gmem_BRESP        : in  std_logic_vector(1 downto 0);
        m_axi_gmem_BVALID       : in  std_logic;
        m_axi_gmem_BREADY       : out std_logic;
        m_axi_gmem_ARADDR       : out std_logic_vector(63 downto 0);
        m_axi_gmem_ARLEN        : out std_logic_vector(7 downto 0);
        m_axi_gmem_ARSIZE       : out std_logic_vector(2 downto 0);
        m_axi_gmem_ARBURST      : out std_logic_vector(1 downto 0);
        m_axi_gmem_ARLOCK       : out std_logic;
        m_axi_gmem_ARCACHE      : out std_logic_vector(3 downto 0);
        m_axi_gmem_ARPROT       : out std_logic_vector(2 downto 0);
        m_axi_gmem_ARQOS        : out std_logic_vector(3 downto 0);
        m_axi_gmem_ARVALID      : out std_logic;
        m_axi_gmem_ARREADY      : in  std_logic;
        m_axi_gmem_RDATA        : in  std_logic_vector(511 downto 0);
        m_axi_gmem_RRESP        : in  std_logic_vector(1 downto 0);
        m_axi_gmem_RLAST        : in  std_logic;
        m_axi_gmem_RVALID       : in  std_logic;
        m_axi_gmem_RREADY       : out std_logic
    );
end entity;

architecture rtl of neutral_atom_kernel is
    constant FRAME_PIXELS       : integer := 784;
    constant PIXELS_PER_BEAT    : integer := 32;
    constant FRAME_BEATS        : integer := (FRAME_PIXELS + PIXELS_PER_BEAT - 1) / PIXELS_PER_BEAT;
    constant SITE_COUNT         : integer := 9;

    constant ADDR_AP_CTRL       : integer := 16#00#;
    constant ADDR_GIE           : integer := 16#04#;
    constant ADDR_IER           : integer := 16#08#;
    constant ADDR_ISR           : integer := 16#0C#;
    constant ADDR_FRAME_LO      : integer := 16#10#;
    constant ADDR_FRAME_HI      : integer := 16#14#;
    constant ADDR_RESULTS_LO    : integer := 16#1C#;
    constant ADDR_RESULTS_HI    : integer := 16#20#;
    constant ADDR_STATUS        : integer := 16#28#;
    constant ADDR_RESULTS_DONE  : integer := 16#2C#;

    type mover_state_t is (
        IDLE,
        FRAME_ADDRESS,
        FRAME_DATA,
        FRAME_PRIME,
        FRAME_UNPACK,
        START_DETECTOR,
        DETECTOR_WAIT,
        RESULT_REQUEST,
        RESULT_RESPONSE,
        COMPLETE
    );
    component detector_core is
        generic (
            WEIGHT_FILE : string := "weights.mem";
            BIAS_FILE   : string := "biases.mem"
        );
        port (
            clk                 : in  std_logic;
            rst                 : in  std_logic;
            frame_write_enable  : in  std_logic;
            frame_write_addr    : in  integer range 0 to FRAME_PIXELS - 1;
            frame_write_data    : in  unsigned(RAW_W - 1 downto 0);
            start               : in  std_logic;
            busy                : out std_logic;
            result_valid        : out std_logic;
            result_site_index   : out integer range 0 to SITE_COUNT - 1;
            result_logits       : out signed_vector(0 to 1)(ACT_W_MAX - 1 downto 0);
            result_decision     : out std_logic;
            preprocess_cycles   : out unsigned(31 downto 0);
            cnn_cycles          : out unsigned(31 downto 0);
            pipeline_cycles     : out unsigned(31 downto 0);
            done                : out std_logic
        );
    end component;

    function merge_bytes(
        old_value : std_logic_vector(31 downto 0);
        new_value : std_logic_vector(31 downto 0);
        strobe    : std_logic_vector(3 downto 0)
    ) return std_logic_vector is
        variable merged : std_logic_vector(31 downto 0) := old_value;
    begin
        for byte_index in 0 to 3 loop
            if strobe(byte_index) = '1' then
                merged(8 * byte_index + 7 downto 8 * byte_index) :=
                    new_value(8 * byte_index + 7 downto 8 * byte_index);
            end if;
        end loop;
        return merged;
    end function;

    signal internal_rst          : std_logic;

    signal aw_pending            : std_logic := '0';
    signal aw_address            : std_logic_vector(6 downto 0) := (others => '0');
    signal w_pending             : std_logic := '0';
    signal w_data                : std_logic_vector(31 downto 0) := (others => '0');
    signal w_strobe              : std_logic_vector(3 downto 0) := (others => '0');
    signal control_bvalid        : std_logic := '0';
    signal control_rvalid        : std_logic := '0';
    signal control_rdata         : std_logic_vector(31 downto 0) := (others => '0');

    signal ap_start_reg          : std_logic := '0';
    signal ap_done_reg           : std_logic := '0';
    signal ap_ready_reg          : std_logic := '0';
    signal auto_restart_reg      : std_logic := '0';
    signal gie_reg               : std_logic := '0';
    signal ier_reg               : std_logic_vector(1 downto 0) := (others => '0');
    signal isr_reg               : std_logic_vector(1 downto 0) := (others => '0');
    signal frame_address_reg     : std_logic_vector(63 downto 0) := (others => '0');
    signal results_address_reg   : std_logic_vector(63 downto 0) := (others => '0');

    signal state                 : mover_state_t := IDLE;
    signal frame_base            : unsigned(63 downto 0) := (others => '0');
    signal results_base          : unsigned(63 downto 0) := (others => '0');
    signal frame_beat            : integer range 0 to FRAME_BEATS - 1 := 0;
    signal pixel_slot            : integer range 0 to PIXELS_PER_BEAT - 1 := 0;
    signal read_word             : std_logic_vector(511 downto 0) := (others => '0');
    type frame_words_t is array(0 to 31) of std_logic_vector(511 downto 0);
    signal frame_words           : frame_words_t;
    signal frame_read_beat       : integer range 0 to FRAME_BEATS - 1 := 0;
    signal burst_beats, burst_remaining : integer range 1 to 16 := 1;
    attribute ram_style : string;
    attribute ram_style of frame_words : signal is "distributed";
    signal result_word           : std_logic_vector(511 downto 0) := (others => '0');
    signal result_seen           : std_logic_vector(SITE_COUNT - 1 downto 0) := (others => '0');
    signal write_address_sent    : std_logic := '0';
    signal write_data_sent       : std_logic := '0';
    signal completed_results     : integer range 0 to SITE_COUNT := 0;
    signal read_error            : std_logic := '0';
    signal write_error           : std_logic := '0';
    signal missing_result_error  : std_logic := '0';

    signal start_accepted        : std_logic := '0';
    signal kernel_complete       : std_logic := '0';
    signal ap_idle               : std_logic;

    signal frame_write_enable    : std_logic := '0';
    signal frame_write_addr      : integer range 0 to FRAME_PIXELS - 1 := 0;
    signal frame_write_data      : unsigned(RAW_W - 1 downto 0) := (others => '0');
    signal detector_start        : std_logic := '0';
    signal detector_busy         : std_logic;
    signal detector_result_valid : std_logic;
    signal detector_result_site  : integer range 0 to SITE_COUNT - 1;
    signal detector_logits       : signed_vector(0 to 1)(ACT_W_MAX - 1 downto 0);
    signal detector_decision     : std_logic;
    signal detector_done         : std_logic;
    signal detector_preprocess_cycles, detector_cnn_cycles, detector_pipeline_cycles : unsigned(31 downto 0);
begin
    internal_rst <= not ap_rst_n;
    ap_idle      <= '1' when state = IDLE else '0';

    s_axi_control_AWREADY <= not aw_pending and not control_bvalid;
    s_axi_control_WREADY  <= not w_pending and not control_bvalid;
    s_axi_control_BRESP   <= "00";
    s_axi_control_BVALID  <= control_bvalid;
    s_axi_control_ARREADY <= not control_rvalid;
    s_axi_control_RDATA   <= control_rdata;
    s_axi_control_RRESP   <= "00";
    s_axi_control_RVALID  <= control_rvalid;
    interrupt <= gie_reg and ((ier_reg(0) and isr_reg(0)) or (ier_reg(1) and isr_reg(1)));

    -- Single-beat result writes and incrementing frame-read bursts.
    m_axi_gmem_AWLEN   <= (others => '0');
    m_axi_gmem_AWSIZE  <= "110";
    m_axi_gmem_AWBURST <= "01";
    m_axi_gmem_AWLOCK  <= '0';
    m_axi_gmem_AWCACHE <= "0011";
    m_axi_gmem_AWPROT  <= "000";
    m_axi_gmem_AWQOS   <= (others => '0');
    -- Nine predictions and the original three counters with their format marker.
    m_axi_gmem_WSTRB <= (51 downto 0 => '1', others => '0');
    m_axi_gmem_WLAST   <= '1';
    m_axi_gmem_ARLEN   <= std_logic_vector(to_unsigned(burst_beats - 1, 8));
    -- 24 full 64-byte reads followed by one 32-byte read (16 pixels).
    m_axi_gmem_ARSIZE  <= "101" when frame_beat = FRAME_BEATS - 1 else "110";
    m_axi_gmem_ARBURST <= "01";
    m_axi_gmem_ARLOCK  <= '0';
    m_axi_gmem_ARCACHE <= "0011";
    m_axi_gmem_ARPROT  <= "000";
    m_axi_gmem_ARQOS   <= (others => '0');

    m_axi_gmem_ARADDR  <= std_logic_vector(frame_base + to_unsigned(frame_beat * 64, 64));
    m_axi_gmem_ARVALID <= '1' when state = FRAME_ADDRESS else '0';
    m_axi_gmem_RREADY  <= '1' when state = FRAME_DATA else '0';

    frame_burst : process(all)
        variable count, boundary_count, offset : integer;
    begin
        count := 1;
        if frame_beat < FRAME_BEATS - 1 then
            count := FRAME_BEATS - 1 - frame_beat;
            if count > 16 then count := 16; end if;
            offset := (to_integer(frame_base(11 downto 0)) + frame_beat * 64) mod 4096;
            boundary_count := (4096 - offset) / 64;
            if boundary_count < count then count := boundary_count; end if;
            if count < 1 then count := 1; end if;
        end if;
        burst_beats <= count;
    end process;

    -- Read the next word on the same edge that emits the current word's last pixel.
    frame_read_beat <= frame_beat + 1 when state = FRAME_UNPACK
        and pixel_slot = PIXELS_PER_BEAT - 1 and frame_beat < FRAME_BEATS - 1
        else frame_beat;

    frame_buffer : process(ap_clk)
    begin
        if rising_edge(ap_clk) then
            if state = FRAME_DATA and m_axi_gmem_RVALID = '1' then
                frame_words(frame_beat) <= m_axi_gmem_RDATA;
            end if;
            read_word <= frame_words(frame_read_beat);
        end if;
    end process;

    m_axi_gmem_AWADDR  <= std_logic_vector(results_base);
    m_axi_gmem_AWVALID <= '1' when state = RESULT_REQUEST and write_address_sent = '0' else '0';
    m_axi_gmem_WDATA   <= result_word;
    m_axi_gmem_WVALID  <= '1' when state = RESULT_REQUEST and write_data_sent = '0' else '0';
    m_axi_gmem_BREADY  <= '1' when state = RESULT_RESPONSE else '0';

    detector : detector_core
        generic map (
            WEIGHT_FILE => "weights.mem",
            BIAS_FILE   => "biases.mem"
        )
        port map (
            clk                 => ap_clk,
            rst                 => internal_rst,
            frame_write_enable  => frame_write_enable,
            frame_write_addr    => frame_write_addr,
            frame_write_data    => frame_write_data,
            start               => detector_start,
            busy                => detector_busy,
            result_valid        => detector_result_valid,
            result_site_index   => detector_result_site,
            result_logits       => detector_logits,
            result_decision     => detector_decision,
            preprocess_cycles   => detector_preprocess_cycles,
            cnn_cycles          => detector_cnn_cycles,
            pipeline_cycles     => detector_pipeline_cycles,
            done                => detector_done
        );

    control_registers : process (ap_clk)
        variable address_index : integer;
        variable merged        : std_logic_vector(31 downto 0);
        variable read_value    : std_logic_vector(31 downto 0);
    begin
        if rising_edge(ap_clk) then
            if ap_rst_n = '0' then
                aw_pending          <= '0';
                aw_address          <= (others => '0');
                w_pending           <= '0';
                w_data              <= (others => '0');
                w_strobe            <= (others => '0');
                control_bvalid      <= '0';
                control_rvalid      <= '0';
                control_rdata       <= (others => '0');
                ap_start_reg        <= '0';
                ap_done_reg         <= '0';
                ap_ready_reg        <= '0';
                auto_restart_reg    <= '0';
                gie_reg             <= '0';
                ier_reg             <= (others => '0');
                isr_reg             <= (others => '0');
                frame_address_reg   <= (others => '0');
                results_address_reg <= (others => '0');
            else
                ap_ready_reg <= '0';

                if s_axi_control_AWVALID = '1' and s_axi_control_AWREADY = '1' then
                    aw_pending <= '1';
                    aw_address <= s_axi_control_AWADDR;
                end if;
                if s_axi_control_WVALID = '1' and s_axi_control_WREADY = '1' then
                    w_pending <= '1';
                    w_data    <= s_axi_control_WDATA;
                    w_strobe  <= s_axi_control_WSTRB;
                end if;

                if aw_pending = '1' and w_pending = '1' and control_bvalid = '0' then
                    address_index := to_integer(unsigned(aw_address));
                    case address_index is
                        when ADDR_AP_CTRL =>
                            merged := merge_bytes((31 downto 8 => '0') & auto_restart_reg &
                                "000000" & ap_start_reg, w_data, w_strobe);
                            ap_start_reg     <= merged(0);
                            auto_restart_reg <= merged(7);
                        when ADDR_GIE =>
                            merged := merge_bytes((31 downto 1 => '0') & gie_reg, w_data, w_strobe);
                            gie_reg <= merged(0);
                        when ADDR_IER =>
                            merged := merge_bytes((31 downto 2 => '0') & ier_reg, w_data, w_strobe);
                            ier_reg <= merged(1 downto 0);
                        when ADDR_ISR =>
                            if w_strobe(0) = '1' then
                                isr_reg <= isr_reg xor w_data(1 downto 0);
                            end if;
                        when ADDR_FRAME_LO =>
                            frame_address_reg(31 downto 0) <=
                                merge_bytes(frame_address_reg(31 downto 0), w_data, w_strobe);
                        when ADDR_FRAME_HI =>
                            frame_address_reg(63 downto 32) <=
                                merge_bytes(frame_address_reg(63 downto 32), w_data, w_strobe);
                        when ADDR_RESULTS_LO =>
                            results_address_reg(31 downto 0) <=
                                merge_bytes(results_address_reg(31 downto 0), w_data, w_strobe);
                        when ADDR_RESULTS_HI =>
                            results_address_reg(63 downto 32) <=
                                merge_bytes(results_address_reg(63 downto 32), w_data, w_strobe);
                        when others =>
                            null;
                    end case;
                    aw_pending     <= '0';
                    w_pending      <= '0';
                    control_bvalid <= '1';
                end if;

                if control_bvalid = '1' and s_axi_control_BREADY = '1' then
                    control_bvalid <= '0';
                end if;

                if s_axi_control_ARVALID = '1' and s_axi_control_ARREADY = '1' then
                    address_index := to_integer(unsigned(s_axi_control_ARADDR));
                    read_value := (others => '0');
                    case address_index is
                        when ADDR_AP_CTRL =>
                            read_value(0) := ap_start_reg;
                            read_value(1) := ap_done_reg;
                            read_value(2) := ap_idle;
                            read_value(3) := ap_ready_reg;
                            read_value(7) := auto_restart_reg;
                            ap_done_reg <= '0';
                        when ADDR_GIE =>
                            read_value(0) := gie_reg;
                        when ADDR_IER =>
                            read_value(1 downto 0) := ier_reg;
                        when ADDR_ISR =>
                            read_value(1 downto 0) := isr_reg;
                        when ADDR_FRAME_LO =>
                            read_value := frame_address_reg(31 downto 0);
                        when ADDR_FRAME_HI =>
                            read_value := frame_address_reg(63 downto 32);
                        when ADDR_RESULTS_LO =>
                            read_value := results_address_reg(31 downto 0);
                        when ADDR_RESULTS_HI =>
                            read_value := results_address_reg(63 downto 32);
                        when ADDR_STATUS =>
                            read_value(0) := read_error;
                            read_value(1) := write_error;
                            read_value(2) := missing_result_error;
                            read_value(8) := detector_busy;
                        when ADDR_RESULTS_DONE =>
                            read_value(3 downto 0) := std_logic_vector(to_unsigned(completed_results, 4));
                        when others =>
                            null;
                    end case;
                    control_rdata  <= read_value;
                    control_rvalid <= '1';
                end if;

                if control_rvalid = '1' and s_axi_control_RREADY = '1' then
                    control_rvalid <= '0';
                end if;

                if start_accepted = '1' then
                    ap_start_reg <= '0';
                    ap_ready_reg <= '1';
                    if ier_reg(1) = '1' then
                        isr_reg(1) <= '1';
                    end if;
                end if;

                if kernel_complete = '1' then
                    ap_done_reg <= '1';
                    if auto_restart_reg = '1' then
                        ap_start_reg <= '1';
                    end if;
                    if ier_reg(0) = '1' then
                        isr_reg(0) <= '1';
                    end if;
                end if;
            end if;
        end if;
    end process;

    data_mover : process (ap_clk)
        variable packed_result : std_logic_vector(31 downto 0);
        variable saw_all       : std_logic_vector(SITE_COUNT - 1 downto 0);
        variable address_seen  : boolean;
        variable data_seen     : boolean;
    begin
        if rising_edge(ap_clk) then
            if ap_rst_n = '0' then
                state                <= IDLE;
                frame_base           <= (others => '0');
                results_base         <= (others => '0');
                frame_beat           <= 0;
                pixel_slot           <= 0;
                burst_remaining      <= 1;
                result_word          <= (others => '0');
                result_seen          <= (others => '0');
                write_address_sent   <= '0';
                write_data_sent      <= '0';
                completed_results    <= 0;
                read_error           <= '0';
                write_error          <= '0';
                missing_result_error <= '0';
                start_accepted       <= '0';
                kernel_complete      <= '0';
                frame_write_enable   <= '0';
                frame_write_addr     <= 0;
                frame_write_data     <= (others => '0');
                detector_start       <= '0';
            else
                start_accepted     <= '0';
                kernel_complete    <= '0';
                frame_write_enable <= '0';
                detector_start     <= '0';

                if state = DETECTOR_WAIT and detector_result_valid = '1' then
                    packed_result := (others => '0');
                    packed_result(9 downto 0)   := std_logic_vector(resize(detector_logits(0), 10));
                    packed_result(19 downto 10) := std_logic_vector(resize(detector_logits(1), 10));
                    packed_result(20)           := detector_decision;
                    result_word(detector_result_site * 32 + 31 downto detector_result_site * 32)
                        <= packed_result;
                    result_seen(detector_result_site) <= '1';
                end if;

                case state is
                    when IDLE =>
                        if ap_start_reg = '1' then
                            frame_base           <= unsigned(frame_address_reg);
                            results_base         <= unsigned(results_address_reg);
                            frame_beat           <= 0;
                            pixel_slot           <= 0;
                            result_word          <= (others => '0');
                            result_seen          <= (others => '0');
                            completed_results    <= 0;
                            read_error           <= '0';
                            write_error          <= '0';
                            missing_result_error <= '0';
                            start_accepted       <= '1';
                            state                <= FRAME_ADDRESS;
                        end if;

                    when FRAME_ADDRESS =>
                        if m_axi_gmem_ARREADY = '1' then
                            burst_remaining <= burst_beats;
                            state <= FRAME_DATA;
                        end if;

                    when FRAME_DATA =>
                        if m_axi_gmem_RVALID = '1' then
                            if m_axi_gmem_RRESP /= "00"
                                or (burst_remaining = 1 and m_axi_gmem_RLAST /= '1')
                                or (burst_remaining > 1 and m_axi_gmem_RLAST /= '0')
                            then
                                read_error <= '1';
                            end if;
                            if burst_remaining > 1 then
                                burst_remaining <= burst_remaining - 1;
                                frame_beat <= frame_beat + 1;
                            elsif frame_beat = FRAME_BEATS - 1 then
                                frame_beat <= 0;
                                pixel_slot <= 0;
                                state <= FRAME_PRIME;
                            else
                                frame_beat <= frame_beat + 1;
                                state <= FRAME_ADDRESS;
                            end if;
                        end if;

                    when FRAME_PRIME =>
                        state <= FRAME_UNPACK;

                    when FRAME_UNPACK =>
                        frame_write_enable <= '1';
                        frame_write_addr <= frame_beat * PIXELS_PER_BEAT + pixel_slot;
                        frame_write_data <= unsigned(
                            read_word(pixel_slot * 16 + RAW_W - 1 downto pixel_slot * 16)
                        );
                        if frame_beat * PIXELS_PER_BEAT + pixel_slot = FRAME_PIXELS - 1 then
                            pixel_slot <= 0;
                            state <= START_DETECTOR;
                        elsif pixel_slot = PIXELS_PER_BEAT - 1 then
                            pixel_slot <= 0;
                            frame_beat <= frame_beat + 1;
                        else
                            pixel_slot <= pixel_slot + 1;
                        end if;

                    when START_DETECTOR =>
                        detector_start <= '1';
                        state <= DETECTOR_WAIT;

                    when DETECTOR_WAIT =>
                        if detector_done = '1' then
                            result_word(319 downto 288) <= std_logic_vector(detector_preprocess_cycles);
                            result_word(351 downto 320) <= std_logic_vector(detector_cnn_cycles);
                            result_word(383 downto 352) <= std_logic_vector(detector_pipeline_cycles);
                            result_word(415 downto 384) <= x"43594331";
                            saw_all := result_seen;
                            if detector_result_valid = '1' then
                                saw_all(detector_result_site) := '1';
                            end if;
                            if saw_all /= (saw_all'range => '1') then
                                missing_result_error <= '1';
                            end if;
                            write_address_sent <= '0';
                            write_data_sent    <= '0';
                            state              <= RESULT_REQUEST;
                        end if;

                    when RESULT_REQUEST =>
                        address_seen := write_address_sent = '1';
                        data_seen    := write_data_sent = '1';
                        if m_axi_gmem_AWREADY = '1' and write_address_sent = '0' then
                            write_address_sent <= '1';
                            address_seen := true;
                        end if;
                        if m_axi_gmem_WREADY = '1' and write_data_sent = '0' then
                            write_data_sent <= '1';
                            data_seen := true;
                        end if;
                        if address_seen and data_seen then
                            state <= RESULT_RESPONSE;
                        end if;

                    when RESULT_RESPONSE =>
                        if m_axi_gmem_BVALID = '1' then
                            if m_axi_gmem_BRESP /= "00" then
                                write_error <= '1';
                            end if;
                            completed_results <= SITE_COUNT;
                            state <= COMPLETE;
                        end if;

                    when COMPLETE =>
                        kernel_complete <= '1';
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;
end architecture;
