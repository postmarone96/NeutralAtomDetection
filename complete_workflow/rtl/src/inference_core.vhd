library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.activation_map_pkg.all;
use work.bias_pkg.all;
use work.helpers_pkg.all;
use work.nn_pkg.all;
use work.weights_pkg.all;

entity inference_core is
    generic (
        WEIGHT_FILE : string := "src/weights.mem";
        BIAS_FILE   : string := "src/biases.mem"
    );
    port (
        clk           : in std_logic;
        rst           : in std_logic;
        load_input    : in std_logic;
        start         : in std_logic;
        input_data    : in signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        neighbor_data : in signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);

        busy          : out std_logic;
        result_valid  : out std_logic;
        result_data   : out signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
        decision      : out std_logic
    );
end entity;

architecture rtl of inference_core is
    signal controller_done              : std_logic;
    signal conv_enable                  : std_logic;
    signal layer_index                  : integer range 0 to N_LAYERS - 1;
    signal conv_index                   : integer range 0 to N_CONV - 1 := 0;
    signal input_index                  : integer range 0 to MAX_LAYER_INPUT_COUNT - 1;
    signal conv_input_channel           : integer range 0 to MAX_CHANNEL - 1 := 0;
    signal conv_tap_index               : integer range 0 to KERNEL_TAP_COUNT - 1 := 0;
    signal output_block_index           : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1;
    signal tap_index                    : integer range 0 to KERNEL_TAP_COUNT - 1;

    signal controller_read_enable       : std_logic;
    signal controller_write_enable      : std_logic;
    signal controller_weight_enable     : std_logic;
    signal controller_load_bias         : std_logic;
    signal controller_mac_valid         : std_logic;
    signal controller_block_valid       : std_logic;

    signal ram_write_enable             : std_logic;
    signal ram_write_data               : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal ram_read_valid               : std_logic;
    signal ram_read_data                : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal ram_neighbor_data            : signed_vector(0 to NEIGHBORS - 1)(NB_W - 1 downto 0);

    signal conv_read_valid              : std_logic;
    signal conv_activation_valid        : std_logic;
    signal conv_activation_data         : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal dense_read_valid             : std_logic;
    signal dense_activation_valid       : std_logic;
    signal dense_activation_data        : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal selected_activation_valid    : std_logic;
    signal selected_activation_data     : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

    signal weight_rom_enable            : std_logic;
    signal weight_rom_valid             : std_logic;
    signal weight_rom_row               : integer range 0 to WEIGHT_ROM_DEPTH - 1;
    signal weight_rom_data              : signed(WEIGHT_ROM_BITS - 1 downto 0);
    signal mac_weights                  : signed_vector(0 to MAX_MAC_UNITS - 1)(WEIGHT_W_MAX - 1 downto 0);
    signal mac_weights_valid            : std_logic;

    signal bias_rom_row                 : integer range 0 to BIAS_ROM_DEPTH - 1;
    signal bias_rom_data                : signed(BIAS_ROM_BITS - 1 downto 0);
    signal mac_biases                   : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
    signal mac_accumulators             : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0);
    signal mac_busy                     : std_logic;
    -- Register the completed MAC block and its identifying metadata.
    signal pipeline_mac_accumulators    : signed_vector(0 to MAX_MAC_UNITS - 1)(ACC_W_MAX - 1 downto 0)
                                        := (others => (others => '0'));
    signal pipeline_mac_valid           : std_logic := '0';
    signal pipeline_mac_layer           : integer range 0 to N_LAYERS - 1 := 0;
    signal pipeline_mac_block           : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1 := 0;

    signal requant_valid                : std_logic;
    signal requant_data                 : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

    -- Register the requantized block and the same metadata.
    signal pipeline_requant_valid       : std_logic := '0';
    signal pipeline_requant_data        : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                                        := (others => (others => '0'));
    signal pipeline_requant_layer       : integer range 0 to N_LAYERS - 1 := 0;
    signal pipeline_requant_block       : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1 := 0;
    signal pipeline_pool_conv_index     : integer range 0 to N_CONV - 1 := 0;

    signal pooled_data                  : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);
    signal postprocessed_data           : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

    signal pipeline_output_valid        : std_logic := '0';
    signal pipeline_output_data         : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0)
                                        := (others => (others => '0'));
    signal pipeline_output_layer        : integer range 0 to N_LAYERS - 1 := 0;
    signal pipeline_output_block        : integer range 0 to MAX_OUTPUT_BLOCK_COUNT - 1 := 0;

    signal assembled_valid              : std_logic;
    signal assembled_data               : signed_vector(0 to MAX_MAC_UNITS - 1)(ACT_W_MAX - 1 downto 0);

    -- Preserve the data/valid pipeline boundaries. Keep layer/block metadata
    -- through synthesis, but allow physical optimization to copy its drivers.
    attribute KEEP       : string;
    attribute DONT_TOUCH : string;
    attribute DONT_TOUCH of conv_index, conv_input_channel, conv_tap_index : signal is "TRUE";

    attribute KEEP of pipeline_mac_accumulators : signal is "TRUE";
    attribute KEEP of pipeline_mac_valid        : signal is "TRUE";
    attribute KEEP of pipeline_mac_layer        : signal is "TRUE";
    attribute KEEP of pipeline_mac_block        : signal is "TRUE";
    attribute KEEP of pipeline_requant_data     : signal is "TRUE";
    attribute KEEP of pipeline_requant_valid    : signal is "TRUE";
    attribute KEEP of pipeline_requant_layer    : signal is "TRUE";
    attribute KEEP of pipeline_requant_block    : signal is "TRUE";
    attribute KEEP of pipeline_output_data      : signal is "TRUE";
    attribute KEEP of pipeline_output_valid     : signal is "TRUE";
    attribute KEEP of pipeline_output_layer     : signal is "TRUE";
    attribute KEEP of pipeline_output_block     : signal is "TRUE";

    attribute DONT_TOUCH of pipeline_mac_accumulators : signal is "TRUE";
    attribute DONT_TOUCH of pipeline_mac_valid        : signal is "TRUE";
    attribute DONT_TOUCH of pipeline_requant_data     : signal is "TRUE";
    attribute DONT_TOUCH of pipeline_requant_valid    : signal is "TRUE";
    attribute DONT_TOUCH of pipeline_output_data      : signal is "TRUE";
    attribute DONT_TOUCH of pipeline_output_valid     : signal is "TRUE";
begin
    process (clk)
    begin
        if rising_edge(clk) then
            if rst = '0' and controller_read_enable = '1' and conv_enable = '1' then
                assert layer_index < N_CONV and input_index < MAX_CHANNEL
                    report "invalid convolution activation request"
                    severity failure;
                conv_index         <= layer_index;
                conv_input_channel <= input_index;
                conv_tap_index     <= tap_index;
            end if;
        end if;
    end process;

    ram_write_enable  <= load_input or controller_write_enable;
    ram_write_data    <= input_data when load_input = '1' else assembled_data;

    activation_memory : entity work.act_ram
        port map (
            clk             => clk,
            write_data      => ram_write_data,
            write_enable    => ram_write_enable,
            read_enable     => controller_read_enable,
            read_valid      => ram_read_valid,
            read_data       => ram_read_data,
            nb_in           => neighbor_data,
            nb_out          => ram_neighbor_data,
            nb_load         => load_input
        );

    conv_read_valid  <= ram_read_valid when conv_enable = '1' else '0';
    dense_read_valid <= ram_read_valid when conv_enable = '0' else '0';

    convolution_input : entity work.conv_input
        port map (
            clk              => clk,
            rst              => rst,
            read_valid       => conv_read_valid,
            read_data        => ram_read_data,
            input_channel    => conv_input_channel,
            conv_index       => conv_index,
            tap_index        => conv_tap_index,
            conv_input_valid => conv_activation_valid,
            conv_input       => conv_activation_data
        );

    fully_connected_input : entity work.dense_input
        port map (
            clk               => clk,
            rst               => rst,
            read_valid        => dense_read_valid,
            read_data         => ram_read_data,
            neighbor_data     => ram_neighbor_data,
            layer_index       => layer_index,
            input_index       => input_index,
            dense_input_valid => dense_activation_valid,
            dense_input_data  => dense_activation_data
        );

    selected_activation_valid <= conv_activation_valid when conv_enable = '1' else dense_activation_valid;
    selected_activation_data <= conv_activation_data when conv_enable = '1' else dense_activation_data;

    weight_router : entity work.weights_handler
        port map (
            clk                => clk,
            conv_enable        => conv_enable,
            layer_index        => layer_index,
            input_index        => input_index,
            output_block_index => output_block_index,
            tap_index          => tap_index,
            read_enable        => controller_weight_enable,
            rom_read_enable    => weight_rom_enable,
            read_valid         => weight_rom_valid,
            rom_row            => weight_rom_row,
            rom_row_data       => weight_rom_data,
            weights            => mac_weights,
            weights_valid      => mac_weights_valid
        );

    weights_memory : entity work.weight_rom
        generic map (
            ROM_DEPTH   => WEIGHT_ROM_DEPTH,
            WORD_LENGTH => WEIGHT_ROM_BITS,
            FILE_NAME   => WEIGHT_FILE
        )
        port map (
            clk         => clk,
            read_enable => weight_rom_enable,
            addr        => weight_rom_row,
            read_valid  => weight_rom_valid,
            data_out    => weight_rom_data
        );

    bias_router : entity work.biases_handler
        port map (
            conv_enable        => conv_enable,
            layer_index        => layer_index,
            output_block_index => output_block_index,
            bias_row           => bias_rom_row,
            bias_rom_data      => bias_rom_data,
            bias_out           => mac_biases
        );

    biases_memory : entity work.bias_rom
        generic map (
            ROM_DEPTH   => BIAS_ROM_DEPTH,
            WORD_LENGTH => BIAS_ROM_BITS,
            FILE_NAME   => BIAS_FILE
        )
        port map (
            addr     => bias_rom_row,
            data_out => bias_rom_data
        );

    multipliers : entity work.mac_array
        generic map (
            ACT_W_MAX     => ACT_W_MAX,
            WEIGHT_W_MAX  => WEIGHT_W_MAX,
            ACC_W_MAX     => ACC_W_MAX,
            MAX_MAC_UNITS => MAX_MAC_UNITS,
            MAC_UNITS     => MAX_MAC_UNITS
        )
        port map (
            clk         => clk,
            rst         => rst,
            valid       => controller_mac_valid,
            load_bias   => controller_load_bias,
            bias        => mac_biases,
            activation  => selected_activation_data,
            weight      => mac_weights,
            busy        => mac_busy,
            accum       => mac_accumulators
        );

    requantizer : entity work.requant_array
        port map (
            layer_index     => pipeline_mac_layer,
            input_valid     => pipeline_mac_valid,
            accum_data      => pipeline_mac_accumulators,
            output_valid    => requant_valid,
            activation_data => requant_data
        );

    pipeline_pool_conv_index <= pipeline_requant_layer
        when pipeline_requant_layer < N_CONV else 0;

    pooling : entity work.max_pooling
        port map (
            conv_index     => pipeline_pool_conv_index,
            activation_in  => pipeline_requant_data,
            activation_out => pooled_data
        );

    postprocessed_data <= pooled_data
        when pipeline_requant_layer < N_CONV else pipeline_requant_data;

    output_assembler : entity work.layer_output
        port map (
            clk                => clk,
            rst                => rst,
            input_valid        => pipeline_output_valid,
            layer_index        => pipeline_output_layer,
            output_block_index => pipeline_output_block,
            input_data         => pipeline_output_data,
            output_valid       => assembled_valid,
            output_data        => assembled_data
        );

    controller : entity work.inference_controller
        port map (
            clk                      => clk,
            rst                      => rst,
            start                    => start,
            activation_valid         => selected_activation_valid,
            weights_valid            => mac_weights_valid,
            layer_output_valid       => assembled_valid,
            mac_busy                 => mac_busy,
            busy                     => busy,
            done                     => controller_done,
            result_valid             => result_valid,
            conv_enable              => conv_enable,
            layer_index              => layer_index,
            input_index              => input_index,
            output_block_index       => output_block_index,
            tap_index                => tap_index,
            activation_read_enable   => controller_read_enable,
            activation_write_enable  => controller_write_enable,
            weight_read_enable       => controller_weight_enable,
            mac_load_bias            => controller_load_bias,
            mac_valid                => controller_mac_valid,
            layer_output_input_valid => controller_block_valid
        );

    result_data <= assembled_data;
    decision    <= '1' when assembled_data(1) > assembled_data(0) else '0';

    timing_pipeline : process (clk, rst)
    begin
        if rst = '1' then
            pipeline_mac_accumulators <= (others => (others => '0'));
            pipeline_mac_valid        <= '0';
            pipeline_mac_layer        <= 0;
            pipeline_mac_block        <= 0;

            pipeline_requant_data  <= (others => (others => '0'));
            pipeline_requant_valid <= '0';
            pipeline_requant_layer <= 0;
            pipeline_requant_block <= 0;

            pipeline_output_data  <= (others => (others => '0'));
            pipeline_output_valid <= '0';
            pipeline_output_layer <= 0;
            pipeline_output_block <= 0;
        elsif rising_edge(clk) then
            -- Stage 1: isolate the MAC accumulators from requantization.
            pipeline_mac_valid <= controller_block_valid;
            if controller_block_valid = '1' then
                pipeline_mac_accumulators <= mac_accumulators;
                pipeline_mac_layer        <= layer_index;
                pipeline_mac_block        <= output_block_index;
            end if;

            -- Stage 2: isolate requantization from pooling.
            pipeline_requant_valid <= requant_valid;
            if requant_valid = '1' then
                pipeline_requant_data  <= requant_data;
                pipeline_requant_layer <= pipeline_mac_layer;
                pipeline_requant_block <= pipeline_mac_block;
            end if;

            -- Stage 3: register pooled convolution data or bypassed dense data.
            pipeline_output_valid <= pipeline_requant_valid;
            if pipeline_requant_valid = '1' then
                pipeline_output_data  <= postprocessed_data;
                pipeline_output_layer <= pipeline_requant_layer;
                pipeline_output_block <= pipeline_requant_block;
            end if;
        end if;
    end process;
end architecture;
