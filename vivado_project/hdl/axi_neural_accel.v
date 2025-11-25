// AXI4-Lite Wrapper for Neural Network Accelerator
// Provides AXI interface for control and status registers

module axi_neural_accel #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 16,
    parameter integer C_M_AXI_DATA_WIDTH = 64,
    parameter integer C_M_AXI_ADDR_WIDTH = 32
)(
    // Clock and Reset
    input wire s_axi_aclk,
    input wire s_axi_aresetn,
    input wire m_axi_aclk,
    input wire m_axi_aresetn,
    
    // AXI4-Lite Slave Interface
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input wire [2:0] s_axi_awprot,
    input wire s_axi_awvalid,
    output wire s_axi_awready,
    input wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input wire s_axi_wvalid,
    output wire s_axi_wready,
    output wire [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input wire [2:0] s_axi_arprot,
    input wire s_axi_arvalid,
    output wire s_axi_arready,
    output wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output wire [1:0] s_axi_rresp,
    output wire s_axi_rvalid,
    input wire s_axi_rready,
    
    // AXI4 Master Interface for Memory Access
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire m_axi_awlock,
    output wire [3:0] m_axi_awcache,
    output wire [2:0] m_axi_awprot,
    output wire [3:0] m_axi_awqos,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output wire [(C_M_AXI_DATA_WIDTH/8)-1:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready,
    output wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arlock,
    output wire [3:0] m_axi_arcache,
    output wire [2:0] m_axi_arprot,
    output wire [3:0] m_axi_arqos,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    
    // Interrupt output
    output wire interrupt
);

    // Register map
    localparam ADDR_CONTROL     = 16'h0000;  // Control register
    localparam ADDR_STATUS      = 16'h0004;  // Status register
    localparam ADDR_OP_TYPE     = 16'h0008;  // Operation type
    localparam ADDR_INPUT_H     = 16'h000C;  // Input height
    localparam ADDR_INPUT_W     = 16'h0010;  // Input width
    localparam ADDR_INPUT_C     = 16'h0014;  // Input channels
    localparam ADDR_OUTPUT_C    = 16'h0018;  // Output channels
    localparam ADDR_KERNEL_SIZE = 16'h001C;  // Kernel size
    localparam ADDR_STRIDE      = 16'h0020;  // Stride
    localparam ADDR_PADDING     = 16'h0024;  // Padding
    localparam ADDR_INPUT_ADDR  = 16'h0028;  // Input data address
    localparam ADDR_WEIGHT_ADDR = 16'h002C;  // Weight data address
    localparam ADDR_OUTPUT_ADDR = 16'h0030;  // Output data address
    localparam ADDR_CYCLE_COUNT = 16'h0034;  // Cycle counter
    localparam ADDR_OP_COUNT    = 16'h0038;  // Operation counter
    localparam ADDR_VERSION     = 16'h003C;  // Version register

    // Control and status registers
    reg [31:0] control_reg;
    reg [31:0] status_reg;
    reg [7:0] operation_type;
    reg [15:0] input_height, input_width, input_channels, output_channels;
    reg [7:0] kernel_size, stride, padding;
    reg [31:0] input_base_addr, weight_base_addr, output_base_addr;
    
    // AXI4-Lite signals
    reg axi_awready;
    reg axi_wready;
    reg [1:0] axi_bresp;
    reg axi_bvalid;
    reg axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0] axi_rresp;
    reg axi_rvalid;
    
    // Internal signals
    wire start_operation;
    wire accel_done, accel_busy;
    wire [31:0] cycle_count, op_count;
    
    // Neural accelerator signals
    wire [31:0] accel_input_addr, accel_weight_addr, accel_output_addr;
    wire [31:0] accel_input_data, accel_weight_data, accel_output_data;
    wire accel_input_valid, accel_weight_valid, accel_output_valid;
    wire accel_input_ready, accel_weight_ready, accel_output_ready;

    // AXI4-Lite interface implementation
    assign s_axi_awready = axi_awready;
    assign s_axi_wready = axi_wready;
    assign s_axi_bresp = axi_bresp;
    assign s_axi_bvalid = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata = axi_rdata;
    assign s_axi_rresp = axi_rresp;
    assign s_axi_rvalid = axi_rvalid;

    // Control logic
    assign start_operation = control_reg[0];
    
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            control_reg <= 32'h0;
            status_reg <= 32'h0;
            operation_type <= 8'h0;
            input_height <= 16'h0;
            input_width <= 16'h0;
            input_channels <= 16'h0;
            output_channels <= 16'h0;
            kernel_size <= 8'h0;
            stride <= 8'h1;
            padding <= 8'h0;
            input_base_addr <= 32'h0;
            weight_base_addr <= 32'h0;
            output_base_addr <= 32'h0;
        end else begin
            // Update status register
            status_reg[0] <= accel_done;
            status_reg[1] <= accel_busy;
            status_reg[31:2] <= 30'h0;
            
            // Clear start bit when operation begins
            if (start_operation && accel_busy) begin
                control_reg[0] <= 1'b0;
            end
        end
    end

    // AXI4-Lite write process
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            axi_wready <= 1'b0;
            axi_bvalid <= 1'b0;
            axi_bresp <= 2'b0;
        end else begin
            if (~axi_awready && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready <= 1'b1;
                axi_wready <= 1'b1;
            end else begin
                axi_awready <= 1'b0;
                axi_wready <= 1'b0;
            end
            
            if (axi_awready && s_axi_awvalid && axi_wready && s_axi_wvalid && ~axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp <= 2'b0; // OKAY response
                
                // Write to registers
                case (s_axi_awaddr)
                    ADDR_CONTROL: begin
                        if (s_axi_wstrb[0]) control_reg[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) control_reg[15:8] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) control_reg[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) control_reg[31:24] <= s_axi_wdata[31:24];
                    end
                    ADDR_OP_TYPE: begin
                        if (s_axi_wstrb[0]) operation_type <= s_axi_wdata[7:0];
                    end
                    ADDR_INPUT_H: begin
                        if (s_axi_wstrb[0]) input_height[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) input_height[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_INPUT_W: begin
                        if (s_axi_wstrb[0]) input_width[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) input_width[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_INPUT_C: begin
                        if (s_axi_wstrb[0]) input_channels[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) input_channels[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_OUTPUT_C: begin
                        if (s_axi_wstrb[0]) output_channels[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) output_channels[15:8] <= s_axi_wdata[15:8];
                    end
                    ADDR_KERNEL_SIZE: begin
                        if (s_axi_wstrb[0]) kernel_size <= s_axi_wdata[7:0];
                    end
                    ADDR_STRIDE: begin
                        if (s_axi_wstrb[0]) stride <= s_axi_wdata[7:0];
                    end
                    ADDR_PADDING: begin
                        if (s_axi_wstrb[0]) padding <= s_axi_wdata[7:0];
                    end
                    ADDR_INPUT_ADDR: begin
                        if (s_axi_wstrb[0]) input_base_addr[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) input_base_addr[15:8] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) input_base_addr[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) input_base_addr[31:24] <= s_axi_wdata[31:24];
                    end
                    ADDR_WEIGHT_ADDR: begin
                        if (s_axi_wstrb[0]) weight_base_addr[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) weight_base_addr[15:8] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) weight_base_addr[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) weight_base_addr[31:24] <= s_axi_wdata[31:24];
                    end
                    ADDR_OUTPUT_ADDR: begin
                        if (s_axi_wstrb[0]) output_base_addr[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) output_base_addr[15:8] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) output_base_addr[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) output_base_addr[31:24] <= s_axi_wdata[31:24];
                    end
                endcase
            end else if (axi_bvalid && s_axi_bready) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI4-Lite read process
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_rvalid <= 1'b0;
            axi_rresp <= 2'b0;
            axi_rdata <= 32'h0;
        end else begin
            if (~axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
                axi_rvalid <= 1'b1;
                axi_rresp <= 2'b0; // OKAY response
                
                // Read from registers
                case (s_axi_araddr)
                    ADDR_CONTROL: axi_rdata <= control_reg;
                    ADDR_STATUS: axi_rdata <= status_reg;
                    ADDR_OP_TYPE: axi_rdata <= {24'h0, operation_type};
                    ADDR_INPUT_H: axi_rdata <= {16'h0, input_height};
                    ADDR_INPUT_W: axi_rdata <= {16'h0, input_width};
                    ADDR_INPUT_C: axi_rdata <= {16'h0, input_channels};
                    ADDR_OUTPUT_C: axi_rdata <= {16'h0, output_channels};
                    ADDR_KERNEL_SIZE: axi_rdata <= {24'h0, kernel_size};
                    ADDR_STRIDE: axi_rdata <= {24'h0, stride};
                    ADDR_PADDING: axi_rdata <= {24'h0, padding};
                    ADDR_INPUT_ADDR: axi_rdata <= input_base_addr;
                    ADDR_WEIGHT_ADDR: axi_rdata <= weight_base_addr;
                    ADDR_OUTPUT_ADDR: axi_rdata <= output_base_addr;
                    ADDR_CYCLE_COUNT: axi_rdata <= cycle_count;
                    ADDR_OP_COUNT: axi_rdata <= op_count;
                    ADDR_VERSION: axi_rdata <= 32'h00010000; // Version 1.0
                    default: axi_rdata <= 32'h0;
                endcase
            end else if (axi_rvalid && s_axi_rready) begin
                axi_arready <= 1'b0;
                axi_rvalid <= 1'b0;
            end
        end
    end

    // Neural Network Accelerator Instance
    neural_accelerator #(
        .DATA_WIDTH(32),
        .ADDR_WIDTH(32),
        .PE_COUNT(16),
        .BUFFER_DEPTH(1024)
    ) u_neural_accel (
        .clk(s_axi_aclk),
        .resetn(s_axi_aresetn),
        
        // Control interface
        .start(start_operation),
        .operation_type(operation_type),
        .input_height(input_height),
        .input_width(input_width),
        .input_channels(input_channels),
        .output_channels(output_channels),
        .kernel_size(kernel_size),
        .stride(stride),
        .padding(padding),
        
        // Status outputs
        .done(accel_done),
        .busy(accel_busy),
        .cycle_count(cycle_count),
        .operation_count(op_count),
        
        // Memory interfaces
        .input_addr(accel_input_addr),
        .input_data(accel_input_data),
        .input_valid(accel_input_valid),
        .input_ready(accel_input_ready),
        
        .weight_addr(accel_weight_addr),
        .weight_data(accel_weight_data),
        .weight_valid(accel_weight_valid),
        .weight_ready(accel_weight_ready),
        
        .output_addr(accel_output_addr),
        .output_data(accel_output_data),
        .output_valid(accel_output_valid),
        .output_ready(accel_output_ready),
        
        .interrupt(interrupt)
    );

    // AXI4 Master Interface for Memory Access
    axi4_master_neural #(
        .C_M_AXI_DATA_WIDTH(C_M_AXI_DATA_WIDTH),
        .C_M_AXI_ADDR_WIDTH(C_M_AXI_ADDR_WIDTH)
    ) u_axi_master (
        .M_AXI_ACLK(m_axi_aclk),
        .M_AXI_ARESETN(m_axi_aresetn),
        
        // AXI4 Master Interface
        .M_AXI_AWADDR(m_axi_awaddr),
        .M_AXI_AWLEN(m_axi_awlen),
        .M_AXI_AWSIZE(m_axi_awsize),
        .M_AXI_AWBURST(m_axi_awburst),
        .M_AXI_AWLOCK(m_axi_awlock),
        .M_AXI_AWCACHE(m_axi_awcache),
        .M_AXI_AWPROT(m_axi_awprot),
        .M_AXI_AWQOS(m_axi_awqos),
        .M_AXI_AWVALID(m_axi_awvalid),
        .M_AXI_AWREADY(m_axi_awready),
        .M_AXI_WDATA(m_axi_wdata),
        .M_AXI_WSTRB(m_axi_wstrb),
        .M_AXI_WLAST(m_axi_wlast),
        .M_AXI_WVALID(m_axi_wvalid),
        .M_AXI_WREADY(m_axi_wready),
        .M_AXI_BRESP(m_axi_bresp),
        .M_AXI_BVALID(m_axi_bvalid),
        .M_AXI_BREADY(m_axi_bready),
        .M_AXI_ARADDR(m_axi_araddr),
        .M_AXI_ARLEN(m_axi_arlen),
        .M_AXI_ARSIZE(m_axi_arsize),
        .M_AXI_ARBURST(m_axi_arburst),
        .M_AXI_ARLOCK(m_axi_arlock),
        .M_AXI_ARCACHE(m_axi_arcache),
        .M_AXI_ARPROT(m_axi_arprot),
        .M_AXI_ARQOS(m_axi_arqos),
        .M_AXI_ARVALID(m_axi_arvalid),
        .M_AXI_ARREADY(m_axi_arready),
        .M_AXI_RDATA(m_axi_rdata),
        .M_AXI_RRESP(m_axi_rresp),
        .M_AXI_RLAST(m_axi_rlast),
        .M_AXI_RVALID(m_axi_rvalid),
        .M_AXI_RREADY(m_axi_rready),
        
        // Neural accelerator interface
        .input_base_addr(input_base_addr),
        .weight_base_addr(weight_base_addr),
        .output_base_addr(output_base_addr),
        
        .input_addr(accel_input_addr),
        .input_data(accel_input_data),
        .input_valid(accel_input_valid),
        .input_ready(accel_input_ready),
        
        .weight_addr(accel_weight_addr),
        .weight_data(accel_weight_data),
        .weight_valid(accel_weight_valid),
        .weight_ready(accel_weight_ready),
        
        .output_addr(accel_output_addr),
        .output_data(accel_output_data),
        .output_valid(accel_output_valid),
        .output_ready(accel_output_ready)
    );
    
    // Interrupt from neural accelerator
    assign interrupt = 1'b0;

endmodule