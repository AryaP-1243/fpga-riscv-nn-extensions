module axi4_master_neural #(
    parameter integer C_M_AXI_DATA_WIDTH = 64,
    parameter integer C_M_AXI_ADDR_WIDTH = 32
) (
    input  wire                         M_AXI_ACLK,
    input  wire                         M_AXI_ARESETN,
    
    // Base addresses from axi_neural_accel
    input  wire [31:0]                  input_base_addr,
    input  wire [31:0]                  weight_base_addr,
    input  wire [31:0]                  output_base_addr,
    
    // Input interface
    output wire [31:0]                  input_addr,
    input  wire [31:0]                  input_data,
    input  wire                         input_valid,
    output wire                         input_ready,
    
    // Weight interface
    output wire [31:0]                  weight_addr,
    input  wire [31:0]                  weight_data,
    input  wire                         weight_valid,
    output wire                         weight_ready,
    
    // Output interface
    output wire [31:0]                  output_addr,
    output wire [31:0]                  output_data,
    output wire                         output_valid,
    input  wire                         output_ready,
    
    // AXI4 Master
    output wire [C_M_AXI_ADDR_WIDTH-1:0] M_AXI_AWADDR,
    output wire [7:0]                   M_AXI_AWLEN,
    output wire [2:0]                   M_AXI_AWSIZE,
    output wire [1:0]                   M_AXI_AWBURST,
    output wire                         M_AXI_AWLOCK,
    output wire [3:0]                   M_AXI_AWCACHE,
    output wire [2:0]                   M_AXI_AWPROT,
    output wire [3:0]                   M_AXI_AWQOS,
    output wire                         M_AXI_AWVALID,
    input  wire                         M_AXI_AWREADY,
    
    output wire [C_M_AXI_DATA_WIDTH-1:0] M_AXI_WDATA,
    output wire [(C_M_AXI_DATA_WIDTH/8)-1:0] M_AXI_WSTRB,
    output wire                         M_AXI_WLAST,
    output wire                         M_AXI_WVALID,
    input  wire                         M_AXI_WREADY,
    
    input  wire [1:0]                   M_AXI_BRESP,
    input  wire                         M_AXI_BVALID,
    output wire                         M_AXI_BREADY,
    
    output wire [C_M_AXI_ADDR_WIDTH-1:0] M_AXI_ARADDR,
    output wire [7:0]                   M_AXI_ARLEN,
    output wire [2:0]                   M_AXI_ARSIZE,
    output wire [1:0]                   M_AXI_ARBURST,
    output wire                         M_AXI_ARLOCK,
    output wire [3:0]                   M_AXI_ARCACHE,
    output wire [2:0]                   M_AXI_ARPROT,
    output wire [3:0]                   M_AXI_ARQOS,
    output wire                         M_AXI_ARVALID,
    input  wire                         M_AXI_ARREADY,
    
    input  wire [C_M_AXI_DATA_WIDTH-1:0] M_AXI_RDATA,
    input  wire [1:0]                   M_AXI_RRESP,
    input  wire                         M_AXI_RLAST,
    input  wire                         M_AXI_RVALID,
    output wire                         M_AXI_RREADY
);

    // Stub: tie-offs for all ports
    assign input_addr   = 32'd0;
    assign input_ready  = 1'b1;
    assign weight_addr  = 32'd0;
    assign weight_ready = 1'b1;
    assign output_addr  = 32'd0;
    assign output_data  = 32'd0;
    assign output_valid = 1'b0;
    
    assign M_AXI_AWADDR = {C_M_AXI_ADDR_WIDTH{1'b0}};
    assign M_AXI_AWLEN  = 8'd0;
    assign M_AXI_AWSIZE = 3'd0;
    assign M_AXI_AWBURST= 2'd1;
    assign M_AXI_AWLOCK = 1'b0;
    assign M_AXI_AWCACHE= 4'd0;
    assign M_AXI_AWPROT = 3'd0;
    assign M_AXI_AWQOS  = 4'd0;
    assign M_AXI_AWVALID= 1'b0;
    
    assign M_AXI_WDATA  = {C_M_AXI_DATA_WIDTH{1'b0}};
    assign M_AXI_WSTRB  = {(C_M_AXI_DATA_WIDTH/8){1'b0}};
    assign M_AXI_WLAST  = 1'b0;
    assign M_AXI_WVALID = 1'b0;
    
    assign M_AXI_BREADY = 1'b1;
    
    assign M_AXI_ARADDR = {C_M_AXI_ADDR_WIDTH{1'b0}};
    assign M_AXI_ARLEN  = 8'd0;
    assign M_AXI_ARSIZE = 3'd0;
    assign M_AXI_ARBURST= 2'd1;
    assign M_AXI_ARLOCK = 1'b0;
    assign M_AXI_ARCACHE= 4'd0;
    assign M_AXI_ARPROT = 3'd0;
    assign M_AXI_ARQOS  = 4'd0;
    assign M_AXI_ARVALID= 1'b0;
    
    assign M_AXI_RREADY = 1'b1;

endmodule
