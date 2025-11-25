module axi4_master #(
    parameter integer C_M_AXI_DATA_WIDTH = 64,
    parameter integer C_M_AXI_ADDR_WIDTH = 32
) (
    input  wire                         M_AXI_ACLK,
    input  wire                         M_AXI_ARESETN,
    
    // Memory interface signals
    output wire [C_M_AXI_ADDR_WIDTH-1:0] mem_addr,
    output wire [C_M_AXI_DATA_WIDTH-1:0] mem_wdata,
    output wire [C_M_AXI_DATA_WIDTH-1:0] mem_rdata,
    output wire [(C_M_AXI_DATA_WIDTH/8)-1:0] mem_wstrb,
    output wire                         mem_valid,
    output wire                         mem_ready,
    output wire                         mem_we,
    
    // AXI4 Master interface
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

    // Stub: tie-offs for all outputs
    assign mem_addr   = {C_M_AXI_ADDR_WIDTH{1'b0}};
    assign mem_wdata  = {C_M_AXI_DATA_WIDTH{1'b0}};
    assign mem_rdata  = {C_M_AXI_DATA_WIDTH{1'b0}};
    assign mem_wstrb  = {(C_M_AXI_DATA_WIDTH/8){1'b0}};
    assign mem_valid  = 1'b0;
    assign mem_ready  = 1'b1;
    assign mem_we     = 1'b0;
    
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
