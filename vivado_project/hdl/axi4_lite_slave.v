module axi4_lite_slave #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 32
) (
    input  wire                         S_AXI_ACLK,
    input  wire                         S_AXI_ARESETN,
    
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  wire [2:0]                   S_AXI_AWPROT,
    input  wire                         S_AXI_AWVALID,
    output wire                         S_AXI_AWREADY,
    
    input  wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire                         S_AXI_WVALID,
    output wire                         S_AXI_WREADY,
    
    output wire [1:0]                   S_AXI_BRESP,
    output wire                         S_AXI_BVALID,
    input  wire                         S_AXI_BREADY,
    
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  wire [2:0]                   S_AXI_ARPROT,
    input  wire                         S_AXI_ARVALID,
    output wire                         S_AXI_ARREADY,
    
    output wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output wire [1:0]                   S_AXI_RRESP,
    output wire                         S_AXI_RVALID,
    input  wire                         S_AXI_RREADY
);

    // Stub: tie-offs for all outputs
    assign S_AXI_AWREADY = 1'b1;
    assign S_AXI_WREADY  = 1'b1;
    assign S_AXI_BRESP   = 2'b00;
    assign S_AXI_BVALID  = 1'b0;
    assign S_AXI_ARREADY = 1'b1;
    assign S_AXI_RDATA   = {C_S_AXI_DATA_WIDTH{1'b0}};
    assign S_AXI_RRESP   = 2'b00;
    assign S_AXI_RVALID  = 1'b0;

endmodule
