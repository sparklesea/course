// input buffer, ref: shifter_buffer
//10kb
module instruction_buffer#(parameter depth = 256)
(
    output reg  [27:0] q,
    input  wire         clk,
    input  wire         reset,
    input  wire         ren,
    input  wire [7:0]  a
    );



    integer i;
    integer j;
    reg [27:0] mem [depth-1:0];
    always @(posedge clk) begin
        if(~reset)begin
            for (i = 0; i < depth; i = i + 1)
                mem[i] <= 28'b0;
        end
        else if(ren) begin
            q <= mem[a];
        end else begin
            q <= 28'd0;
        end
    end

endmodule