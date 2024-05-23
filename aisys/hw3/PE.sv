module PE(
	// interface to system
    input wire clk,
    input wire reset,
    input wire c_en,                              // compute enable
    input wire p_en,                              // preload enable
    // interface to PE row .....

    input wire signed[31:0]in_weight,             // wire from weight buffer direction
    input wire signed[31:0]in_input,              // wire from input buffer direction
    input wire signed[31:0]in_preload,

    output wire signed[31:0]flow_output,
    output wire signed[31:0]flow_weight,
    output wire signed[31:0]result
	);


reg signed [31:0] accu_sum_reg;
reg signed [31:0] flow_weight_reg;
reg signed [31:0] flow_output_reg;

assign result=accu_sum_reg;
assign flow_weight=flow_weight_reg;
assign flow_output=flow_output_reg;

always@ (posedge clk) begin
    if (!reset)begin
        flow_output_reg<=0;
        flow_weight_reg<=0;
        accu_sum_reg<=0;
    end
    else begin
        flow_output_reg<=in_input;
        flow_weight_reg<=in_weight;
        if(p_en) accu_sum_reg<=in_preload;
        if(c_en) accu_sum_reg<=accu_sum_reg+in_input*in_weight;
    end
end


endmodule